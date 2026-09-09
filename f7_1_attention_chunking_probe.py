"""
f7_1_attention_chunking_probe.py
=================================
Phase F7-1: Attention Sequence Chunking -- Experimental Pilot Runner.

Authorized by: Director Resolution 2026-09-09 (conditional on hardening).
Workload: 1280x720, 33 frames, 5 exploratory steps, adaptive mode, seed 42,
guidance 5.0, chunk size C=2048.

HARDENING (per resolution Art. 3):
  * No fabricated results. Every number printed/reported is tagged as one of:
      - MEASURED  : read from instrumentation during this run.
      - OBSERVED  : sampled telemetry (NVML, timings).
      - DERIVED   : computed from measured/observed values.
      - HYPOTHESIS: expectation, never presented as a result.
  * The dry-run performs NO inference and reports NO results.
  * A numerical equivalence check between chunked and monolithic attention is
    MEASURED (not asserted bit-exact) on synthetic tensors of the real dtype.
  * Real runtime attention facts (backend, S, heads, head_dim, dtype, variant)
    are logged from the interception point (see chunked_attention.py).

GOVERNANCE (resolution Art. 5/6/10):
  Only this file and marley/ops/chunked_attention.py are modified. The runtime
  (MarleyEndToEndPipeline, streamers, AdaptiveEngine, VAE, diffusers source,
  allocator, F6/F7-0) is NOT modified. No opportunistic optimization.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline
from marley.ops.chunked_attention import (
    NumericalIntegrityError,
    SafeAttentionAbort,
    apply_chunked_attention_to_dit_blocks,
)

HARD_GATE_VRAM_MB = 4800.0
SAFE_ABORT_NVML_MB = 5900.0          # operational safety near 6 GB physical
RESERVED_LOG_MB = 4800.0
F7_0_OVERALL_CADENCE_S = 68.75       # plan-declared F7-0 baseline (overall)
F7_D1_STEPS2_5_CADENCE_S = 59.93     # measured F7-D1 regime, steps 2..5
GATE_B_CADENCE_S = 90.0              # absolute cadence abort threshold

CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
CANONICAL_FPS = 16
F7_0_BASELINE_NVML_MB = 6058.5
F7_D1_NVML_MB = 6088.4
F7_0_RESERVED_MB = 5894.0
F7_0_ALLOC_MB = 2187.5


def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class ContinuousNVMLSampler:
    """High-frequency (25 ms) hardware VRAM sampler (OBSERVED telemetry)."""

    def __init__(self, device_index: int = 0, interval_ms: float = 25.0) -> None:
        self.interval_s = interval_ms / 1000.0
        self.device_index = device_index
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        self._handle = None
        self.peak_mb = 0.0
        self.peak_timestamp: Optional[str] = None
        self._lock = threading.Lock()

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_available = True
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.peak_mb = info.used / (1024 * 1024)
            self.peak_timestamp = datetime.datetime.now().isoformat()
        except Exception as exc:
            print(f"[WARN] NVML sampler init failed: {exc}", file=sys.stderr)

    def start(self) -> None:
        if not self._nvml_available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import pynvml
        while self._running:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mb = info.used / (1024 * 1024)
                now_str = datetime.datetime.now().isoformat()
                with self._lock:
                    if used_mb > self.peak_mb:
                        self.peak_mb = used_mb
                        self.peak_timestamp = now_str
            except Exception:
                pass
            time.sleep(self.interval_s)

    def current_used_mb(self) -> Optional[float]:
        if not self._nvml_available or self._handle is None:
            return None
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024 * 1024)
        except Exception:
            return None

    def sample_instant(self) -> Dict[str, Any]:
        out = {"nvml_used_mb": 0.0, "gpu_temp_c": None, "gpu_clock_mhz": None, "gpu_util_pct": None}
        if not self._nvml_available or self._handle is None:
            return out
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            out["nvml_used_mb"] = round(info.used / (1024 * 1024), 2)
            out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            out["gpu_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            out["gpu_util_pct"] = util.gpu
        except Exception:
            pass
        return out

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb


def measure_attention_equivalence(
    seq_len: int,
    heads: int,
    head_dim: int,
    chunk_size: int,
    dtype: torch.dtype,
    batch: int = 1,
) -> Dict[str, Any]:
    """
    MEASURED numerical equivalence between chunked-query and monolithic
    attention over synthetic tensors of the real dtype. NOT bit-exact by
    assumption; max-abs-diff and cosine similarity are measured and returned.
    """
    from diffusers.models.transformers.transformer_wan import dispatch_attention_fn

    torch.manual_seed(7)
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)

    # Monolithic reference (as the unmodified runtime would dispatch it).
    ref = dispatch_attention_fn(q, k, v, None, 0.0, False, backend=None, parallel_config=None)

    # Chunked query path (identical to the processor's per-chunk dispatch).
    pieces = []
    for q_chunk in q.split(chunk_size, dim=1):
        pieces.append(
            dispatch_attention_fn(q_chunk, k, v, None, 0.0, False, backend=None, parallel_config=None)
        )
    chunked = torch.cat(pieces, dim=1)

    torch.cuda.synchronize()
    flat_ref = ref.reshape(-1).float()
    flat_chunked = chunked.reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(flat_ref, flat_chunked, dim=0).item())
    max_abs = float((ref.float() - chunked.float()).abs().max().item())
    denom = float(flat_ref.abs().max().item()) + 1e-9
    rel_max = max_abs / denom
    return {
        "seq_len": int(seq_len),
        "heads": int(heads),
        "head_dim": int(head_dim),
        "chunk_size": int(chunk_size),
        "dtype": str(dtype),
        "batch": int(batch),
        "max_abs_diff_MEASURED": max_abs,
        "relative_max_diff_MEASURED": rel_max,
        "cosine_similarity_MEASURED": cos,
        "bit_exact": bool(max_abs == 0.0),
    }


def resolve_attention_dims(pipeline) -> Tuple[int, int]:
    """Read heads / head_dim from the loaded transformer config (no forward)."""
    try:
        cfg = pipeline.transformer.config
    except Exception:
        cfg = None
    heads, head_dim = None, None
    if isinstance(cfg, dict):
        heads = cfg.get("num_attention_heads")
        hd = cfg.get("attention_head_dim")
        if hd is not None and not isinstance(hd, int) and isinstance(hd, (list, tuple)):
            hd = hd[0]
        head_dim = hd
    elif cfg is not None:
        heads = getattr(cfg, "num_attention_heads", None)
        hd = getattr(cfg, "attention_head_dim", None)
        if hd is not None and not isinstance(hd, int):
            try:
                hd = hd[0]
            except Exception:
                pass
        head_dim = hd
    # Fallback known Wan2.1-T2V-1.3B defaults if config not yet resolvable.
    if heads is None or head_dim is None:
        try:
            block = pipeline.blocks[0]
            attn = getattr(block, "attn1", None)
            if attn is not None:
                heads = int(attn.heads)
        except Exception:
            heads = None
        if head_dim is None:
            try:
                qkv = getattr(pipeline.blocks[0].attn1, "qkv", None)
                inner = getattr(qkv, "out_features", None)
                if inner is not None and heads:
                    head_dim = inner // heads
            except Exception:
                head_dim = None
    return (heads or 24, head_dim or 128)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Runtime Phase F7-1 -- Attention Chunking Pilot")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--mode", type=str, default="adaptive", choices=["sync", "async_fp16", "async_int8", "adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eq-seq-len", type=int, default=8192, help="Synthetic seq length for equivalence check")
    p.add_argument("--output", type=str, default="logs/f7_1_chunked_attention_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_1_ATTENTION_CHUNKING_PILOT_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true", help="Validate params/env WITHOUT inference or results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1
    grid_seq = latent_h * latent_w

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-1: ATTENTION SEQUENCE CHUNKING PILOT")
    print("  (Hardening-compliant. Results tagged MEASURED/OBSERVED/DERIVED/HYPOTHESIS)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: {args.steps}")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Guidance: {args.guidance}")
    print(f"  Chunk Size C: {args.chunk_size} | Grid tokens (spatial plane): {grid_seq}")
    print(f"  Hard Gate NVML: <= {HARD_GATE_VRAM_MB:.0f} MB | Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB")

    if args.dry_run:
        print("\n[DRY-RUN] Verificacion de parametros y entorno. SIN inferencia, SIN resultados.")
        print(f"  Latent shape esperado: [1, 16, {latent_f}, {latent_h}, {latent_w}]")
        print(f"  Grid espacial: {latent_h}x{latent_w} = {grid_seq}")
        print("  Nota: la longitud de secuencia REAL de self-attention se medira en runtime")
        print("        (puede incluir el eje temporal) y NO se asume igual a este grid.")
        print("  Primitivas a usar: ChunkedWanAttnProcessor sobre attn1/attn2, pipeline.generate().")
        print("[DRY-RUN] Parametros validados. Preparado para ejecucion real.")
        return

    sampler = ContinuousNVMLSampler(device_index=0, interval_ms=25.0)
    sampler.start()

    init_ram = get_process_ram_mb()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    metrics = None
    video_tensor = None
    error_occurred: Optional[str] = None
    abort_kind: Optional[str] = None
    event_log: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    try:
        pipeline = MarleyEndToEndPipeline(
            device=args.device,
            dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16,
            text_dtype=torch.bfloat16,
        )

        heads, head_dim = resolve_attention_dims(pipeline)
        print(f"\n>> DIMS (DERIVED from config): heads={heads}, head_dim={head_dim}")

        # Numerical equivalence check -- MEASURED, on synthetic tensors of real dtype.
        print(f"\n>> Measuring attention equivalence (chunked vs monolithic), C={args.chunk_size}...")
        eq = measure_attention_equivalence(
            seq_len=args.eq_seq_len,
            heads=heads,
            head_dim=head_dim,
            chunk_size=args.chunk_size,
            dtype=torch.float16,
        )
        print(f"   [MEASURED] max_abs_diff={eq['max_abs_diff_MEASURED']:.3e} "
              f"rel={eq['relative_max_diff_MEASURED']:.3e} "
              f"cosine={eq['cosine_similarity_MEASURED']:.6f} "
              f"bit_exact={eq['bit_exact']}")

        # Apply instrumented chunked attention to every DiT block.
        print(f"\n>> Applying ChunkedWanAttnProcessor (C={args.chunk_size}) to DiT blocks (instrumented)...")
        processors = apply_chunked_attention_to_dit_blocks(
            pipeline.blocks,
            chunk_size=args.chunk_size,
            enabled=True,
            instrument=True,
            event_log=event_log,
            nvml_used_mb_fn=sampler.current_used_mb,
            safe_abort_nvml_mb=SAFE_ABORT_NVML_MB,
            reserved_log_mb=RESERVED_LOG_MB,
        )
        print(f"   [OK] {len(processors)} processors attached (2 per block).")

        print(f"\n>> Executing F7-1 720p inference ({args.steps} steps)...")
        video_tensor, metrics = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            mode=args.mode,
            output_video_path=None,
            nvml_sampler=sampler,
        )
    except NumericalIntegrityError as exc:
        abort_kind = "GATE_A_NUMERICAL"
        error_occurred = str(exc)
        print(f"\n[GATE A ABORT] {error_occurred}")
    except SafeAttentionAbort as exc:
        abort_kind = "GATE_C_SAFE_OPERATIONAL"
        error_occurred = str(exc)
        print(f"\n[SAFE ABORT OPERATIVO] {error_occurred}")
    except Exception as exc:
        abort_kind = "RUNTIME_ERROR"
        error_occurred = str(exc)
        print(f"\n[F7-1 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = sampler.stop()

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram = get_process_ram_mb()

    # Cadence references.
    step_times = list(metrics.step_times_s) if (metrics and metrics.step_times_s) else []
    denoise_total = metrics.denoise_time_s if metrics else 0.0
    steps2_5 = step_times[1:] if len(step_times) >= 2 else []
    cadence_steps2_5 = (sum(steps2_5) / len(steps2_5)) if steps2_5 else None
    overhead_vs_68_75 = ((cadence_steps2_5 / F7_0_OVERALL_CADENCE_S) - 1.0) * 100.0 if cadence_steps2_5 else None

    # Gate B evaluation (OBSERVED cadence, evaluated post-run per two-file scope).
    gate_b_fail = bool(cadence_steps2_5 is not None and cadence_steps2_5 > GATE_B_CADENCE_S)

    # Integrity / container.
    nan_inf = False
    mp4_ok = False
    output_video_path = metrics.output_video_path if metrics else None
    if metrics:
        nan_inf = bool(metrics.nan_inf_detected)
        if metrics.output_video_path and os.path.exists(metrics.output_video_path):
            mp4_ok = os.path.getsize(metrics.output_video_path) > 0

    completion_ok = (abort_kind is None)
    memory_gate_pass = completion_ok and (peak_nvml <= HARD_GATE_VRAM_MB)

    # Result classification A/B/C/D (DERIVED from measured values).
    reduction_mb = F7_0_BASELINE_NVML_MB - peak_nvml
    result_note = ""
    if abort_kind == "GATE_A_NUMERICAL":
        result_class = "D"  # integrity failure
        result_note = "falla de integridad numerica"
    elif memory_gate_pass and completion_ok:
        result_class = "A"
        result_note = "Hard Gate de memoria cumplido"
    elif abort_kind == "GATE_C_SAFE_OPERATIONAL":
        # Operational safe-stop: memory remained in the F7-0 regime (>> 4800 MB).
        result_class = "C"
        result_note = "sin efecto material sobre el Peak NVML (abortado por salvaguarda operativa cercana al limite fisico de 6 GB)"
    elif reduction_mb > 200.0:
        result_class = "B"
        result_note = "reduccion significativa pero insuficiente para el Hard Gate"
    elif completion_ok:
        result_class = "C"
        result_note = "sin efecto material sobre el Peak NVML"
    else:
        result_class = "D"
        result_note = "corrida incompleta / error de ejecucion"

    print("\n" + "=" * 80)
    print("  PHASE F7-1: SCORECARD")
    print("=" * 80)
    print(f"  Peak NVML fisico       [OBSERVED] : {peak_nvml:7.1f} MB (Gate<=4800: {'PASS' if memory_gate_pass else 'FAIL'})")
    print(f"  Reduccion vs F7-0      [DERIVED]  : {reduction_mb:+7.1f} MB")
    print(f"  PyTorch Allocated peak [OBSERVED] : {peak_alloc:7.1f} MB (F7-0: {F7_0_ALLOC_MB})")
    print(f"  PyTorch Reserved peak  [OBSERVED] : {peak_reserved:7.1f} MB (F7-0: {F7_0_RESERVED_MB})")
    print(f"  Host RSS               [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total       [OBSERVED] : {t_total:7.2f} s ({t_total/60:.2f} min)")
    if cadence_steps2_5 is not None:
        print(f"  Cadencia steps 2-5     [OBSERVED] : {cadence_steps2_5:7.2f} s/step (Gate B 90 s: {'FAIL' if gate_b_fail else 'ok'})")
    print(f"  NaN/Inf                [MEASURED] : {'PASS' if not nan_inf else 'FAIL'}")
    print(f"  MP4 valido             [OBSERVED] : {'PASS' if mp4_ok else 'n/a'}")
    print(f"  Resultado clasificado  [DERIVED]  : {result_class} ({result_note})")
    print(f"  Abort                   : {abort_kind or 'none'}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-1 (Attention Sequence Chunking Pilot)",
        "evidence_class": "All measured/observed; hypotheses labelled",
        "workload": {
            "resolution": f"{args.width}x{args.height}",
            "frames": args.frames,
            "steps": args.steps,
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "guidance_scale": args.guidance,
        },
        "numerical_equivalence": eq,
        "attention_observed": [e for e in event_log if e.get("event") == "attention_observed"],
        "attention_alloc_delta": [e for e in event_log if e.get("event") == "attention_alloc_delta"],
        "reserved_events": [e for e in event_log if e.get("event") == "reserved_above_log_threshold"],
        "timings": {
            "total_wall_clock_s": round(t_total, 2),
            "denoise_total_s": round(denoise_total, 2),
            "step_times_s": [round(t, 2) for t in step_times],
            "cadence_steps2_5_s": round(cadence_steps2_5, 2) if cadence_steps2_5 else None,
            "vae_decode_s": round(metrics.vae_decode_time_s, 2) if metrics else None,
        },
        "memory_mb": {
            "peak_nvml_used": round(peak_nvml, 1),
            "peak_nvml_timestamp": sampler.peak_timestamp,
            "peak_torch_alloc": round(peak_alloc, 1),
            "peak_torch_reserved": round(peak_reserved, 1),
            "allocator_delta_mb": round(peak_reserved - peak_alloc, 1),
            "peak_process_ram": round(peak_ram, 1),
        },
        "comparison_vs_baselines": {
            "f7_0_baseline_nvml_mb": F7_0_BASELINE_NVML_MB,
            "f7_d1_nvml_mb": F7_D1_NVML_MB,
            "nvml_reduction_vs_f7_0_mb": round(reduction_mb, 1),
            "f7_0_reserved_mb": F7_0_RESERVED_MB,
            "f7_0_alloc_mb": F7_0_ALLOC_MB,
            "cadence_ref_overall_68_75_s": F7_0_OVERALL_CADENCE_S,
            "cadence_ref_steps2_5_59_93_s": F7_D1_STEPS2_5_CADENCE_S,
            "overhead_pct_vs_68_75": round(overhead_vs_68_75, 2) if overhead_vs_68_75 is not None else None,
        },
        "gates": {
            "gate_A_numerical_ok": not nan_inf,
            "gate_B_cadence_ok": not gate_b_fail,
            "gate_C_memory_pass": memory_gate_pass,
            "safe_abort_nvml_mb": SAFE_ABORT_NVML_MB,
        },
        "verdict": {
            "execution_completed": completion_ok,
            "abort_kind": abort_kind,
            "memory_hard_gate_pass": memory_gate_pass,
            "result_class": result_class,
            "result_note": result_note,
            "output_video": output_video_path if mp4_ok else None,
            "error_detail": error_occurred,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    # ---- Report (honest labels, measured/observed/derived/hypothesis) ----
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# Informe de Piloto Experimental: Phase F7-1 (Attention Sequence Chunking)\n\n")
        w(f"**Fecha:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Resolucion:** {args.width}x{args.height} @ {args.frames} frames ({args.steps} pasos), modo `{args.mode}`  \n")
        w(f"**Chunk Size:** C = {args.chunk_size}  \n\n")
        w("**Clasificacion de evidencia:** `MEASURED` = instrumentado en esta corrida; `OBSERVED` = telemetria muestreada; "
          "`DERIVED` = calculado; `HYPOTHESIS` = expectativa no demostrada.\n\n")
        w("---\n\n## 1. Hechos Medidos / Observados\n\n")
        w("| Metrica | Clase | Valor | Baseline F7-0 |\n")
        w("| :--- | :---: | :---: | :---: |\n")
        w(f"| Peak NVML fisico | OBSERVED | {peak_nvml:.1f} MB | {F7_0_BASELINE_NVML_MB:.1f} MB |\n")
        w(f"| PyTorch Allocated | OBSERVED | {peak_alloc:.1f} MB | {F7_0_ALLOC_MB:.1f} MB |\n")
        w(f"| PyTorch Reserved | OBSERVED | {peak_reserved:.1f} MB | {F7_0_RESERVED_MB:.1f} MB |\n")
        w(f"| Host RSS | OBSERVED | {peak_ram:.1f} MB | ~2,150 MB |\n")
        w(f"| Wall-clock total | OBSERVED | {t_total:.2f} s | 521.93 s |\n")
        w(f"| Cadencia steps 2-5 | OBSERVED | "
          f"{cadence_steps2_5:.2f} s/step" if cadence_steps2_5 is not None else "n/a (aborto antes de completar 2 pasos)"
          + " | " + f"{F7_D1_STEPS2_5_CADENCE_S:.2f} s/step (F7-D1) |\n")
        w(f"| NaN/Inf | MEASURED | {'0 NaNs / 0 Infs' if not nan_inf else 'NaN/Inf presente'} | 0 |\n")
        w(f"| MP4 valido | OBSERVED | {'Si' if mp4_ok else 'No/No generado'} | Si |\n\n")
        w("### Equivalencia numerica chunked vs monolitico (MEASURED, sintetico)\n\n")
        w(f"* Max abs diff: {eq['max_abs_diff_MEASURED']:.3e}\n")
        w(f"* Diff relativa: {eq['relative_max_diff_MEASURED']:.3e}\n")
        w(f"* Cosine similarity: {eq['cosine_similarity_MEASURED']:.6f}\n")
        w(f"* Bit-exacta: {eq['bit_exact']}  (NO se afirma equivalencia bit-exacta)\n")
        w(f"* Config: seq={eq['seq_len']}, heads={eq['heads']}, head_dim={eq['head_dim']}, "
          f"chunk={eq['chunk_size']}, dtype={eq['dtype']}\n\n")
        w("### Atencion interceptada en runtime (MEASURED)\n\n")
        obs = [e for e in event_log if e.get("event") == "attention_observed"]
        if obs:
            w("| Tag | Variante | S real | Heads | Head_dim | dtype | Backend cfg | Chunked |\n")
            w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for e in obs:
                w(f"| {e['tag']} | {e['variant']} | {e['seq_len_S']} | {e['heads']} | {e['head_dim']} | "
                  f"{e['dtype']} | {e['configured_backend']} | {e['chunked']} |\n")
        else:
            w("_No se registro ninguna atencion interceptada._\n")
        w("\n---\n\n## 2. Comparaciones (DERIVED)\n\n")
        w(f"* Reduccion Peak NVML vs F7-0: **{reduction_mb:+.1f} MB**\n")
        if overhead_vs_68_75 is not None:
            w(f"* Overhead de cadencia steps 2-5 vs 68.75 s (plan): {overhead_vs_68_75:+.1f}%\n")
        w(f"* Resultado clasificado: **{result_class}** ({result_note})\n\n")
        w("---\n\n## 3. Inferencias (DERIVED)\n\n")
        if result_class == "A":
            w("El chunking C=2048 se asocia a un Peak NVML dentro del Hard Gate de 4.800 MB con integridad "
              "numerica y video valido. La hipotesis recibe evidencia favorable preliminar.\n")
        elif result_class == "B":
            w("El Peak NVML disminuye respecto a F7-0 pero permanece por encima de 4.800 MB. Evidencia "
              "favorable pero insuficiente; C=2048 no basta para el gate.\n")
        elif result_class == "C":
            w("La memoria se mantiene en el regimen de F7-0. La hipotesis de que la atencion es el evento "
              "dominante del pico debe reconsiderarse.\n")
        else:
            w("Reduccion con degradacion (numerica, de rendimiento o aborto). El mecanismo no se considera "
              "viable en esta configuracion, o la corrida aborto.\n")
        w("\n---\n\n## 4. Hipotesis (NO demostradas)\n\n")
        w("* Que el chunking ataca el evento temporal que produce la expansion del pool del allocator.\n")
        w("* Que S = grid espacial 14,400 coincide con la secuencia real de self-attention (a verificar por la "
          "telemetria de atencion medida).\n")
        w("\n---\n\n## 5. Recomendacion (para decision del Director)\n\n")
        w("El piloto NO autoriza por si mismo cambios permanentes al runtime. Se remite a la decision del "
          "Director: integrar C=2048, repetir, cambiar chunk, abandonar linea o investigar otra causa.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
