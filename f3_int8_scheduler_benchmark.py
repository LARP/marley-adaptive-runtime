"""
f3_int8_scheduler_benchmark.py
==============================
Phase F3 + INT8 — Isolation Benchmark (transfer-volume reduction)

Tests the hypothesis from docs/F3_INT8_EXPERIMENT_PLAN_01.md on real Wan2.1 DiT blocks:
reduce the H2D payload per DiT block via INT8 linear projections (F1.7: 88.6 MB -> 45.9 MB,
-48.2%) while KEEPING the proven BudgetedAsyncStreamer double-buffer scheduler and FP16
compute. The scheduler race-condition protocol is inherited unchanged from
marley/ops/async_stream.py via INT8BudgetedStreamer (marley/ops/async_stream_int8.py).

Conditions (all using identical Marley execution infrastructure):
  Sync  (A)  — INT8 payload, synchronous per-block transfer.
  Async (B)  — INT8 payload, double-buffered async prefetch.
Frozen FP16 baseline reference (certified F3): Sync 18,127 ms / Async 16,824 ms (+7.2%).

Reported in addition to the 7-metric scorecard:
  - INT8 vs FP16 H2D payload bytes per block (byte-volume reduction %)
  - on-device dequantize fidelity (cosine similarity, 0 NaNs/Infs)

Kill Gates (same as F3; worst-case over reps):
  Peak VRAM > 4,800 MB  -> ABORT
  Effective overlap < 5% -> FAIL
  Wall-clock speedup < 0% -> FAIL (async slower than sync)
  NaN/Inf in any output  -> ABORT

Measurement-integrity (Enmiendas 1-3) preserved: external wall-clock = official metric,
alternating A/B order with discarded warmup, real overlap/stall via CUDA events.

Usage
-----
  python f3_int8_scheduler_benchmark.py
  python f3_int8_scheduler_benchmark.py --steps 5 --reps 3
  python f3_int8_scheduler_benchmark.py --steps 5 --reps 5 --no-warmup

Output
------
  logs/f3_int8_scheduler_benchmark.json
  Console human-readable report
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

KILL_GATE_VRAM_MB = 4800.0
KILL_GATE_OVERLAP_MIN_PCT = 5.0


class NVMLSampler:
    """Background thread polling physical GPU VRAM at 50 ms; tracks peak."""

    def __init__(self, device_index: int = 0, interval_ms: float = 50.0) -> None:
        self.interval_s = interval_ms / 1000.0
        self._peak_mb = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self._nvml_available = True
        except Exception as e:
            print(f"[NVML] pynvml unavailable: {e}. VRAM via torch.cuda.")
            self._handle = None
            self._pynvml = None

    def _sample_mb(self) -> float:
        if self._nvml_available:
            try:
                return self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 ** 2)
            except Exception:
                pass
        return torch.cuda.memory_reserved(0) / (1024 ** 2)

    def start(self) -> None:
        self._peak_mb = self._sample_mb()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            mb = self._sample_mb()
            if mb > self._peak_mb:
                self._peak_mb = mb
            time.sleep(self.interval_s)

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_mb


def check_kill_gates(
    peak_vram_mb: float,
    nan_inf: bool,
    overlap_pct: Optional[float] = None,
    speedup_pct: Optional[float] = None,
    stage: str = "",
) -> bool:
    passed = True
    if nan_inf:
        print(f"[KILL GATE] x {stage}: NaN/Inf detected in outputs. ABORT.")
        passed = False
    if peak_vram_mb > KILL_GATE_VRAM_MB:
        print(f"[KILL GATE] x {stage}: Peak VRAM {peak_vram_mb:.0f} MB > {KILL_GATE_VRAM_MB:.0f} MB. ABORT.")
        passed = False
    if overlap_pct is not None and overlap_pct < KILL_GATE_OVERLAP_MIN_PCT:
        print(f"[KILL GATE] x {stage}: Effective overlap {overlap_pct:.1f}% < {KILL_GATE_OVERLAP_MIN_PCT:.0f}%.")
        passed = False
    if speedup_pct is not None and speedup_pct < 0.0:
        print(f"[KILL GATE] x {stage}: Wall-clock speedup {speedup_pct:.1f}% < 0% (Async slower than Sync).")
        passed = False
    return passed


def summarize(values: List[float]) -> Dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    mean = statistics.fmean(values) if hasattr(statistics, "fmean") else sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "min": min(values), "max": max(values), "std": std}


def measure_dequant_fidelity(streamer: "INT8BudgetedStreamer", device: torch.device) -> Dict:
    """
    Compare on-device dequantized weights against the original FP16 weights of block 0.
    Returns cosine similarity + NaN/Inf + max abs error (averaged over INT8 layers).
    """
    import torch.nn.functional as F
    cos_sims, rmse, nan_inf = [], [], False
    orig_fp16 = {n: p.detach().cpu().half() for n, p in streamer.blocks[0].named_parameters()}
    slot0 = streamer.gpu_slots[0]
    for name in streamer._q2d_names[0]:
        deq = slot0[name].detach().cpu().half()
        ref = orig_fp16[name]
        if ref.numel() != deq.numel():
            continue
        ref_f = ref.flatten().double()
        deq_f = deq.flatten().double()
        if torch.isnan(deq_f).any() or torch.isinf(deq_f).any():
            nan_inf = True
            continue
        cos = F.cosine_similarity(ref_f, deq_f, dim=0).clamp_max(1.0).item()
        rm = torch.sqrt(torch.mean((ref_f - deq_f) ** 2)).item()
        cos_sims.append(cos)
        rmse.append(rm)
    return {
        "layers_checked": len(cos_sims),
        "mean_cosine_similarity": (sum(cos_sims) / len(cos_sims)) if cos_sims else None,
        "min_cosine_similarity": min(cos_sims) if cos_sims else None,
        "mean_rmse": (sum(rmse) / len(rmse)) if rmse else None,
        "nan_inf": nan_inf,
    }


def run_benchmark(args: argparse.Namespace) -> Dict:
    print("=" * 70)
    print("  Marley Runtime - Phase F3 + INT8 Benchmark")
    print("  Isolation: transfer-volume reduction, scheduler unchanged")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device found. INT8 benchmark requires a CUDA GPU.")
        sys.exit(1)

    device = torch.device("cuda", 0)
    dtype = torch.float16
    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"\nDevice : {device_name}")
    print(f"VRAM   : {vram_total_mb:.0f} MB total")
    print(f"Steps  : {args.steps}  Reps: {args.reps}  Frames: {args.frames}")
    print()

    print("[Setup] Loading Wan2.1 transformer blocks (CPU, FP16)...")
    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
        transformer = WanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            subfolder="transformer",
            torch_dtype=dtype,
        )
        transformer = transformer.to("cpu")
        torch.cuda.empty_cache()
        blocks = list(transformer.blocks)
        num_blocks = len(blocks)
        print(f"[Setup] Loaded {num_blocks} DiT blocks on CPU.")
    except Exception as e:
        print(f"[ERROR] Failed to load Wan2.1 transformer: {e}")
        sys.exit(1)

    print("[Setup] Initializing INT8BudgetedStreamer (quantize -> pin -> GPU slots)...")
    from marley.ops.async_stream_int8 import INT8BudgetedStreamer
    streamer = INT8BudgetedStreamer(blocks=blocks, device=device, dtype=dtype)

    payload_int8 = streamer.host_payload_bytes(0)
    payload_fp16 = streamer.fp16_equivalent_bytes(0)
    byte_reduction_pct = (payload_fp16 - payload_int8) / payload_fp16 * 100.0
    print(f"[Setup] Per-block H2D payload  FP16: {payload_fp16/1e6:.2f} MB | "
          f"INT8: {payload_int8/1e6:.2f} MB | reduction -{byte_reduction_pct:.1f}%")

    print("[Setup] Probing block input shapes (CPU)...")
    hidden_states, encoder_hidden_states, temb, rotary_emb = streamer.probe_shapes(
        transformer=transformer,
        latent_shape=(1, 16, 5, 60, 104),
        text_len=226,
        text_dim=4096,
    )

    nvml = NVMLSampler(device_index=0, interval_ms=50.0)

    # Stage block 0 into slot 0 so the fidelity check reads real dequantized weights.
    streamer._load_block_sync(0, slot=0)

    # On-device dequantize fidelity check (block 0 already staged into slot 0).
    print("\n[Fidelity] On-device INT8 dequantize vs original FP16 (block 0)...")
    fidelity = measure_dequant_fidelity(streamer, device)
    print(f"[Fidelity] layers={fidelity['layers_checked']} cos(mean)="
          f"{fidelity['mean_cosine_similarity']:.6f} cos(min)="
          f"{fidelity['min_cosine_similarity']:.6f} NaN/Inf={fidelity['nan_inf']}")

    def _run_condition(mode: str, h_in: torch.Tensor) -> Tuple[object, float]:
        if mode == "sync":
            fn = streamer.execute_sync_loop
        else:
            fn = streamer.execute_async_loop
        nvml.start()
        t0 = time.perf_counter()
        mm = fn(
            h_in, encoder_hidden_states, temb, rotary_emb,
            num_steps=args.steps, num_frames=args.frames, nan_check=True,
        )
        wall = time.perf_counter() - t0
        vram = nvml.stop()
        mm.peak_vram_mb = vram
        return mm, wall

    # Warmup (discarded) to stabilize clocks/driver state.
    if args.warmup:
        print("[Warmup] One discarded Sync + Async pass...")
        for mode in ("sync", "async"):
            _run_condition(mode, hidden_states.clone())
        print("[Warmup] Done (results discarded).\n")

    per_rep: List[Dict] = []
    wall_a_s, wall_b_s = [], []
    speedups, overlaps, peaks, nan_any = [], [], [], False
    stalls_total_ms: List[float] = []

    for rep in range(args.reps):
        print(f"--- Rep {rep + 1}/{args.reps} ---")
        h_a = hidden_states.clone()
        h_b = hidden_states.clone()
        do_async_first = (rep % 2 == 1)
        if do_async_first:
            metrics_b, wall_b = _run_condition("async", h_b)
            metrics_a, wall_a = _run_condition("sync", h_a)
            order = "B/A"
        else:
            metrics_a, wall_a = _run_condition("sync", h_a)
            metrics_b, wall_b = _run_condition("async", h_b)
            order = "A/B"

        speedup_pct = ((wall_a - wall_b) / wall_a) * 100.0
        overlap_pct = metrics_b.effective_overlap_pct
        peak_vram = max(metrics_a.peak_vram_mb, metrics_b.peak_vram_mb)
        wall_a_s.append(wall_a); wall_b_s.append(wall_b)
        speedups.append(speedup_pct); overlaps.append(overlap_pct); peaks.append(peak_vram)
        nan_any = nan_any or metrics_a.nan_inf_detected or metrics_b.nan_inf_detected
        stalls_total_ms.append(metrics_b.prefetch_latency_s * 1000.0)

        print(f"  Order {order} | Sync wall {wall_a*1000:8.1f} ms | "
              f"Async wall {wall_b*1000:8.1f} ms | Speedup {speedup_pct:+6.1f}% | "
              f"Overlap {overlap_pct:5.1f}% | Stall {metrics_b.prefetch_latency_s*1000:6.2f} ms | "
              f"Peak VRAM {peak_vram:5.0f} MB | NaN {nan_any}")

        per_rep.append({
            "rep": rep + 1, "order": order, "speedup_pct": speedup_pct,
            "overlap_pct": overlap_pct,
            "wall_a_ms": wall_a * 1000.0, "wall_b_ms": wall_b * 1000.0,
            "internal_a_ms": metrics_a.total_denoise_time_s * 1000.0,
            "internal_b_ms": metrics_b.total_denoise_time_s * 1000.0,
            "h2d_a_ms": metrics_a.total_h2d_transfer_time_s * 1000.0,
            "h2d_b_ms": metrics_b.total_h2d_transfer_time_s * 1000.0,
            "stall_b_ms": metrics_b.prefetch_latency_s * 1000.0,
            "forced_syncs_a": metrics_a.forced_sync_count,
            "forced_syncs_b": metrics_b.forced_sync_count,
            "peak_vram_mb": peak_vram,
            "nan_inf": nan_any,
            "timeline": {
                "per_block_copy_ms": [v * 1000.0 for v in metrics_b.per_block_copy_s],
                "per_block_stall_ms": [v * 1000.0 for v in metrics_b.per_block_stall_s],
                "per_block_compute_ms": [v * 1000.0 for v in metrics_b.per_block_compute_s],
            },
        })

    agg_speedup = summarize(speedups)
    agg_wall_a = summarize(wall_a_s)
    agg_wall_b = summarize(wall_b_s)
    agg_overlap = summarize(overlaps)
    peak_vram = max(peaks)
    min_speedup = min(speedups)
    min_overlap = min(overlaps)
    mean_stall_ms = sum(stalls_total_ms) / len(stalls_total_ms)

    passed = check_kill_gates(
        peak_vram_mb=peak_vram, nan_inf=nan_any,
        overlap_pct=min_overlap, speedup_pct=min_speedup, stage="F3+INT8",
    )

    print()
    print("=" * 70)
    print("  F3+INT8 SCORECARD (external wall-clock, official)")
    print("=" * 70)
    print(f"  {'External wall-clock':<30} {agg_wall_a['mean']*1000:>11.1f} ms   {agg_wall_b['mean']*1000:>11.1f} ms")
    print(f"  {'  (min/max/std)':<30} {agg_wall_a['min']*1000:>8.1f}/{agg_wall_a['max']*1000:<8.1f}ms {agg_wall_b['min']*1000:>8.1f}/{agg_wall_b['max']*1000:<8.1f}ms (sA {agg_wall_a['std']*1000:.0f}, sB {agg_wall_b['std']*1000:.0f})")
    print(f"  {'Effective overlap':<30} {'0.0%':>22} {agg_overlap['mean']:>20.1f}%")
    print(f"  {'  (min/max)':<30} {'':>22} {agg_overlap['min']:>12.1f}%/{agg_overlap['max']:<7.1f}%")
    print(f"  {'Stall total (real ms)':<30} {'0.00':>22} {mean_stall_ms:>19.2f}")
    print(f"  {'Peak VRAM (NVML max)':<30} {'':>22} {peak_vram:>18.0f} MB")
    print(f"  {'Forced syncs (sync/async)':<30} {per_rep[0]['forced_syncs_a']:>11} {per_rep[0]['forced_syncs_b']:>22}")
    print(f"  {'NaN/Inf':<30} {str(nan_any):>22}")
    print(f"  {'Payload reduction INT8':<30} {'':>18} -{byte_reduction_pct:.1f}%")
    print(f"  {'Dequant cos sim (mean)':<30} {'':>18} {fidelity['mean_cosine_similarity']:.6f}" if fidelity['mean_cosine_similarity'] else "")
    print(f"\n  Wall-clock speedup (mean): {agg_speedup['mean']:+.1f}%  "
          f"(min {agg_speedup['min']:+.1f}% / max {agg_speedup['max']:+.1f}% / std {agg_speedup['std']:.1f}%)")
    print(f"  Verdict: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    results = {
        "phase": "F3+INT8",
        "device": device_name,
        "vram_total_mb": vram_total_mb,
        "dtype": str(dtype),
        "num_blocks": num_blocks,
        "num_steps": args.steps,
        "num_frames": args.frames,
        "num_reps": args.reps,
        "warmup": bool(args.warmup),
        "payload_bytes": {
            "fp16_equiv": payload_fp16,
            "int8": payload_int8,
            "reduction_pct": byte_reduction_pct,
        },
        "dequant_fidelity": fidelity,
        "aggregate": {
            "speedup_pct": agg_speedup,
            "wall_clock_sync_ms": {k: v * 1000.0 for k, v in agg_wall_a.items()},
            "wall_clock_async_ms": {k: v * 1000.0 for k, v in agg_wall_b.items()},
            "overlap_pct": agg_overlap,
            "stall_total_ms_mean": mean_stall_ms,
            "peak_vram_mb": peak_vram,
            "nan_inf_detected": nan_any,
            "forced_syncs_sync": per_rep[0]["forced_syncs_a"],
            "forced_syncs_async": per_rep[0]["forced_syncs_b"],
        },
        "per_rep": per_rep,
        "pass": passed,
        "kill_gates": {
            "peak_vram_mb_limit": KILL_GATE_VRAM_MB,
            "overlap_min_pct": KILL_GATE_OVERLAP_MIN_PCT,
            "nan_inf": nan_any,
        },
    }

    streamer.restore_all_to_cpu()
    streamer.release()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase F3 + INT8 - transfer-volume isolation benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=5,
                        help="Denoising steps to simulate (default 5 for quick run; 30 for full).")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--reps", type=int, default=3,
                        help="A/B repetitions, alternating order (default 3).")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
    parser.add_argument("--output", type=str, default="logs/f3_int8_scheduler_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmark(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Output] Results saved to: {out}")
