"""
f8_screening_runner.py
======================
Phase F8 — 10-Step A/B Screening Benchmark (Clean Processes Architecture)
Protocol Reference: docs/F8_PREREGISTERED_SCREENING_PROTOCOL_01.md

Architecture:
  Process Coordinator (Monitor):
    1. Launches Condition A (Control FP16) in a dedicated clean subprocess.
    2. Tracks execution, records Step 1-10 metrics (Allocated, Reserved, Peak NVML).
    3. Captures Step 10 latent vector for fidelity reference.
    4. Waits for Condition A to exit cleanly and destroy CUDA context.
    5. Enforces 60-second system cooldown window for WDDM page recovery.
    6. Launches Condition B (Treatment INT8 Quantized) in a dedicated clean subprocess.
    7. Tracks execution, records Step 1-10 metrics.
    8. Calculates latent cosine similarity on Step 10 latents.
    9. Evaluates Peak NVML reduction vs F8 Decision Matrix.
   10. Saves JSON telemetry and Markdown report.

Decision Thresholds:
  - Strong: >= 196.0 MB reduction -> GO (authorize 30-step F8 Stage B)
  - Promising: 100.0 - 195.0 MB -> CONDITIONAL GO (minor micro-intervention)
  - Weak: < 50.0 MB -> NO-GO (quantization discarded)
  - Null: ~0.0 MB -> PERMANENT CLOSE (720p permanently dropped on 6 GB)
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline
from marley.core.adaptive import AdaptiveEngine
from marley.ops.async_stream_int8 import INT8BudgetedStreamer
from marley.ops.async_stream import BudgetedAsyncStreamer

HARD_GATE_PHYSICAL_VRAM_MB = 4800.0
SAFE_ABORT_NVML_MB = 5800.0
CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
DEFAULT_STEPS = 10
DEFAULT_FRAMES = 33
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def nvml_device_mb() -> float:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)


class NVMLSampler:
    """High-frequency physical VRAM sampler (20 ms)."""

    def __init__(self, device_index: int = 0, interval_ms: float = 20.0,
                 safe_abort_mb: Optional[float] = SAFE_ABORT_NVML_MB) -> None:
        self.interval_s = interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._handle = None
        self.peak_mb = 0.0
        self.peak_timestamp: Optional[str] = None
        self.safe_abort_triggered = False
        self.safe_abort_mb = safe_abort_mb
        self._lock = threading.Lock()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._available = True
            self.peak_mb = self._read_used()
        except Exception as exc:
            print(f"[WARN] NVML init failed: {exc}", file=sys.stderr)

    def _read_used(self) -> float:
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024 * 1024)
        except Exception:
            return 0.0

    def start(self) -> None:
        if not self._available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                used_mb = self._read_used()
                with self._lock:
                    if used_mb > self.peak_mb:
                        self.peak_mb = used_mb
                        self.peak_timestamp = datetime.datetime.now().isoformat()
                    if self.safe_abort_mb is not None and used_mb > self.safe_abort_mb:
                        self.safe_abort_triggered = True
            except Exception:
                pass
            time.sleep(self.interval_s)

    def sample_instant(self) -> Dict[str, Any]:
        out = {"nvml_used_mb": 0.0}
        if not self._available or self._handle is None:
            return out
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            out["nvml_used_mb"] = round(info.used / (1024 * 1024), 2)
        except Exception:
            pass
        return out

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb


def sample_stability(seconds: float, interval_ms: float = 1000.0) -> Dict[str, Any]:
    import statistics
    samples: List[float] = []
    n = max(1, int(seconds * 1000.0 / interval_ms))
    for _ in range(n):
        samples.append(nvml_device_mb())
        time.sleep(interval_ms / 1000.0)
    return {
        "n": len(samples),
        "mean_mb": round(statistics.fmean(samples), 2),
        "min_mb": round(min(samples), 2),
        "max_mb": round(max(samples), 2),
        "spread_mb": round(max(samples) - min(samples), 2),
    }


class ScreeningWorkerPipeline(MarleyEndToEndPipeline):
    """Executes 10 steps DiT under either FP16 control or INT8 quantized treatment."""

    def run_screening_denoise(
        self, prompt: str, negative_prompt: str, height: int, width: int,
        num_frames: int, num_steps: int, guidance_scale: float, seed: int,
        quantized: bool, nvml: NVMLSampler,
    ) -> Dict[str, Any]:
        t_global = time.perf_counter()
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {}

        # 1. Text embeddings
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # 2. Latents & Scheduler
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = self.pipe.prepare_latents(
            batch_size=1, num_channels_latents=self.transformer.config.in_channels,
            height=height, width=width, num_frames=num_frames,
            dtype=torch.float32, device=device, generator=generator,
        )

        def pressure_sampler() -> float:
            return nvml.sample_instant()["nvml_used_mb"]

        # Streamer configuration
        if quantized:
            print("[Worker] Initializing INT8BudgetedStreamer (Selective DiT Projections Quantized)...")
            streamer = INT8BudgetedStreamer(blocks=self.blocks, device=device, dtype=dtype)
            mode = "async_int8"
        else:
            print("[Worker] Initializing BudgetedAsyncStreamer (FP16 Control Baseline)...")
            streamer = BudgetedAsyncStreamer(blocks=self.blocks, device=device, dtype=dtype)
            mode = "async_fp16"

        torch.set_grad_enabled(False)
        batch_size, num_channels, num_frames_in, h_in, w_in = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        pn_f = num_frames_in // p_t
        pn_h = h_in // p_h
        pn_w = w_in // p_w
        transformer_blocks_orig = self.transformer.blocks
        self.transformer.blocks = nn.ModuleList([])

        def _snap():
            return {
                "alloc_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
                "nvml_mb": nvml.sample_instant()["nvml_used_mb"],
            }

        step_telemetry: List[Dict[str, Any]] = []
        step_times: List[float] = []

        try:
            for step_idx, t in enumerate(timesteps):
                ts = time.perf_counter()
                if nvml.safe_abort_triggered:
                    raise RuntimeError(f"SAFE ABORT TRIGGERED: NVML exceeded {SAFE_ABORT_NVML_MB} MB")

                # Step-start drain
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

                latent_model_input = latents.to(dtype)
                rotary_emb = self.transformer.rope(latent_model_input)
                hidden_states = self.transformer.patch_embedding(latent_model_input)
                hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
                timestep = t.expand(latent_model_input.shape[0])
                temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
                    timestep, prompt_embeds, None)
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                # COND pass
                hidden_states_cond = hidden_states.clone()
                self._execute_dit_custom_forward(
                    streamer=streamer, mode=mode, hidden_states=hidden_states_cond,
                    encoder_hidden_states=enc_hidden_states, timestep_proj=timestep_proj,
                    rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames)
                shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
                shift = shift.to(hidden_states_cond.device)
                scale = scale.to(hidden_states_cond.device)
                hidden_states_cond = (self.transformer.norm_out(hidden_states_cond.float()) * (1 + scale) + shift).type_as(hidden_states_cond)
                hidden_states_cond = self.transformer.proj_out(hidden_states_cond)
                hidden_states_cond = hidden_states_cond.reshape(
                    batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_cond = hidden_states_cond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                # Surgical seam release
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

                # UNCOND pass
                temb_u, timestep_proj_u, enc_hidden_states_u, _ = self.transformer.condition_embedder(
                    timestep, negative_prompt_embeds, None)
                timestep_proj_u = timestep_proj_u.unflatten(1, (6, -1))
                hidden_states_uncond = hidden_states.clone()
                self._execute_dit_custom_forward(
                    streamer=streamer, mode=mode, hidden_states=hidden_states_uncond,
                    encoder_hidden_states=enc_hidden_states_u, timestep_proj=timestep_proj_u,
                    rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames)
                shift_u, scale_u = (self.transformer.scale_shift_table.to(temb_u.device) + temb_u.unsqueeze(1)).chunk(2, dim=1)
                shift_u = shift_u.to(hidden_states_uncond.device)
                scale_u = scale_u.to(hidden_states_uncond.device)
                hidden_states_uncond = (self.transformer.norm_out(hidden_states_uncond.float()) * (1 + scale_u) + shift_u).type_as(hidden_states_uncond)
                hidden_states_uncond = self.transformer.proj_out(hidden_states_uncond)
                hidden_states_uncond = hidden_states_uncond.reshape(
                    batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_uncond = hidden_states_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                torch.cuda.synchronize(device)

                step_s = round(time.perf_counter() - ts, 2)
                step_times.append(step_s)
                snap = _snap()
                step_record = {
                    "step": step_idx + 1,
                    "step_time_s": step_s,
                    "alloc_mb": snap["alloc_mb"],
                    "reserved_mb": snap["reserved_mb"],
                    "end_nvml_mb": snap["nvml_mb"],
                    "peak_so_far_mb": round(nvml.peak_mb, 1),
                }
                step_telemetry.append(step_record)
                print(f"   Step [{step_idx+1}/{num_steps}] {step_s:.2f}s | Reserved={snap['reserved_mb']:.1f} "
                      f"Alloc={snap['alloc_mb']:.1f} NVML={snap['nvml_mb']:.1f} (Peak so far {nvml.peak_mb:.1f})")

        finally:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
            torch.cuda.empty_cache()
            self.transformer.blocks = transformer_blocks_orig

        out["step_telemetry"] = step_telemetry
        out["step_times_s"] = step_times
        out["cadence_avg_s"] = round(sum(step_times) / len(step_times), 2) if step_times else 0.0
        out["total_denoise_s"] = round(time.perf_counter() - t_global, 2)

        # Latent sample hash / summary for cosine fidelity check
        lat_flat = latents.detach().float().flatten().cpu()
        out["latent_norm"] = round(torch.norm(lat_flat).item(), 4)
        out["has_nan"] = bool(torch.isnan(lat_flat).any().item())
        out["has_inf"] = bool(torch.isinf(lat_flat).any().item())

        # Save latent tensor to disk for inter-process cosine calculation
        latent_dump_path = f"logs/f8_latent_{'quantized' if quantized else 'control'}.pt"
        torch.save(lat_flat, latent_dump_path)
        out["latent_dump_path"] = latent_dump_path

        return out


def run_worker_mode(args: argparse.Namespace) -> None:
    """Invoked inside dedicated clean subprocess."""
    print(f"\n[Worker: PID {os.getpid()}] Mode: {'TREATMENT (INT8)' if args.quantized else 'CONTROL (FP16)'}")
    # 1. Baseline S0
    s0_stats = sample_stability(args.s0_seconds, interval_ms=1000.0)
    print(f"[Worker] S0 baseline: {s0_stats['mean_mb']:.1f} MB (spread: {s0_stats['spread_mb']:.1f} MB)")

    # 2. CUDA warm-up
    torch.cuda.init()
    _w = torch.empty(int(10 * 1024 * 1024 / 4), dtype=torch.float32, device=args.device).fill_(0.0)
    del _w
    gc.collect(); torch.cuda.empty_cache()

    # 3. NVML Continuous Sampler
    sampler = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    sampler.start()

    # 4. Pipeline Execution
    pipe = ScreeningWorkerPipeline(device=args.device, dit_dtype=torch.float16,
                                   vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
    s1_mb = nvml_device_mb()
    print(f"[Worker] S1 operational: {s1_mb:.1f} MB")

    workload_res = None
    exec_err = None
    try:
        workload_res = pipe.run_screening_denoise(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            quantized=args.quantized, nvml=sampler,
        )
    except Exception as exc:
        exec_err = str(exc)
        print(f"[Worker ERROR] Execution exception: {exec_err}")
    finally:
        peak_nvml = sampler.stop()

    print(f"[Worker] Finished! Peak Physical VRAM: {peak_nvml:.1f} MB")

    worker_result = {
        "pid": os.getpid(),
        "quantized": args.quantized,
        "s0_stats": s0_stats,
        "s1_mb": s1_mb,
        "peak_nvml_mb": peak_nvml,
        "workload": workload_res,
        "error": exec_err,
    }

    with open(args.worker_out, "w", encoding="utf-8") as f:
        json.dump(worker_result, f, indent=2)

    # Clean exit to trigger OS/WDDM handle release
    print(f"[Worker: PID {os.getpid()}] Exiting process cleanly...")
    sys.stdout.flush()
    os._exit(0)


def run_coordinator_mode(args: argparse.Namespace) -> None:
    """Orchestrates Clean Process A -> 60s Cooldown -> Clean Process B."""
    print("=" * 80)
    print("  PHASE F8 -- SCREENING BENCHMARK: CLEAN PROCESSES A/B SCREENING")
    print("  (Evaluating if Selective Quantization recovers the ~196 MB deficit)")
    print("  Protocol Reference: docs/F8_PREREGISTERED_SCREENING_PROTOCOL_01.md")
    print("=" * 80)

    py_exe = sys.executable
    control_out = "logs/f8_screening_control_a.json"
    treatment_out = "logs/f8_screening_treatment_b.json"

    # -------------------------------------------------------------
    # FASE 1: PROCESO A (CONTROL FP16 LIMPIO)
    # -------------------------------------------------------------
    print("\n[Coordinator] >>> LAUNCHING CONDITION A: CONTROL FP16 (CLEAN PROCESS) <<<")
    cmd_a = [
        py_exe, __file__, "--worker-mode",
        "--steps", str(args.steps), "--frames", str(args.frames),
        "--width", str(args.width), "--height", str(args.height),
        "--s0-seconds", str(args.s0_seconds),
        "--worker-out", control_out,
    ]
    t0_a = time.perf_counter()
    proc_a = subprocess.run(cmd_a)
    wall_a = round(time.perf_counter() - t0_a, 2)
    print(f"[Coordinator] Condition A finished with code {proc_a.returncode} in {wall_a}s.")

    if proc_a.returncode != 0 or not os.path.exists(control_out):
        print(f"[Coordinator FATAL] Condition A failed. Aborting screening.")
        return

    with open(control_out, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    peak_a = data_a["peak_nvml_mb"]
    print(f"\n[Coordinator] Condition A (Control FP16) Measured Peak: {peak_a:.1f} MB")

    # -------------------------------------------------------------
    # FASE 2: ENFRIAMIENTO OBLIGATORIO DEL SISTEMA OPERATIVO (60 s)
    # -------------------------------------------------------------
    cooldown_s = args.cooldown_seconds
    print(f"\n[Coordinator] >>> COMMENCING {cooldown_s:.0f}s MANDATORY SYSTEM COOLDOWN WINDOW <<<")
    print(f"[Coordinator] Allowing WDDM/DWM to collect pages and restore clean hardware state...")
    for remaining in range(int(cooldown_s), 0, -10):
        m = nvml_device_mb()
        print(f"   Cooldown remaining: {remaining:2d}s | NVML physical: {m:.1f} MB")
        time.sleep(min(10.0, float(remaining)))

    # -------------------------------------------------------------
    # FASE 3: PROCESO B (TRATAMIENTO INT8 LIMPIO)
    # -------------------------------------------------------------
    print("\n[Coordinator] >>> LAUNCHING CONDITION B: TREATMENT INT8 (CLEAN PROCESS) <<<")
    cmd_b = [
        py_exe, __file__, "--worker-mode", "--quantized",
        "--steps", str(args.steps), "--frames", str(args.frames),
        "--width", str(args.width), "--height", str(args.height),
        "--s0-seconds", str(args.s0_seconds),
        "--worker-out", treatment_out,
    ]
    t0_b = time.perf_counter()
    proc_b = subprocess.run(cmd_b)
    wall_b = round(time.perf_counter() - t0_b, 2)
    print(f"[Coordinator] Condition B finished with code {proc_b.returncode} in {wall_b}s.")

    if proc_b.returncode != 0 or not os.path.exists(treatment_out):
        print(f"[Coordinator FATAL] Condition B failed. Aborting screening.")
        return

    with open(treatment_out, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    peak_b = data_b["peak_nvml_mb"]
    print(f"\n[Coordinator] Condition B (Treatment INT8) Measured Peak: {peak_b:.1f} MB")

    # -------------------------------------------------------------
    # FASE 4: ANÁLISIS COMPARATIVO Y MATRIZ DE DECISIÓN
    # -------------------------------------------------------------
    delta_peak = round(peak_a - peak_b, 2)
    print(f"\n[Coordinator] Delta Peak Reduction (Peak_A - Peak_B): {delta_peak:+.1f} MB")

    # Calculate Latent Cosine Similarity
    cos_sim = None
    lat_a_path = (data_a.get("workload") or {}).get("latent_dump_path")
    lat_b_path = (data_b.get("workload") or {}).get("latent_dump_path")
    if lat_a_path and lat_b_path and os.path.exists(lat_a_path) and os.path.exists(lat_b_path):
        t_a = torch.load(lat_a_path, weights_only=True)
        t_b = torch.load(lat_b_path, weights_only=True)
        cos_sim = round(F.cosine_similarity(t_a.unsqueeze(0), t_b.unsqueeze(0)).item(), 6)
        print(f"[Coordinator] Numerical Fidelity (Cosine Similarity on Step 10 Latents): {cos_sim:.6f}")

    # Decision Matrix Evaluation
    if delta_peak >= 196.0:
        verdict = "🟢 RESULTADO FUERTE (GO)"
        verdict_action = "GO: Cuantización redujo >= 196 MB. Autoriza validación completa de 30 pasos."
    elif delta_peak >= 100.0:
        verdict = "🟡 RESULTADO PROMETEDOR (GO CONDICIONAL)"
        verdict_action = "GO CONDICIONAL: Reducción entre 100 y 195 MB. Justifica micro-intervención mínima."
    elif delta_peak >= 50.0:
        verdict = "🔴 RESULTADO DÉBIL (NO-GO)"
        verdict_action = "NO-GO: Reducción insuficiente (< 100 MB). Se descarta la cuantización como solución viable."
    else:
        verdict = "🔴 RESULTADO NULO (CIERRE DEFINITIVO)"
        verdict_action = "CIERRE DEFINITIVO: Sin efecto apreciable (< 50 MB). La línea 720p se cierra definitivamente."

    cadence_a = (data_a.get("workload") or {}).get("cadence_avg_s", 0.0)
    cadence_b = (data_b.get("workload") or {}).get("cadence_avg_s", 0.0)

    # Assemble Full Scorecard
    scorecard = {
        "timestamp": datetime.datetime.now().isoformat(),
        "protocol_reference": "docs/F8_PREREGISTERED_SCREENING_PROTOCOL_01.md",
        "control_a_fp16": {
            "s0_mean_mb": data_a["s0_stats"]["mean_mb"],
            "s1_mb": data_a["s1_mb"],
            "peak_nvml_mb": peak_a,
            "cadence_avg_s": cadence_a,
            "telemetry": (data_a.get("workload") or {}).get("step_telemetry", []),
        },
        "treatment_b_int8": {
            "s0_mean_mb": data_b["s0_stats"]["mean_mb"],
            "s1_mb": data_b["s1_mb"],
            "peak_nvml_mb": peak_b,
            "cadence_avg_s": cadence_b,
            "telemetry": (data_b.get("workload") or {}).get("step_telemetry", []),
        },
        "comparison": {
            "delta_peak_reduction_mb": delta_peak,
            "cosine_similarity_step10": cos_sim,
            "cadence_delta_s": round(cadence_b - cadence_a, 2),
            "verdict": verdict,
            "verdict_action": verdict_action,
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n>> Screening telemetry saved to: {args.output}")

    # Generate Markdown Report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F8 Screening Report — A/B Evaluation of Selective DiT Quantization\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Protocol Reference:** [`docs/F8_PREREGISTERED_SCREENING_PROTOCOL_01.md`](F8_PREREGISTERED_SCREENING_PROTOCOL_01.md)  \n")
        w(f"**Official Verdict:** **{verdict}**  \n\n")
        w("---\n\n")
        w("## 1. Executive Screening Scorecard\n\n")
        w(f"> **Governance Resolution:** {verdict_action}\n\n")
        w("| Condition | S0 Baseline | S1 Operac. | Peak NVML (Steps 1-10) | Avg Cadence | Latent Cosine Sim |\n")
        w("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        w(f"| **Condition A: Control FP16** | {data_a['s0_stats']['mean_mb']:.1f} MB | {data_a['s1_mb']:.1f} MB | **{peak_a:.1f} MB** | {cadence_a:.2f} s/step | Baseline (1.000000) |\n")
        w(f"| **Condition B: Treatment INT8** | {data_b['s0_stats']['mean_mb']:.1f} MB | {data_b['s1_mb']:.1f} MB | **{peak_b:.1f} MB** | {cadence_b:.2f} s/step | **{cos_sim if cos_sim is not None else 'N/A'}** |\n")
        w(f"| **Delta (Reduction)** | — | — | **{delta_peak:+.1f} MB** | {cadence_b - cadence_a:+.2f} s/step | Tolerance $\ge 0.990$ |\n\n")
        w("## 2. Step-by-Step Telemetry Progression (Steps 1–10)\n\n")
        w("| Step | Control A NVML (MB) | Control A Reserved | Treatment B NVML (MB) | Treatment B Reserved | Delta NVML |\n")
        w("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        steps_a = (data_a.get("workload") or {}).get("step_telemetry", [])
        steps_b = (data_b.get("workload") or {}).get("step_telemetry", [])
        for i in range(max(len(steps_a), len(steps_b))):
            sa = steps_a[i] if i < len(steps_a) else {}
            sb = steps_b[i] if i < len(steps_b) else {}
            nvml_a = sa.get("end_nvml_mb", 0.0)
            nvml_b = sb.get("end_nvml_mb", 0.0)
            diff = round(nvml_a - nvml_b, 1) if (nvml_a and nvml_b) else 0.0
            w(f"| {i+1} | {nvml_a:.1f} | {sa.get('reserved_mb', 0.0):.1f} | {nvml_b:.1f} | {sb.get('reserved_mb', 0.0):.1f} | {diff:+.1f} MB |\n")
        w("\n## 3. Epistemological Implications\n\n")
        w(f"The objective measurement demonstrates a physical VRAM peak delta of **{delta_peak:+.1f} MB**. ")
        if delta_peak >= 196.0:
            w("The reduction satisfies the preregistered threshold of 196 MB, confirming that selective DiT quantization sufficiently relieves memory pressure in the Step 5-7 window. Phase F8 Stage B (full 30-step validation) is authorized.\n")
        elif delta_peak >= 100.0:
            w("The reduction provides a promising signal (100-195 MB), but does not fully clear the 196 MB gap alone. A secondary minor intervention is warranted before full validation.\n")
        else:
            w("The reduction is insufficient to justify 720p feasibility under the 4.8 GB hard gate. The quantization lever does not overcome the physical hardware boundary on RTX 3050 Laptop under Windows 11 WDDM.\n")

    print(f">> Report saved to: {args.report}")
    print("=" * 80)
    print(f"  OFFICIAL SCREENING VERDICT: {verdict}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F8 -- 10-Step A/B Screening")
    p.add_argument("--worker-mode", action="store_true", help="Internal flag: runs worker subroutine")
    p.add_argument("--quantized", action="store_true", help="Internal flag: runs INT8 quantized condition")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--s0-seconds", type=float, default=30.0, help="S0 observation seconds per condition")
    p.add_argument("--cooldown-seconds", type=float, default=60.0, help="Mandatory cooldown between A and B")
    p.add_argument("--worker-out", type=str, default="logs/f8_worker_tmp.json")
    p.add_argument("--output", type=str, default="logs/f8_screening_ab_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F8_SCREENING_AB_REPORT_01.md")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker_mode:
        run_worker_mode(args)
    else:
        run_coordinator_mode(args)
