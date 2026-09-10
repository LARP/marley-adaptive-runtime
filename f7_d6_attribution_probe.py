"""
f7_d6_attribution_probe.py
==========================
Phase F7-D6 (measurement metrology) -- Characterize how much of the device-wide NVML
"used" reading is an idle/OS/WDDM baseline vs. a workload-induced increment.

REVISED per External Consultant (2026-09-09). This is a PROPOSAL; it is NOT approved
and must NOT be run without Director authorization. It does NOT reclassify F7-D5 and
does NOT change the 4,800 MB gate.

Trigger: F7-D5 NEGATIVE (peak NVML device "used" 5,096.1 MB > 4,800 gate) although
torch.reserved was stable ~3,766 MB across all 30 steps (no allocator divergence).
Idle (no Marley process) device "used" ~= 1,071 MB.

This probe measures in-session NVML milestones:
  S0 = idle_baseline (device used, pre-context; sampled 30-60 s for stability)
  S1 = ctx_baseline  (device used after pipeline init + empty_cache)
  S2 = workload_peak (device used peak during a reduced 3-5 step DiT window)
  S3 = post_run_floor(device used after release + empty_cache; verify S3 ~ S0)
and derives:
  ctx_overhead                              = S1 - S0
  denoise_resident                          = S2 - S1
  workload_induced_device_residency_delta   = S2 - S0   (device-wide growth, NOT proven ownership)

Consultant directives incorporated:
  - usedGpuMemory=None under WDDM means the counter is UNAVAILABLE, NOT 0 MB.
  - S2-S0 is named "workload-induced device residency delta", not "process attributable peak".
  - S0 stability is sampled (min/max/mean/median/std/spread).
  - A known CUDA control workload (~500 MB alloc/compute/free) validates the instrument.
  - cudaMemGetInfo is recorded alongside NVML and torch counters.
  - Two-layer metric under study (A: device-wide raw; B: incremental workload residency).
  - Falsification scenarios are reported and left open.

Runtime UNTOUCHED (subclass seam), same seam+step-start release discipline as F7-D5,
VAE skipped. Evidence: MEASURED / OBSERVED / DERIVED / HYPOTHESIS.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline
from marley.core.adaptive import AdaptiveEngine

HARD_GATE_VRAM_MB = 4800.0
SAFE_ABORT_NVML_MB = 5050.0
CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
PROBE_STEPS = 3
D4_PEAK_NVML_MB = 4708.9
D4_PEAK_RESERVED_MB = 3776.0
D5_PEAK_NVML_MB = 5096.1
D5_PEAK_RESERVED_MB = 3776.0


def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class NVMLSampler:
    """High-frequency physical VRAM sampler + temp/clock/util."""

    def __init__(self, device_index: int = 0, interval_ms: float = 20.0,
                 safe_abort_mb: Optional[float] = None) -> None:
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
        out = {"nvml_used_mb": 0.0, "gpu_temp_c": None, "gpu_clock_mhz": None, "gpu_util_pct": None}
        if not self._available or self._handle is None:
            return out
        try:
            import pynvml
            out["nvml_used_mb"] = round(self._read_used(), 2)
            out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            out["gpu_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
            out["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        except Exception:
            pass
        return out

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb


def nvml_device_mb() -> float:
    """Instant read of device-wide used (MB)."""
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)


def nvml_process_summary() -> Dict[str, Any]:
    """Enumerate compute/graphics processes and summed attributable usedGpuMemory."""
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    out = {"compute_n": 0, "compute_attrib_mb": 0.0, "graphics_n": 0, "graphics_attrib_mb": 0.0}
    for fld_n, fld_mb, fn in (
        ("compute_n", "compute_attrib_mb", pynvml.nvmlDeviceGetComputeRunningProcesses),
        ("graphics_n", "graphics_attrib_mb", pynvml.nvmlDeviceGetGraphicsRunningProcesses),
    ):
        try:
            for p in fn(h):
                out[fld_n] += 1
                if p.usedGpuMemory:
                    out[fld_mb] += p.usedGpuMemory / (1024 * 1024)
        except Exception:
            pass
    out["note"] = ("usedGpuMemory=None means the counter is UNAVAILABLE under WDDM, "
                   "NOT that the process uses 0 MB")
    return out


def cuda_mem_getinfo_mb() -> Optional[Dict[str, float]]:
    """Return {free_mb, total_mb, used_mb} from cudaMemGetInfo (process-visible pool)."""
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {"free_mb": round(free_b / (1024 * 1024), 2),
                "total_mb": round(total_b / (1024 * 1024), 2),
                "used_mb": round((total_b - free_b) / (1024 * 1024), 2)}
    except Exception:
        return None


def sample_stability(seconds: float, interval_ms: float = 50.0) -> Dict[str, float]:
    """Sample NVML device `used` for `seconds` and return min/max/mean/median/std (MB)."""
    import statistics
    samples: List[float] = []
    n = max(1, int(seconds * 1000 / interval_ms))
    for _ in range(n):
        samples.append(nvml_device_mb())
        time.sleep(interval_ms / 1000.0)
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "min_mb": round(min(samples), 2),
        "max_mb": round(max(samples), 2),
        "mean_mb": round(statistics.fmean(samples), 2),
        "median_mb": round(statistics.median(samples), 2),
        "std_mb": round(statistics.pstdev(samples), 2),
        "spread_mb": round(max(samples) - min(samples), 2),
    }


def run_control_workload(mb: float = 500.0, device: str = "cuda:0") -> Dict[str, Any]:
    """Known CUDA control: allocate ~mb, compute, free, synchronize+empty_cache.
    Used to validate that the incremental instrument responds proportionally."""
    out: Dict[str, Any] = {"requested_mb": mb}
    dev = torch.device(device)
    try:
        n_el = int(mb * 1024 * 1024 // 4)  # float32
        pre = round(nvml_device_mb(), 2)
        t = torch.zeros((n_el,), dtype=torch.float32, device=dev)
        torch.cuda.synchronize(dev)
        during = round(nvml_device_mb(), 2)
        ctrl_peak = during
        # small compute
        t.add_(1.0)
        torch.cuda.synchronize(dev)
        del t
        gc.collect()
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        post = round(nvml_device_mb(), 2)
        out.update({
            "pre_mb": pre, "during_mb": during, "post_mb": post,
            "delta_alloc_mb": round(during - pre, 2),
            "delta_return_mb": round(post - pre, 2),
            "expected_mb": mb,
            "responds_proportionally": abs((during - pre) - mb) <= max(0.35 * mb, 200.0),
        })
    except Exception as exc:
        out["error"] = str(exc)
    return out


class AttributionProbePipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched). Runs PROBE_STEPS DiT steps with the
    F7-D5 seam+step-start release discipline, capturing per-step reserved and peak."""

    def run_probe(self, prompt, negative_prompt, height, width, num_frames,
                  num_steps, guidance_scale, seed, mode, nvml) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {}
        t_global = time.perf_counter()

        generator = torch.Generator(device="cpu").manual_seed(seed)
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        latents = self.pipe.prepare_latents(
            batch_size=1, num_channels_latents=self.transformer.config.in_channels,
            height=height, width=width, num_frames=num_frames,
            dtype=torch.float32, device=device, generator=generator,
        )

        def pressure_sampler() -> float:
            return nvml.sample_instant()["nvml_used_mb"]

        if mode == "adaptive":
            streamer = AdaptiveEngine(
                blocks=self.blocks, device=device, dtype=dtype, window=1,
                pressure_sampler=pressure_sampler,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

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

        step_times = []
        step_end_reserved = []
        step_end_alloc = []
        seam_count = 0
        step_start_count = 0
        denoise_start = time.perf_counter()

        try:
            for step_idx, t in enumerate(timesteps):
                ts = time.perf_counter()
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                step_start_count += 1
                latent_model_input = latents.to(dtype)
                rotary_emb = self.transformer.rope(latent_model_input)
                hidden_states = self.transformer.patch_embedding(latent_model_input)
                hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
                timestep = t.expand(latent_model_input.shape[0])
                temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
                    timestep, prompt_embeds, None)
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                hidden_states_cond = hidden_states.clone()
                self._execute_dit_custom_forward(
                    streamer=streamer, mode=mode, hidden_states=hidden_states_cond,
                    encoder_hidden_states=enc_hidden_states, timestep_proj=timestep_proj,
                    rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames)
                shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
                shift = shift.to(hidden_states_cond.device); scale = scale.to(hidden_states_cond.device)
                hidden_states_cond = (self.transformer.norm_out(hidden_states_cond.float()) * (1 + scale) + shift).type_as(hidden_states_cond)
                hidden_states_cond = self.transformer.proj_out(hidden_states_cond)
                hidden_states_cond = hidden_states_cond.reshape(
                    batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_cond = hidden_states_cond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                seam_count += 1

                temb_u, timestep_proj_u, enc_hidden_states_u, _ = self.transformer.condition_embedder(
                    timestep, negative_prompt_embeds, None)
                timestep_proj_u = timestep_proj_u.unflatten(1, (6, -1))
                hidden_states_uncond = hidden_states.clone()
                self._execute_dit_custom_forward(
                    streamer=streamer, mode=mode, hidden_states=hidden_states_uncond,
                    encoder_hidden_states=enc_hidden_states_u, timestep_proj=timestep_proj_u,
                    rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames)
                shift_u, scale_u = (self.transformer.scale_shift_table.to(temb_u.device) + temb_u.unsqueeze(1)).chunk(2, dim=1)
                shift_u = shift_u.to(hidden_states_uncond.device); scale_u = scale_u.to(hidden_states_uncond.device)
                hidden_states_uncond = (self.transformer.norm_out(hidden_states_uncond.float()) * (1 + scale_u) + shift_u).type_as(hidden_states_uncond)
                hidden_states_uncond = self.transformer.proj_out(hidden_states_uncond)
                hidden_states_uncond = hidden_states_uncond.reshape(
                    batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
                ).permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred_uncond = hidden_states_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                torch.cuda.synchronize(device)

                step_s = time.perf_counter() - ts
                step_times.append(step_s)
                snap = _snap()
                step_end_reserved.append(snap["reserved_mb"])
                step_end_alloc.append(snap["alloc_mb"])
                print(f"   Step [{step_idx+1}/{num_steps}] {step_s:.2f}s | end Reserved={snap['reserved_mb']:.1f} "
                      f"Alloc={snap['alloc_mb']:.1f} NVML={snap['nvml_mb']:.1f} (peak so far {nvml.peak_mb:.1f})")

        finally:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
            torch.cuda.empty_cache()
            self.transformer.blocks = transformer_blocks_orig

        out["step_times_s"] = [round(x, 2) for x in step_times]
        out["step_end_reserved_mb"] = step_end_reserved
        out["step_end_alloc_mb"] = step_end_alloc
        out["denoise_time_s"] = round(time.perf_counter() - denoise_start, 2)
        out["seam_count"] = seam_count
        out["step_start_count"] = step_start_count
        out["total_wall_s"] = round(time.perf_counter() - t_global, 2)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7-D6 -- NVML metric attribution probe")
    p.add_argument("--steps", type=int, default=PROBE_STEPS)
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--s0-seconds", type=float, default=30.0,
                   help="Duration of the S0 idle-baseline stability sampling (Consultant recommends 30-60 s)")
    p.add_argument("--control-mb", type=float, default=500.0,
                   help="Size of the known CUDA control workload (alloc/compute/free)")
    p.add_argument("--skip-control", action="store_true",
                   help="Skip the known CUDA control workload (not recommended)")
    p.add_argument("--output", type=str, default="logs/f7_d6_attribution_probe_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D6_METRIC_ATTRIBUTION_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def classify(so_mb, so_spread_mb, ctx_mb, peak_mb, reserved_peak_mb,
             control_ok, s3_mb) -> str:
    delta = peak_mb - so_mb
    stable = so_spread_mb is not None and so_spread_mb <= 150.0
    floor_ok = s3_mb is not None and abs(s3_mb - so_mb) <= 200.0
    if stable and control_ok and floor_ok and so_mb >= 700.0 and delta <= 4000.0:
        return (f"INSTRUMENT VALIDATED (baseline {so_mb:.1f} stable spread {so_spread_mb:.1f}; "
                f"control OK; S3~S0; workload-induced device residency delta {delta:.1f} <= 4000)")
    if so_mb >= 700.0 and delta <= HARD_GATE_VRAM_MB:
        return (f"PARTIAL (baseline {so_mb:.1f}, spread {so_spread_mb}, control_ok={control_ok}, "
                f"floor_ok={floor_ok}; delta {delta:.1f} <=4800 but >4000)")
    return (f"NEGATIVE (baseline {so_mb:.1f} spread {so_spread_mb} control_ok={control_ok} "
            f"floor_ok={floor_ok}; delta {delta:.1f})")


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D6: NVML METRIC ATTRIBUTION PROBE")
    print("  (Characterizing idle/OS/WDDM baseline vs workload-induced device residency)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: {args.steps} (probe)")
    print(f"  Mode: {args.mode} | Seed: {args.seed}")
    print(f"  Hard Gate: <= {HARD_GATE_VRAM_MB:.0f} MB | D5 peak (device used): {D5_PEAK_NVML_MB:.1f} | D5 reserved: {D5_PEAK_RESERVED_MB:.0f}")
    print("  Governance: PROPOSAL per External Consultant; NOT approved; does NOT reclassify F7-D5 nor change the gate.")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print(f"  S0 idle stability sample: {args.s0_seconds:.0f} s; control workload ~{args.control_mb:.0f} MB "
              f"({'skipped' if args.skip_control else 'enabled'})")
        print(f"  {args.steps} DiT steps with seam+step-start release; VAE SKIPPED.")
        print("  Milestones: S0 idle_baseline -> S1 ctx_baseline -> S2 workload_peak -> S3 floor; "
              "derive workload-induced device residency delta (S2-S0).")
        print("[DRY-RUN] Validated. Awaiting execution authorization.")
        return

    # S0: pre-context idle baseline with STABILITY sampling (Consultant protocol)
    try:
        s0_stats = sample_stability(seconds=args.s0_seconds, interval_ms=50.0)
    except Exception as exc:
        s0_stats = {"error": str(exc)}
        print(f"[WARN] S0 stability sampling failed: {exc}")
    S0 = round(s0_stats.get("mean_mb", 0.0), 2)
    so_spread = s0_stats.get("spread_mb")
    pre_procs = nvml_process_summary()
    print(f"\n  S0 idle_baseline (device used, pre-context) mean : {S0} MB "
          f"(min {s0_stats.get('min_mb')} / max {s0_stats.get('max_mb')} / median {s0_stats.get('median_mb')} / "
          f"std {s0_stats.get('std_mb')} / spread {so_spread})")
    print(f"     processes: compute_n={pre_procs['compute_n']} attrib_mb={pre_procs['compute_attrib_mb']:.1f} | "
          f"graphics_n={pre_procs['graphics_n']} attrib_mb={pre_procs['graphics_attrib_mb']:.1f}")
    print(f"     note: usedGpuMemory=None means the counter is UNAVAILABLE under WDDM, NOT 0 MB")

    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    probe = None
    control = None
    error_occurred: Optional[str] = None
    t_start = time.perf_counter()
    try:
        pipeline = AttributionProbePipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
        # S1: post-init context baseline
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        S1 = round(nvml_device_mb(), 2)
        meminfo_s1 = cuda_mem_getinfo_mb()
        print(f"  S1 ctx_baseline (post-init, post-empty_cache): {S1} MB | cudaMemGetInfo={meminfo_s1}")
        # Known CUDA control workload (validates the instrument BEFORE Wan)
        if not args.skip_control:
            print(f"  [control] running known CUDA control workload ~{args.control_mb:.0f} MB ...")
            control = run_control_workload(mb=args.control_mb, device=args.device)
            print(f"  [control] pre={control.get('pre_mb')} during={control.get('during_mb')} "
                  f"post={control.get('post_mb')} | delta_alloc={control.get('delta_alloc_mb')} "
                  f"delta_return={control.get('delta_return_mb')} | responds_proportionally="
                  f"{control.get('responds_proportionally')}")
        probe = pipeline.run_probe(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode=args.mode, nvml=nvml)
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D6 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        S2 = nvml.stop()
        # S3: post-run empty_cache context floor
        try:
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            S3 = round(nvml_device_mb(), 2)
            meminfo_s3 = cuda_mem_getinfo_mb()
        except Exception:
            S3 = None
            meminfo_s3 = None

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB.")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()
    post_procs = nvml_process_summary()

    ctx_overhead = round(S1 - S0, 1)
    denoise_resident = round(S2 - S1, 1)
    workload_delta = round(S2 - S0, 1)   # workload-induced device residency delta (NOT proven ownership)
    so_share = round(S0, 1)
    control_ok = bool(control.get("responds_proportionally")) if control else False

    result = classify(so_share, so_spread, S1, S2, peak_reserved, control_ok, S3)

    print("\n" + "=" * 80)
    print("  PHASE F7-D6: ATTRIBUTION SCORECARD")
    print("=" * 80)
    print(f"  S0 idle_baseline (pre-context)    [MEASURED] : {so_share:8.1f} MB (spread {so_spread} MB)")
    print(f"  S1 ctx_baseline (post-init)       [MEASURED] : {S1:8.1f} MB")
    print(f"  S2 workload_peak (device used)    [MEASURED] : {S2:8.1f} MB (D5: {D5_PEAK_NVML_MB:.1f})")
    if S3 is not None:
        print(f"  S3 post-empty_cache floor         [MEASURED] : {S3:8.1f} MB (floor_ok={abs(S3-so_share)<=200.0})")
    print(f"  ctx_overhead (S1-S0)              [DERIVED]  : {ctx_overhead:8.1f} MB")
    print(f"  denoise_resident (S2-S1)          [DERIVED]  : {denoise_resident:8.1f} MB")
    print(f"  WORKLOAD-INDUCED device delta     [DERIVED]  : {workload_delta:8.1f} MB (S2-S0; NOT proven ownership)")
    if control:
        print(f"  control workload (~{args.control_mb:.0f} MB)     [OBSERVED] : delta_alloc={control.get('delta_alloc_mb')} "
              f"delta_return={control.get('delta_return_mb')} responds={control.get('responds_proportionally')}")
    print(f"  Peak Reserved (torch)             [OBSERVED] : {peak_reserved:8.1f} MB (D5: {D5_PEAK_RESERVED_MB:.0f})")
    print(f"  Peak Allocated (torch)            [OBSERVED] : {peak_alloc:8.1f} MB")
    print(f"  Host RSS                          [OBSERVED] : {peak_ram:8.1f} MB")
    print(f"  Wall-clock                        [OBSERVED] : {t_total:8.2f} s")
    print(f"  Result                             [DERIVED]  : {result}")
    print(f"  Abort/Error                       : {error_occurred or 'none'}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D6 (NVML metric attribution probe)",
        "evidence_class": "All measured/observed; derived computed",
        "governance_note": ("PROPOSAL per Consultant: NOT a reclassification of F7-D5 and NOT a gate change. "
                            "S2-S0 is 'workload-induced device residency delta', not proven process ownership."),
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": args.mode, "seed": args.seed},
        "s0_stability": s0_stats,
        "milestones_mb": {"S0_idle_baseline": so_share, "S1_ctx_baseline": round(S1, 1),
                          "S2_workload_peak": round(S2, 1),
                          "S3_post_empty_cache_floor": S3},
        "derived_mb": {"ctx_overhead": ctx_overhead, "denoise_resident": denoise_resident,
                       "workload_induced_device_residency_delta": workload_delta, "so_share": so_share},
        "control_workload": control,
        "cross_checks_mb": {"peak_torch_reserved": round(peak_reserved, 1),
                            "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_process_ram": round(peak_ram, 1),
                            "cudaMemGetInfo_S1": meminfo_s1,
                            "cudaMemGetInfo_S3": meminfo_s3},
        "per_step": (probe or {}).get("step_end_reserved_mb"),
        "denoise_time_s": (probe or {}).get("denoise_time_s"),
        "processes_pre": pre_procs,
        "processes_post": post_procs,
        "comparison": {"d5_peak_nvml_device_used_mb": D5_PEAK_NVML_MB,
                       "d5_peak_torch_reserved_mb": D5_PEAK_RESERVED_MB},
        "result_class": result,
        "verdict": {"execution_completed": error_occurred is None,
                    "s0_stable_spread_le_150mb": (so_spread is not None and so_spread <= 150.0),
                    "control_workload_ok": control_ok,
                    "s3_approx_s0": (S3 is not None and abs(S3 - so_share) <= 200.0),
                    "workload_induced_device_residency_delta_le_4800": workload_delta <= HARD_GATE_VRAM_MB,
                    "workload_induced_device_residency_delta_le_4000": workload_delta <= 4000.0,
                    "crosscheck_denoise_resident_vs_reserved": round(abs(denoise_resident - peak_reserved), 1),
                    "error_detail": error_occurred},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7-D6 — NVML Metric Attribution Probe -- Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, {args.steps} DiT steps, mode `{args.mode}` (VAE skipped)  \n")
        w("**Status:** measurement-metrology probe (per External Consultant). Does NOT reclassify F7-D5.  \n")
        w("---\n\n## 1. Result\n\n")
        w(f"- S0 idle_baseline (pre-context): mean {so_share:.1f} MB "
          f"(min {s0_stats.get('min_mb')} / max {s0_stats.get('max_mb')} / median {s0_stats.get('median_mb')} / "
          f"std {s0_stats.get('std_mb')} / spread {so_spread})\n")
        w(f"- S1 ctx_baseline (post-init): {S1:.1f} MB\n")
        w(f"- S2 workload_peak (device used): {S2:.1f} MB (D5: {D5_PEAK_NVML_MB:.1f})\n")
        if S3 is not None:
            w(f"- S3 post_empty_cache floor: {S3:.1f} MB (S3~S0: {abs(S3 - so_share) <= 200.0})\n")
        w(f"- ctx_overhead (S1-S0): {ctx_overhead:.1f} MB\n")
        w(f"- denoise_resident (S2-S1): {denoise_resident:.1f} MB\n")
        w(f"- **WORKLOAD-INDUCED device residency delta (S2-S0): {workload_delta:.1f} MB** "
          f"(NOT proven process ownership)\n")
        if control:
            w(f"- Control workload (~{args.control_mb:.0f} MB): delta_alloc={control.get('delta_alloc_mb')} "
              f"delta_return={control.get('delta_return_mb')} responds_proportionally={control.get('responds_proportionally')}\n")
        w(f"- Peak Reserved (torch): {peak_reserved:.1f} MB (D5: {D5_PEAK_RESERVED_MB:.0f})\n")
        w(f"- Classification: **{result}**\n")
        w("\n## 2. Interpretation\n\n")
        w("The device-wide `nvmlDeviceGetMemoryInfo().used` metric does not start at zero: "
          "an idle baseline (S0) is observed even with no Marley workload. `usedGpuMemory=None` "
          "per PID under WDDM means the counter is UNAVAILABLE, not 0 MB. The quantity `S2 - S0` "
          "is a **workload-induced device residency delta** (device-wide growth), NOT proven "
          "process ownership; it is cross-checked against `torch.reserved` and a known CUDA control "
          "workload validates that the incremental instrument responds proportionally.\n")
        w("\n## 3. Falsification scenarios (remain open)\n\n")
        w("1. S0 varies by hundreds of MB with system state -> subtraction not robust.\n")
        w("2. Desktop/WDDM grows its own residency during the workload -> S2-S0 overestimates Marley.\n")
        w("3. Marley induces driver allocations not in torch.reserved -> NVML-reserved may be real Marley memory.\n")
        w("4. Device-wide increment grows while reserved stays flat -> investigate driver/WDDM/contexts/shared memory.\n")
        w("5. Full 30-step run increases the delta vs the short probe -> 3-5 step probe insufficient.\n")
        w("\n## 4. Governance\n\n")
        w("Observational attribution probe in an isolated runner (VAE skipped, reduced steps). "
          "Does NOT reclassify F7-D5 and authorizes NO runtime modification and NO gate change. "
          "Two-layer metric under study (A: device-wide raw, always reported; B: incremental workload "
          "residency, label pending validation). Any redefinition of the ground-truth metric of "
          "ROADMAP section 2 is a separate Director/Consejero decision.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
