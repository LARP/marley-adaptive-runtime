"""
f7_d6_attribution_probe.py
==========================
Phase F7-D6 (measurement metrology) -- Characterize how much of the device-wide NVML
"used" reading is an idle/OS/WDDM baseline vs. a workload-induced increment.

REVISED and AUTHORIZED per External Consultant:
- Authorizations no.1-4 (2026-09-09)
- Methodological Resolution (2026-09-10, docs/private/F7_D6_CONSULTANT_RESOLUTION_01.md):
  * S3 return-to-baseline decoupled from instrument validity -> phenomenological characterization.
  * Control A executed post-CUDA init with deterministic warm-up (~10 MB), pre-pipeline.
  * NVML per-process is auxiliary under Windows WDDM; complemented with psutil.
  * Temporal windows: S0=60s, S3=120s, S4=60s @ 1 Hz continuous telemetry.

This is a metrology experiment: it does NOT reclassify F7-D5 and does NOT change the
4,800 MB gate. F7-D5 remains NEGATIVE; no 30-step rerun.

Authorized corrected protocol:
  S0 idle 60 s
  -> torch.cuda.init() + deterministic warm-up (~10 MB)
  -> Control A (~500 MB)
  -> load Wan2.1 pipeline
  -> S1 operational baseline
  -> Wan 3 DiT steps (VAE skipped)
  -> empty_cache()
  -> S3 120 s (trajectory characterization)
  -> Control B (~500 MB)
  -> S4 60 s (final observation).
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


def get_process_host_memory_mb() -> Dict[str, float]:
    p = psutil.Process(os.getpid())
    mem = p.memory_info()
    return {
        "working_set_mb": round(mem.rss / (1024 * 1024), 2),
        "commit_private_mb": round(mem.vms / (1024 * 1024), 2),
    }


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
    """Enumerate compute/graphics processes and target PID attributable usedGpuMemory."""
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    my_pid = os.getpid()
    out: Dict[str, Any] = {
        "compute_n": 0, "compute_attrib_mb": 0.0,
        "graphics_n": 0, "graphics_attrib_mb": 0.0,
        "target_pid": my_pid,
        "target_pid_found": False,
        "target_pid_gpu_mem": None,
    }
    for fld_n, fld_mb, fn in (
        ("compute_n", "compute_attrib_mb", pynvml.nvmlDeviceGetComputeRunningProcesses),
        ("graphics_n", "graphics_attrib_mb", pynvml.nvmlDeviceGetGraphicsRunningProcesses),
    ):
        try:
            for p in fn(h):
                out[fld_n] += 1
                if p.pid == my_pid:
                    out["target_pid_found"] = True
                    out["target_pid_gpu_mem"] = (
                        round(p.usedGpuMemory / (1024 * 1024), 2)
                        if p.usedGpuMemory is not None else "UNAVAILABLE_WDDM"
                    )
                if p.usedGpuMemory:
                    out[fld_mb] += p.usedGpuMemory / (1024 * 1024)
        except Exception:
            pass
    out["host_process_memory"] = get_process_host_memory_mb()
    out["note"] = ("usedGpuMemory=None/UNAVAILABLE means the counter is UNAVAILABLE under WDDM, "
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


def sample_stability(seconds: float, interval_ms: float = 1000.0, record_series: bool = True) -> Dict[str, Any]:
    """Sample NVML device `used` for `seconds` at `interval_ms` (default 1 Hz per Consultant)
    and return min/max/mean/median/std (MB) plus trajectory."""
    import statistics
    samples: List[float] = []
    series: List[Dict[str, Any]] = []
    n = max(1, int(seconds * 1000.0 / interval_ms))
    t0 = time.perf_counter()
    for _ in range(n):
        mb = nvml_device_mb()
        samples.append(mb)
        if record_series:
            series.append({
                "t_s": round(time.perf_counter() - t0, 2),
                "nvml_used_mb": mb,
            })
        time.sleep(interval_ms / 1000.0)
    if not samples:
        return {"n": 0}
    res: Dict[str, Any] = {
        "n": len(samples),
        "interval_s": interval_ms / 1000.0,
        "min_mb": round(min(samples), 2),
        "max_mb": round(max(samples), 2),
        "mean_mb": round(statistics.fmean(samples), 2),
        "median_mb": round(statistics.median(samples), 2),
        "std_mb": round(statistics.pstdev(samples), 2),
        "spread_mb": round(max(samples) - min(samples), 2),
    }
    if record_series:
        res["trajectory"] = series
    return res


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
        _meminfo_peak: Optional[Dict[str, float]] = None
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
                # capture cudaMemGetInfo at workload peak for coherence cross-check
                _mi = cuda_mem_getinfo_mb()
                if _mi and (_meminfo_peak is None or _mi["used_mb"] > _meminfo_peak["used_mb"]):
                    _meminfo_peak = _mi
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
        out["meminfo_workload_peak"] = _meminfo_peak
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
    p.add_argument("--s0-seconds", type=float, default=60.0,
                   help="Duration of the S0 idle-baseline stability sampling (Consultant: 60 s)")
    p.add_argument("--s3-seconds", type=float, default=120.0,
                   help="Duration of the S3 post-workload stability sampling (Consultant Resolution: 120 s)")
    p.add_argument("--s4-seconds", type=float, default=60.0,
                   help="Duration of the S4 final observation sampling (Consultant Resolution: 60 s)")
    p.add_argument("--control-mb", type=float, default=500.0,
                   help="Size of the known CUDA control workload (alloc/compute/free)")
    p.add_argument("--skip-control", action="store_true",
                   help="Skip the known CUDA control workloads (NOT recommended; disables instrument validation)")
    p.add_argument("--output", type=str, default="logs/f7_d6_attribution_probe_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D6_METRIC_ATTRIBUTION_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Classification -- 3 levels per Consultant Resolution (docs/private/F7_D6_CONSULTANT_RESOLUTION_01.md)
#   🟢 INSTRUMENT VALIDATED / 🟡 INCONCLUSIVE / 🔴 INSTRUMENT NOT VALIDATED
# Primary instrument validity criteria:
#   S0 spread <= 150 MB; control A/B relative diff <= 10%;
#   control A signal floor = max(3*S0_std, 150 MB);
#   workload measurable floor = max(3*S0_std, 150 MB);
#   cudaMemGetInfo qualitative coherence.
# Note: S3/S4 return is decoupled from sensor validity -> phenomenological characterization.
# ---------------------------------------------------------------------------

def _diff_pct(a: float, b: float) -> Optional[float]:
    if a is None or b is None or abs(a) < 1e-9:
        return None
    return abs(b - a) / abs(a) * 100.0


def classify_authorized(s0_stats, s3_stats, s4_stats, control_a, control_b,
                        s1_mb, s2_mb, s0_mb, meminfo_s1, meminfo_s2, meminfo_s3,
                        hard_gate_mb: float = HARD_GATE_VRAM_MB) -> Dict[str, Any]:
    """Return a dict with the 3-level classification and per-condition flags."""
    cond: Dict[str, Any] = {}

    s0_std = s0_stats.get("std_mb")
    s0_spread = s0_stats.get("spread_mb")
    cond["s0_stable"] = bool(s0_spread is not None and s0_spread <= 150.0)

    signal_floor = max(3.0 * (s0_std or 0.0), 150.0)
    cond["signal_floor_mb"] = round(signal_floor, 1)

    da = control_a.get("delta_alloc_mb") if control_a else None
    db = control_b.get("delta_alloc_mb") if control_b else None
    cond["control_a_delta_mb"] = da
    cond["control_b_delta_mb"] = db
    cond["control_a_observable"] = bool(da is not None and da >= signal_floor)
    diff_abs = (abs(db - da) if (da is not None and db is not None) else None)
    diff_rel = _diff_pct(da, db) if (da is not None and db is not None) else None
    cond["control_abs_diff_mb"] = round(diff_abs, 1) if diff_abs is not None else None
    cond["control_rel_diff_pct"] = round(diff_rel, 2) if diff_rel is not None else None
    # Guard: if Control A is below floor, do NOT compute the 10% tolerance -> INCONCLUSIVE
    if not cond["control_a_observable"]:
        cond["control_consistent"] = False
        cond["control_guard_tripped"] = True
    else:
        cond["control_guard_tripped"] = False
        cond["control_consistent"] = bool(diff_rel is not None and diff_rel <= 10.0)

    # Workload increment: S2 relative to the operational baseline S1 (workload's own addition)
    work_inc = (s2_mb - s1_mb) if (s1_mb is not None and s2_mb is not None) else None
    cond["workload_increment_mb"] = round(work_inc, 1) if work_inc is not None else None
    cond["workload_measurable"] = bool(work_inc is not None and work_inc >= signal_floor)

    # S3 post-workload characterization (phenomenological per Consultant Resolution §2)
    s3_mean = s3_stats.get("mean_mb")
    s3_spread = s3_stats.get("spread_mb")
    cond["s3_mean_mb"] = s3_mean
    cond["s3_spread_mb"] = s3_spread
    s3_vs_s0 = (abs(s3_mean - s0_mb) if (s3_mean is not None and s0_mb is not None) else None)
    cond["s3_vs_s0_abs_mb"] = round(s3_vs_s0, 1) if s3_vs_s0 is not None else None
    s3_vs_s1 = (round(s3_mean - s1_mb, 1) if (s3_mean is not None and s1_mb is not None) else None)
    cond["s3_vs_s1_mb"] = s3_vs_s1
    cond["s3_stable"] = bool(s3_spread is not None and s3_spread <= 150.0)
    cond["s3_returns_to_baseline"] = bool(s3_vs_s0 is not None and s3_vs_s0 <= 200.0 and cond["s3_stable"])
    cond["s3_phenomenon"] = (
        "returns_to_baseline" if cond["s3_returns_to_baseline"]
        else "persistent_device_residency"
    )

    # cudaMemGetInfo coherence (qualitative/temporal): CUDA free should fall S1->S2 and
    # recover S2->S3, i.e. CUDA used should rise then fall, matching NVML direction.
    def _used(mi):
        return mi.get("used_mb") if mi else None
    u1, u2, u3 = _used(meminfo_s1), _used(meminfo_s2), _used(meminfo_s3)
    cuda_rise = (u1 is not None and u2 is not None and u2 >= u1)
    cuda_fall = (u2 is not None and u3 is not None and u3 <= u2)
    nvml_rise = (s1_mb is not None and s2_mb is not None and s2_mb >= s1_mb)
    cond["cudaMemGetInfo_coherent"] = bool(cuda_rise and cuda_fall and nvml_rise)

    # S4 final observation (confirmatory/informative, phenomenological)
    s4_mean = s4_stats.get("mean_mb")
    s4_spread = s4_stats.get("spread_mb")
    cond["s4_mean_mb"] = s4_mean
    cond["s4_spread_mb"] = s4_spread
    cond["s4_stable"] = bool(s4_spread is not None and s4_spread <= 150.0)
    cond["s4_vs_s3_abs_mb"] = (round(abs(s4_mean - s3_mean), 1)
                               if (s4_mean is not None and s3_mean is not None) else None)
    cond["s4_vs_s0_abs_mb"] = (round(abs(s4_mean - s0_mb), 1)
                               if (s4_mean is not None and s0_mb is not None) else None)

    # ---- 3-level verdict (Primary Instrument Validity per Resolution §2) ----
    if not cond["s0_stable"]:
        verdict = "INSTRUMENT NOT VALIDATED"
        reason = "S0 idle baseline unstable (spread > 150 MB)"
    elif cond["control_guard_tripped"]:
        verdict = "INCONCLUSIVE"
        reason = "Control A signal below measurement floor; relative tolerance not evaluated"
    elif not cond["control_consistent"]:
        verdict = "INSTRUMENT NOT VALIDATED"
        reason = f"Control A/B not reproducible (relative difference {cond.get('control_rel_diff_pct')}% > 10%)"
    elif not cond["workload_measurable"]:
        verdict = "INCONCLUSIVE"
        reason = "Wan workload increment not clearly measurable above the noise floor"
    elif not cond["cudaMemGetInfo_coherent"]:
        verdict = "INCONCLUSIVE"
        reason = "cudaMemGetInfo direction inconsistent with NVML"
    else:
        verdict = "INSTRUMENT VALIDATED"
        reason = ("S0 stable; controls consistent (<=10%); workload measurable; "
                  "cudaMemGetInfo coherent. S3/S4 characterized as system residency phenomenon.")

    cond["verdict"] = verdict
    cond["reason"] = reason
    return cond


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D6: NVML METRIC ATTRIBUTION PROBE")
    print("  (Characterizing idle/OS/WDDM baseline vs workload-induced device residency)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: {args.steps} (probe)")
    print(f"  Mode: {args.mode} | Seed: {args.seed}")
    print(f"  Hard Gate: <= {HARD_GATE_VRAM_MB:.0f} MB | D5 peak (device used): {D5_PEAK_NVML_MB:.1f} | D5 reserved: {D5_PEAK_RESERVED_MB:.0f}")
    print("  Governance: AUTHORIZED metrology probe (Consultant no.1-4); does NOT reclassify F7-D5 nor change the gate.")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print(f"  S0 idle: {args.s0_seconds:.0f} s | S3 post-workload: {args.s3_seconds:.0f} s | S4 final: {args.s4_seconds:.0f} s")
        print(f"  Control A/B ~{args.control_mb:.0f} MB ({'skipped (NOT recommended)' if args.skip_control else 'enabled'}); tolerance +/-10%")
        print(f"  {args.steps} DiT steps with seam+step-start release; VAE SKIPPED.")
        print("  Authorized protocol: S0 -> Control A -> init -> S1 -> Wan -> S3 -> Control B -> S4")
        print("  Classification: INSTRUMENT VALIDATED / INCONCLUSIVE / INSTRUMENT NOT VALIDATED")
        print("[DRY-RUN] Validated. Awaiting execution authorization.")
        return

    # ---- S0: idle baseline (60 s, 1 Hz), pre-context ----
    try:
        s0_stats = sample_stability(seconds=args.s0_seconds, interval_ms=1000.0, record_series=True)
    except Exception as exc:
        s0_stats = {"error": str(exc)}
        print(f"[WARN] S0 stability sampling failed: {exc}")
    S0 = round(s0_stats.get("mean_mb", 0.0), 2)
    so_spread = s0_stats.get("spread_mb")
    pre_procs = nvml_process_summary()
    print(f"\n  S0 idle_baseline (device used, pre-context) mean : {S0} MB "
          f"(min {s0_stats.get('min_mb')} / max {s0_stats.get('max_mb')} / median {s0_stats.get('median_mb')} / "
          f"std {s0_stats.get('std_mb')} / spread {so_spread} MB)")
    print(f"     processes: compute_n={pre_procs['compute_n']} attrib_mb={pre_procs['compute_attrib_mb']:.1f} | "
          f"graphics_n={pre_procs['graphics_n']} attrib_mb={pre_procs['graphics_attrib_mb']:.1f}")
    print(f"     host memory: working_set={pre_procs.get('host_process_memory', {}).get('working_set_mb')} MB | "
          f"commit_private={pre_procs.get('host_process_memory', {}).get('commit_private_mb')} MB")
    print("     note: usedGpuMemory=None means the counter is UNAVAILABLE under WDDM, NOT 0 MB")

    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # ---- CUDA Context Init + Deterministic Warm-up (~10 MB), per Resolution §3 ----
    print("  Initializing CUDA context + deterministic warm-up (~10 MB) ...")
    torch.cuda.init()
    _dummy = torch.empty((10 * 1024 * 1024 // 4,), dtype=torch.float32, device=args.device)
    torch.cuda.synchronize(args.device)
    del _dummy
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    print("  CUDA context initialized & warm-up freed.")

    # ---- Control A (POST CUDA init, PRE pipeline load), per Resolution §3 ----
    control_a = None
    if not args.skip_control:
        print(f"  [Control A] known CUDA control workload ~{args.control_mb:.0f} MB (context initialized, pre-pipeline) ...")
        control_a = run_control_workload(mb=args.control_mb, device=args.device)
        print(f"  [Control A] pre={control_a.get('pre_mb')} during={control_a.get('during_mb')} "
              f"post={control_a.get('post_mb')} | delta_alloc={control_a.get('delta_alloc_mb')} "
              f"delta_return={control_a.get('delta_return_mb')}")
        gc.collect(); torch.cuda.synchronize(args.device); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)

    pipeline = None
    probe = None
    control_b = None
    s3_stats: Dict[str, Any] = {}
    s4_stats: Dict[str, Any] = {}
    error_occurred: Optional[str] = None
    S1 = S2 = S3 = None
    meminfo_s1 = meminfo_s2 = meminfo_s3 = meminfo_s4 = None
    t_start = time.perf_counter()
    try:
        # ---- init PyTorch + pipeline ----
        pipeline = AttributionProbePipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
        gc.collect(); torch.cuda.synchronize(args.device); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)
        # ---- S1: operational baseline (system + CUDA/PyTorch + loaded pipeline) ----
        S1 = round(nvml_device_mb(), 2)
        meminfo_s1 = cuda_mem_getinfo_mb()
        print(f"  S1 operational_baseline (post-init): {S1} MB | cudaMemGetInfo={meminfo_s1}")
        # ---- Wan workload (3 DiT steps) ----
        probe = pipeline.run_probe(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode=args.mode, nvml=nvml)
        meminfo_s2 = (probe or {}).get("meminfo_workload_peak")
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D6 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        S2 = nvml.stop()

    # ---- S3: post-workload 120 s stability & trajectory sampling (1 Hz), per Resolution §5 ----
    if error_occurred is None:
        try:
            gc.collect(); torch.cuda.synchronize(args.device); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)
            s3_stats = sample_stability(seconds=args.s3_seconds, interval_ms=1000.0, record_series=True)
            S3 = s3_stats.get("mean_mb")
            meminfo_s3 = cuda_mem_getinfo_mb()
            print(f"  S3 post_workload ({args.s3_seconds:.0f} s, 1 Hz) mean {S3} MB "
                  f"(spread {s3_stats.get('spread_mb')} MB) | cudaMemGetInfo={meminfo_s3}")
        except Exception as exc:
            print(f"[WARN] S3 sampling failed: {exc}")

        # ---- Control B (after workload), per Resolution §3/§6 ----
        if not args.skip_control:
            print(f"  [Control B] known CUDA control workload ~{args.control_mb:.0f} MB (post-workload) ...")
            control_b = run_control_workload(mb=args.control_mb, device=args.device)
            print(f"  [Control B] pre={control_b.get('pre_mb')} during={control_b.get('during_mb')} "
                  f"post={control_b.get('post_mb')} | delta_alloc={control_b.get('delta_alloc_mb')} "
                  f"delta_return={control_b.get('delta_return_mb')}")
            gc.collect(); torch.cuda.synchronize(args.device); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)

        # ---- S4: final 60 s observation (1 Hz), per Resolution §5 ----
        try:
            gc.collect(); torch.cuda.synchronize(args.device); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)
            s4_stats = sample_stability(seconds=args.s4_seconds, interval_ms=1000.0, record_series=True)
            meminfo_s4 = cuda_mem_getinfo_mb()
            print(f"  S4 final_observation ({args.s4_seconds:.0f} s, 1 Hz) mean {s4_stats.get('mean_mb')} MB "
                  f"(spread {s4_stats.get('spread_mb')} MB) | cudaMemGetInfo={meminfo_s4}")
        except Exception as exc:
            print(f"[WARN] S4 sampling failed: {exc}")

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB.")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()
    post_procs = nvml_process_summary()

    ctx_overhead = round(S1 - S0, 1) if S1 is not None else None
    workload_increment = round(S2 - S1, 1) if (S1 is not None and S2 is not None) else None
    workload_delta = round(S2 - S0, 1) if S2 is not None else None   # NOT proven ownership
    so_share = round(S0, 1)

    cond = classify_authorized(s0_stats, s3_stats, s4_stats, control_a, control_b,
                               S1, S2, S0, meminfo_s1, meminfo_s2, meminfo_s3)
    result = cond.get("verdict", "INCONCLUSIVE")

    print("\n" + "=" * 80)
    print("  PHASE F7-D6: INSTRUMENT VALIDATION SCORECARD")
    print("=" * 80)
    print(f"  S0 idle_baseline (pre-context)    [MEASURED] : {so_share:8.1f} MB (spread {so_spread} MB)")
    print(f"  Control A delta                   [MEASURED] : {cond.get('control_a_delta_mb')} MB")
    print(f"  S1 operational_baseline           [MEASURED] : {S1} MB (post-init)")
    print(f"  S2 workload_peak (device used)    [MEASURED] : {S2} MB (D5: {D5_PEAK_NVML_MB:.1f})")
    print(f"  S3 post_workload ({args.s3_seconds:.0f} s) mean      [MEASURED] : {cond.get('s3_mean_mb')} MB (spread {cond.get('s3_spread_mb')})")
    print(f"  Control B delta                   [MEASURED] : {cond.get('control_b_delta_mb')} MB")
    print(f"  S4 final ({args.s4_seconds:.0f} s) mean           [MEASURED] : {cond.get('s4_mean_mb')} MB (spread {cond.get('s4_spread_mb')})")
    print(f"  workload increment (S2-S1)        [DERIVED]  : {workload_increment} MB")
    print(f"  WORKLOAD-INDUCED device delta     [DERIVED]  : {workload_delta} MB (S2-S0; NOT proven ownership)")
    print(f"  signal floor                      [DERIVED]  : {cond.get('signal_floor_mb')} MB")
    print(f"  control rel diff                  [DERIVED]  : {cond.get('control_rel_diff_pct')} % (<=10 -> {cond.get('control_consistent')})")
    print(f"  S3 returns to baseline            [DERIVED]  : {cond.get('s3_returns_to_baseline')} (|S3-S0|={cond.get('s3_vs_s0_abs_mb')} MB)")
    print(f"  cudaMemGetInfo coherent           [DERIVED]  : {cond.get('cudaMemGetInfo_coherent')}")
    print(f"  Peak Reserved (torch)             [OBSERVED] : {peak_reserved:8.1f} MB (D5: {D5_PEAK_RESERVED_MB:.0f})")
    print(f"  Peak Allocated (torch)            [OBSERVED] : {peak_alloc:8.1f} MB")
    print(f"  Host RSS                          [OBSERVED] : {peak_ram:8.1f} MB")
    print(f"  Wall-clock                        [OBSERVED] : {t_total:8.2f} s")
    print(f"  VERDICT                            [DERIVED]  : {result}")
    print(f"  reason                             [DERIVED]  : {cond.get('reason')}")
    print(f"  Abort/Error                       : {error_occurred or 'none'}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D6 (NVML metric attribution probe)",
        "evidence_class": "All measured/observed; derived computed",
        "governance_note": ("Metrology per External Consultant authorizations no.1-4: NOT a reclassification of "
                            "F7-D5 and NOT a gate change. S2-S0 is 'workload-induced device residency delta', "
                            "not proven process ownership."),
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": args.mode, "seed": args.seed},
        "s0_stability": s0_stats,
        "s3_stability": s3_stats,
        "s4_stability": s4_stats,
        "milestones_mb": {"S0_idle_baseline": so_share,
                          "S1_operational_baseline": S1,
                          "S2_workload_peak": round(S2, 1) if S2 is not None else None,
                          "S3_post_workload_mean": cond.get("s3_mean_mb"),
                          "S4_final_mean": cond.get("s4_mean_mb")},
        "derived_mb": {"ctx_overhead": ctx_overhead,
                       "workload_increment_S2_S1": workload_increment,
                       "workload_induced_device_residency_delta": workload_delta,
                       "so_share": so_share},
        "control_A": control_a,
        "control_B": control_b,
        "classification": cond,
        "cross_checks_mb": {"peak_torch_reserved": round(peak_reserved, 1),
                            "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_process_ram": round(peak_ram, 1),
                            "cudaMemGetInfo_S1": meminfo_s1,
                            "cudaMemGetInfo_S2_peak": meminfo_s2,
                            "cudaMemGetInfo_S3": meminfo_s3,
                            "cudaMemGetInfo_S4": meminfo_s4},
        "per_step": (probe or {}).get("step_end_reserved_mb"),
        "denoise_time_s": (probe or {}).get("denoise_time_s"),
        "processes_pre": pre_procs,
        "processes_post": post_procs,
        "comparison": {"d5_peak_nvml_device_used_mb": D5_PEAK_NVML_MB,
                       "d5_peak_torch_reserved_mb": D5_PEAK_RESERVED_MB},
        "result_class": result,
        "verdict": {"execution_completed": error_occurred is None,
                    "error_detail": error_occurred, **{k: v for k, v in cond.items() if k != "verdict"}},
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
        w("**Status:** measurement-metrology probe (per External Consultant authorizations no.1-4). "
          "Does NOT reclassify F7-D5 and does NOT change the gate.  \n")
        w("---\n\n## 1. Instrument Validation Result\n\n")
        w(f"- **VERDICT: {result}**\n")
        w(f"- Reason: {cond.get('reason')}\n")
        w("\n### 1.1 Milestones\n\n")
        w(f"- S0 idle_baseline (60 s): mean {so_share:.1f} MB "
          f"(min {s0_stats.get('min_mb')} / max {s0_stats.get('max_mb')} / median {s0_stats.get('median_mb')} / "
          f"std {s0_stats.get('std_mb')} / spread {so_spread})\n")
        w(f"- S1 operational_baseline (post-init): {S1} MB\n")
        w(f"- S2 workload_peak (device used): {S2} MB (D5: {D5_PEAK_NVML_MB:.1f})\n")
        w(f"- S3 post_workload mean: {cond.get('s3_mean_mb')} MB (spread {cond.get('s3_spread_mb')})\n")
        w(f"- S4 final mean: {cond.get('s4_mean_mb')} MB (spread {cond.get('s4_spread_mb')})\n")
        w(f"- workload increment (S2-S1): {workload_increment} MB\n")
        w(f"- **workload-induced device residency delta (S2-S0): {workload_delta} MB** (NOT proven ownership)\n")
        w("\n### 1.2 Conditions\n\n")
        for k in ("s0_stable", "control_a_observable", "control_consistent", "workload_measurable",
                  "cudaMemGetInfo_coherent", "s3_returns_to_baseline", "s3_phenomenon", "s4_stable"):
            w(f"- {k}: {cond.get(k)}\n")
        w(f"- control A delta: {cond.get('control_a_delta_mb')} MB | control B delta: {cond.get('control_b_delta_mb')} MB "
          f"| rel diff: {cond.get('control_rel_diff_pct')} %\n")
        w(f"- signal floor: {cond.get('signal_floor_mb')} MB\n")
        w(f"- Peak Reserved (torch): {peak_reserved:.1f} MB (D5: {D5_PEAK_RESERVED_MB:.0f})\n")
        w("\n## 2. Interpretation\n\n")
        w("The device-wide `nvmlDeviceGetMemoryInfo().used` metric does not start at zero: "
          "an idle baseline (S0) is observed even with no Marley workload. `usedGpuMemory=None` "
          "per PID under WDDM means the counter is UNAVAILABLE, not 0 MB. The quantity `S2 - S0` "
          "is a **workload-induced device residency delta** (device-wide growth), NOT proven "
          "process ownership; it is cross-checked against `torch.reserved` and validated by known "
          "CUDA control workloads (A pre-workload, B post-workload).\n")
        w("\n## 3. Falsification scenarios (remain open)\n\n")
        w("1. S0 varies by hundreds of MB with system state -> subtraction not robust.\n")
        w("2. Desktop/WDDM grows its own residency during the workload -> S2-S0 overestimates Marley.\n")
        w("3. Marley induces driver allocations not in torch.reserved -> NVML-reserved may be real Marley memory.\n")
        w("4. Device-wide increment grows while reserved stays flat -> investigate driver/WDDM/contexts/shared memory.\n")
        w("5. Full 30-step run increases the delta vs the short probe -> 3-step probe insufficient.\n")
        w("\n## 4. Governance\n\n")
        w("Observational metrology probe in an isolated runner (VAE skipped, reduced steps). "
          "Does NOT reclassify F7-D5 and authorizes NO runtime modification and NO gate change. "
          "Two-layer metric under study (A: device-wide raw, always reported; B: incremental workload "
          "residency, label pending validation). Any redefinition of the ground-truth metric of "
          "ROADMAP section 2 is a separate Director/Consejero decision.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
