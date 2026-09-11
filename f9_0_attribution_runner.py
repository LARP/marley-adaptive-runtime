"""
f9_0_attribution_runner.py
==========================
Phase F9-0 -- Preregistered Memory Peak Attribution Diagnostic.

Protocol (frozen, authoritative):
  docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md
Charter:
  docs/F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md

Nature
------
This runner is DIAGNOSTIC, not optimization. It is observational and reversible:
it does NOT modify the runtime, does NOT change the 4,800 MB gate, and does NOT
reclassify F7-D5 / F7 Stage B / F8. It reuses the validated F7-D6 instrument
(baseline S0, S1, ~500 MB controls A/B, cudaMemGetInfo, temp/clocks/utilization).

Frozen reference: F7-D5 = 5,096.1 MB (peak_nvml_used), deficit +296.1 MB vs gate.

Protocol execution (per run, §4)
--------------------------------
  S0  idle baseline (60 s @ 1 Hz), pre-CUDA-context, exclusivity check (§2.3)
  -> init CUDA context + first kernel (deterministic warm-up)
  -> Control A (~500 MB) [F7-D6 instrument]
  -> S1 operational baseline (pre-model-load)
     Overhead_CUDA/Driver/Runtime = S1 - S0                      (category 6)
  -> load pipeline (persistent non-block DiT weights snapshot)
  -> canonical workload: 30 DiT steps, adaptive, seam + step-start drain
     (VAE measured separately, never mixed with the DiT peak)
  -> Peak_NVML_raw, Delta_induced, allocator snapshots, CUDA event windows
  -> S3 post-workload (60 s @ 1 Hz) + Control B (~500 MB)
  -> release / cooldown

  N >= 5 independent runs (clean subprocess per run), cooldown between runs,
  execution order recorded. Median, dispersion and CI reported.

Attribution model (7 mutually exclusive categories, §2.2)
--------------------------------------------------------
All quantities are evaluated at the NVML peak instant (allocated/reserved
linearly interpolated from synchronized snapshots):

  1 pesos residentes    = persistent_non_block_weights + streaming_slot_resident
  2 activaciones        = allocated_at_peak - (1)
  3 workspace kernels   = max NVML excess above (S0 + overhead + reserved)
                          within CUDA-event-bounded windows (transient)
  4 memoria del allocator = reserved_at_peak - allocated_at_peak
  5 fragmentacion       = inactive_split (SUBSET of 4, reported, NOT additive)
  6 overhead CUDA/driver/runtime = S1 - S0
  7 otros/transitorios  = Delta_induced - sum(1..6)   (residual, never split)

Closure (§5): CLOSED - ATTRIBUTED when >=90 % of Delta_induced is explained,
the peak window is localized, exclusivity holds, and no peak underestimation.
Max 2 instrumentation iterations; after that -> ATTRIBUTION INCOMPLETE - DOCUMENTED.

Governance: NO run is launched without explicit Director authorization.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline
from marley.core.adaptive import AdaptiveEngine

MB = 1024.0 * 1024.0

HARD_GATE_VRAM_MB = 4800.0
SAFE_ABORT_NVML_MB = 5050.0
F7_D5_PEAK_NVML_MB = 5096.1
F7_D5_DEFICIT_MB = 296.1
F7_D5_PEAK_RESERVED_MB = 3776.0
NVML_INTERVAL_MS = 20.0
NVML_MIN_ACCEPTABLE_HZ = 10.0
S0_STABILITY_SPREAD_MB = 150.0
ATTRIBUTION_CLOSURE_FRACTION = 0.90

CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
CANONICAL_FPS = 16
DEFAULT_STEPS = 30
DEFAULT_FRAMES = 33
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_RUNS = 5
DEFAULT_S0_SECONDS = 60.0
DEFAULT_S3_SECONDS = 60.0
DEFAULT_COOLDOWN_SECONDS = 120.0
DEFAULT_CONTROL_MB = 500.0


# ---------------------------------------------------------------------------
# Host / device probes
# ---------------------------------------------------------------------------

def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def get_process_host_memory_mb() -> Dict[str, float]:
    mem = psutil.Process(os.getpid()).memory_info()
    return {
        "working_set_mb": round(mem.rss / MB, 2),
        "commit_private_mb": round(mem.vms / MB, 2),
    }


def nvml_device_mb() -> float:
    """Instant read of device-wide NVML `used` (MB)."""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / MB


def cuda_mem_getinfo_mb() -> Optional[Dict[str, float]]:
    """Process-visible CUDA pool from cudaMemGetInfo."""
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "free_mb": round(free_b / MB, 2),
            "total_mb": round(total_b / MB, 2),
            "used_mb": round((total_b - free_b) / MB, 2),
        }
    except Exception:
        return None


def pid_has_cuda_context(pid: int) -> Optional[bool]:
    """Windows-only equivalent mechanism: does `pid` hold a real CUDA driver context?

    On Windows/WDDM, `nvmlDeviceGetComputeRunningProcesses` conflates CUDA compute
    clients with D3D graphics clients (all reported as "C+G" with usedGpuMemory=None),
    so it cannot by itself distinguish a real CUDA compute context from DWM/browser
    graphics residency. A process that holds a CUDA context has `nvcuda.dll` loaded;
    pure graphics clients do not. Returns True/False, or None when undeterminable
    (e.g. access denied on a protected process).
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010
        LIST_MODULES_ALL = 0x03
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EnumProcessModulesEx.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        psapi.GetModuleBaseNameW.argtypes = [
            wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        psapi.GetModuleBaseNameW.restype = wintypes.DWORD

        h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not h:
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            needed = wintypes.DWORD()
            if not psapi.EnumProcessModulesEx(h, None, 0, ctypes.byref(needed), LIST_MODULES_ALL):
                return None
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
            if count <= 0:
                return False
            mods = (wintypes.HMODULE * count)()
            if not psapi.EnumProcessModulesEx(h, mods, needed.value, ctypes.byref(needed), LIST_MODULES_ALL):
                return None
            buf = ctypes.create_unicode_buffer(512)
            for m in mods:
                if psapi.GetModuleBaseNameW(h, m, buf, 512):
                    if buf.value.lower() in ("nvcuda.dll",):
                        return True
            return False
        finally:
            k32.CloseHandle(h)
    except Exception:
        return None


def nvml_process_summary() -> Dict[str, Any]:
    """Enumerate compute/graphics processes and quantify exclusivity (§2.3).

    `exclusive_other_contexts` is TRUE when no process other than this one holds a
    real CUDA compute context. Desktop/DWM graphics residency does NOT invalidate
    the run: it is quantified as S0_idle and reported.
    """
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    my_pid = os.getpid()
    out: Dict[str, Any] = {
        "compute_n": 0, "compute_attrib_mb": 0.0,
        "graphics_n": 0, "graphics_attrib_mb": 0.0,
        "target_pid": my_pid, "target_pid_found": False, "target_pid_gpu_mem": None,
        "other_compute_pids": [], "other_graphics_pids": [],
        "unknown_context_pids": [],
        "compute_enumeration_ok": False,
        "exclusive_other_contexts": None,
        "detection_method": ("NVML compute list refined by real CUDA context detection "
                             "(nvcuda.dll loaded in process); graphics/DWM does not invalidate"),
    }
    candidate_pids: List[int] = []
    for key_n, key_mb, fn in (
        ("compute_n", "compute_attrib_mb", pynvml.nvmlDeviceGetComputeRunningProcesses),
        ("graphics_n", "graphics_attrib_mb", pynvml.nvmlDeviceGetGraphicsRunningProcesses),
    ):
        is_compute = "compute" in key_n
        try:
            procs = list(fn(handle))
            if is_compute:
                out["compute_enumeration_ok"] = True
            for p in procs:
                out[key_n] += 1
                if p.pid == my_pid:
                    out["target_pid_found"] = True
                    out["target_pid_gpu_mem"] = (
                        round(p.usedGpuMemory / MB, 2)
                        if p.usedGpuMemory is not None else "UNAVAILABLE_WDDM"
                    )
                elif p.pid not in candidate_pids:
                    candidate_pids.append(p.pid)
                if p.usedGpuMemory:
                    out[key_mb] += p.usedGpuMemory / MB
        except Exception as exc:
            if is_compute:
                out["compute_enumeration_ok"] = False
                out["exclusive_other_contexts"] = None
                out["compute_enumeration_error"] = str(exc)

    # Refine candidates: only real CUDA compute contexts break exclusivity.
    if out["compute_enumeration_ok"]:
        out["exclusive_other_contexts"] = True
        for pid in candidate_pids:
            has_cuda = pid_has_cuda_context(pid)
            if has_cuda is True:
                out["other_compute_pids"].append(pid)
                out["exclusive_other_contexts"] = False
            elif has_cuda is None:
                out["unknown_context_pids"].append(pid)
            else:
                out["other_graphics_pids"].append(pid)
    out["host_process_memory"] = get_process_host_memory_mb()
    out["note"] = ("usedGpuMemory=None/UNAVAILABLE under WDDM means the counter is "
                   "UNAVAILABLE, NOT that the process uses 0 MB")
    return out


def sample_stability(seconds: float, interval_ms: float = 1000.0,
                     record_series: bool = True) -> Dict[str, Any]:
    """Sample NVML device `used` for `seconds` and return statistics + trajectory."""
    samples: List[float] = []
    series: List[Dict[str, Any]] = []
    n = max(1, int(seconds * 1000.0 / interval_ms))
    t0 = time.perf_counter()
    for _ in range(n):
        mb = nvml_device_mb()
        samples.append(mb)
        if record_series:
            series.append({"t_s": round(time.perf_counter() - t0, 3), "nvml_used_mb": mb})
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


def run_control_workload(mb: float = DEFAULT_CONTROL_MB, device: str = "cuda:0") -> Dict[str, Any]:
    """Known CUDA control: allocate ~mb, compute, free, sync + empty_cache.

    Validates that the incremental NVML instrument responds proportionally
    (reused from the F7-D6 validated instrument).
    """
    out: Dict[str, Any] = {"requested_mb": mb}
    dev = torch.device(device)
    try:
        n_el = int(mb * MB // 4)
        pre = round(nvml_device_mb(), 2)
        t = torch.zeros((n_el,), dtype=torch.float32, device=dev)
        torch.cuda.synchronize(dev)
        during = round(nvml_device_mb(), 2)
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


# ---------------------------------------------------------------------------
# NVML high-frequency sampler (20 ms / 50 Hz) with phase marking
# ---------------------------------------------------------------------------

class NVMLSampler:
    """High-frequency physical VRAM sampler + temp/clock/utilization.

    Records the full device-wide `used` series with timestamps, phase markers,
    per-sample polling cost, and the localized peak window.
    """

    def __init__(self, device_index: int = 0, interval_ms: float = NVML_INTERVAL_MS,
                 safe_abort_mb: Optional[float] = SAFE_ABORT_NVML_MB) -> None:
        self.interval_s = interval_ms / 1000.0
        self.requested_hz = 1000.0 / interval_ms
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._handle = None
        self.peak_mb = 0.0
        self.peak_timestamp: Optional[str] = None
        self.peak_t_perf: Optional[float] = None
        self.safe_abort_triggered = False
        self.safe_abort_mb = safe_abort_mb
        self.series: List[Dict[str, Any]] = []
        self.phase_marks: List[Dict[str, Any]] = []
        self._sample_costs_ms: List[float] = []
        self._sample_i = 0
        self._thermal_every = max(1, int(1.0 / self.interval_s))  # ~1 Hz thermal sampling
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
            return info.used / MB
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
            t0 = time.perf_counter()
            try:
                used_mb = self._read_used()
                t1 = time.perf_counter()
                entry: Dict[str, Any] = {"t_perf": t1, "t_iso": datetime.datetime.now().isoformat(),
                                         "nvml_used_mb": round(used_mb, 2)}
                self._sample_i += 1
                if self._sample_i % self._thermal_every == 0:
                    thermal = self.sample_instant()
                    entry["gpu_temp_c"] = thermal.get("gpu_temp_c")
                    entry["gpu_clock_mhz"] = thermal.get("gpu_clock_mhz")
                    entry["gpu_util_pct"] = thermal.get("gpu_util_pct")
                with self._lock:
                    self.series.append(entry)
                    if used_mb > self.peak_mb:
                        self.peak_mb = used_mb
                        self.peak_timestamp = datetime.datetime.now().isoformat()
                        self.peak_t_perf = t1
                    if self.safe_abort_mb is not None and used_mb > self.safe_abort_mb:
                        self.safe_abort_triggered = True
                self._sample_costs_ms.append((t1 - t0) * 1000.0)
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self.interval_s - elapsed))

    def mark_phase(self, label: str) -> None:
        with self._lock:
            self.phase_marks.append({"label": label, "t_perf": time.perf_counter(),
                                     "t_iso": datetime.datetime.now().isoformat(),
                                     "nvml_used_mb": round(self._read_used(), 2)})

    def reset_peak(self) -> None:
        with self._lock:
            self.peak_mb = self._read_used()
            self.peak_timestamp = None
            self.peak_t_perf = None

    def sample_instant(self) -> Dict[str, Any]:
        out = {"nvml_used_mb": 0.0, "gpu_temp_c": None, "gpu_clock_mhz": None, "gpu_util_pct": None}
        if not self._available or self._handle is None:
            return out
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            out["nvml_used_mb"] = round(info.used / MB, 2)
            out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            out["gpu_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
            out["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        except Exception:
            pass
        return out

    def achieved_hz(self) -> Optional[float]:
        if len(self.series) < 2:
            return None
        span = self.series[-1]["t_perf"] - self.series[0]["t_perf"]
        if span <= 0:
            return None
        return round((len(self.series) - 1) / span, 2)

    def polling_cost_stats(self) -> Dict[str, Any]:
        costs = list(self._sample_costs_ms)
        if not costs:
            return {}
        return {
            "n": len(costs),
            "mean_ms": round(statistics.fmean(costs), 4),
            "median_ms": round(statistics.median(costs), 4),
            "max_ms": round(max(costs), 4),
        }

    def peak_window(self, half_window_s: float = 1.0,
                    t_perf: Optional[float] = None) -> Optional[Dict[str, Any]]:
        t_ref = t_perf if t_perf is not None else self.peak_t_perf
        if t_ref is None:
            return None
        lo = t_ref - half_window_s
        hi = t_ref + half_window_s
        with self._lock:
            local = [s for s in self.series if lo <= s["t_perf"] <= hi]
        if not local:
            return None
        return {
            "start_iso": local[0]["t_iso"],
            "end_iso": local[-1]["t_iso"],
            "start_t_perf": local[0]["t_perf"],
            "end_t_perf": local[-1]["t_perf"],
            "n_samples": len(local),
            "local_min_mb": min(s["nvml_used_mb"] for s in local),
            "local_max_mb": max(s["nvml_used_mb"] for s in local),
        }

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb

    def export_series(self, downsample: int = 1,
                      upto_t_perf: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._lock:
            series = list(self.series)
        if upto_t_perf is not None:
            series = [s for s in series if s["t_perf"] <= upto_t_perf]
        if downsample <= 1:
            return series
        return series[::downsample]


# ---------------------------------------------------------------------------
# Allocator / residency accounting
# ---------------------------------------------------------------------------

_ALLOC_STAT_KEYS = {
    "allocated_current": "allocated_bytes.all.current",
    "reserved_current": "reserved_bytes.all.current",
    "active_current": "active_bytes.all.current",
    "inactive_split_current": "inactive_split_bytes.all.current",
    "inactive_split_all_blocks": "inactive_split.all.current",
    "segment_current": "segment.all.current",
    "requested_current": "requested_bytes.all.current",
}


def allocator_snapshot(device: torch.device) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / MB, 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / MB, 2),
        "t_perf": time.perf_counter(),
        "t_iso": datetime.datetime.now().isoformat(),
    }
    try:
        stats = torch.cuda.memory_stats(device)
        for name, key in _ALLOC_STAT_KEYS.items():
            if key in stats:
                snap[name + "_mb"] = round(stats[key] / MB, 2)
    except Exception:
        pass
    snap["fragmentation_mb"] = snap.get("inactive_split_current_mb")
    return snap


def module_param_bytes(module: nn.Module, dtype: torch.dtype) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    total = 0
    for p in module.parameters():
        total += p.numel() * elem
    for b in module.buffers():
        total += b.numel() * elem
    return total


def streamer_resident_bytes(streamer: Any) -> int:
    """Sum bytes of all GPU weight slots currently allocated by the streamer(s)."""
    candidates: List[Any] = [streamer]
    if hasattr(streamer, "_fp16"):
        candidates = [streamer._fp16, streamer._int8]
    total = 0
    for s in candidates:
        for slot in getattr(s, "gpu_slots", []) or []:
            for t in slot.values():
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    return total


# ---------------------------------------------------------------------------
# Canonical workload with attribution instrumentation
# ---------------------------------------------------------------------------

class F9AttributionPipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched). Replicates the F7-D5 FAVORABLE
    configuration (adaptive, step-start drain + cond->uncond seam release)
    and captures allocator snapshots and CUDA event windows at every boundary.
    """

    def run_attribution(
        self, prompt: str, negative_prompt: str, height: int, width: int,
        num_frames: int, num_steps: int, guidance_scale: float, seed: int,
        mode: str, nvml: NVMLSampler, output_video_path: Optional[str] = None,
        export_video: bool = False,
    ) -> Dict[str, Any]:
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

        snapshots: List[Dict[str, Any]] = []
        cuda_events: List[Dict[str, Any]] = []
        step_times: List[float] = []
        step_end_reserved: List[float] = []
        step_end_alloc: List[float] = []
        seam_count = 0
        step_start_count = 0
        peak_streamer_resident_mb = 0.0

        def _snap(phase: str, step: int, pas: str) -> Dict[str, Any]:
            nonlocal peak_streamer_resident_mb
            s = allocator_snapshot(device)
            s["phase"] = phase
            s["step"] = step
            s["pass"] = pas
            s["nvml_mb"] = nvml.sample_instant()["nvml_used_mb"]
            s["meminfo"] = cuda_mem_getinfo_mb()
            resident = streamer_resident_bytes(streamer) / MB
            s["streamer_resident_mb"] = round(resident, 2)
            peak_streamer_resident_mb = max(peak_streamer_resident_mb, resident)
            snapshots.append(s)
            return s

        def _event(label: str, step: int, pas: str) -> None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            cuda_events.append({"label": label, "step": step, "pass": pas,
                                "t_perf": time.perf_counter(), "event": ev})

        denoise_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        nvml.mark_phase("dit_loop_start")
        try:
            for step_idx, t in enumerate(timesteps):
                ts = time.perf_counter()
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                step_start_count += 1
                _event("step_start", step_idx + 1, "pre")
                _snap("step_start", step_idx + 1, "pre")

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
                shift = shift.to(hidden_states_cond.device)
                scale = scale.to(hidden_states_cond.device)
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
                _event("seam_cond_uncond", step_idx + 1, "cond")
                _snap("seam", step_idx + 1, "cond")

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

                step_s = time.perf_counter() - ts
                step_times.append(step_s)
                snap = _snap("step_end", step_idx + 1, "uncond")
                step_end_reserved.append(snap["reserved_mb"])
                step_end_alloc.append(snap["allocated_mb"])
                _event("step_end", step_idx + 1, "uncond")
                print(f"   Step [{step_idx+1}/{num_steps}] {step_s:.2f}s | end Reserved={snap['reserved_mb']:.1f} "
                      f"Alloc={snap['allocated_mb']:.1f} NVML={snap['nvml_mb']:.1f} (peak so far {nvml.peak_mb:.1f})")

        finally:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
            torch.cuda.empty_cache()
            self.transformer.blocks = transformer_blocks_orig

        # Resolve CUDA event windows (GPU timeline between consecutive boundaries).
        torch.cuda.synchronize(device)
        for i in range(1, len(cuda_events)):
            try:
                cuda_events[i]["elapsed_ms_since_prev"] = round(
                    cuda_events[i - 1]["event"].elapsed_time(cuda_events[i]["event"]), 3)
            except Exception:
                cuda_events[i]["elapsed_ms_since_prev"] = None
        for ev in cuda_events:
            ev.pop("event", None)

        denoise_s = time.perf_counter() - denoise_start
        nvml.mark_phase("dit_loop_end")

        # ---- Capture the DiT-loop peak BEFORE the separate VAE phase ----
        dit_peak_mb = round(nvml.peak_mb, 2)
        dit_peak_t_perf = nvml.peak_t_perf
        dit_peak_iso = nvml.peak_timestamp
        dit_nvml_series = nvml.export_series(upto_t_perf=dit_peak_t_perf) \
            if dit_peak_t_perf is not None else nvml.export_series()
        nvml.mark_phase("dit_peak_captured")
        nvml.reset_peak()

        out["dit_peak_nvml_mb"] = dit_peak_mb
        out["dit_peak_t_perf"] = dit_peak_t_perf
        out["dit_peak_iso"] = dit_peak_iso
        out["dit_nvml_series"] = dit_nvml_series
        out["dit_torch_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated(device) / MB, 2)
        out["dit_torch_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / MB, 2)
        out["step_times_s"] = [round(x, 2) for x in step_times]
        out["step_end_reserved_mb"] = step_end_reserved
        out["step_end_alloc_mb"] = step_end_alloc
        out["denoise_time_s"] = round(denoise_s, 2)
        out["cadence_avg_s"] = round(sum(step_times) / len(step_times), 2) if step_times else None
        out["seam_count"] = seam_count
        out["step_start_count"] = step_start_count
        out["snapshots"] = snapshots
        out["cuda_events"] = cuda_events
        out["peak_streamer_resident_mb"] = round(peak_streamer_resident_mb, 2)
        out["latents"] = latents.detach().float().cpu()
        out["has_nan"] = bool(torch.isnan(out["latents"]).any().item())
        out["has_inf"] = bool(torch.isinf(out["latents"]).any().item())
        out["total_wall_s"] = round(time.perf_counter() - t_global, 2)

        # ---- VAE decode (separate phase; its peak is NOT mixed with DiT) ----
        out["vae"] = {}
        if output_video_path is not None or export_video:
            vae_start = time.perf_counter()
            nvml.mark_phase("vae_start")
            latents_vae = latents.to(self.vae_dtype)
            latents_mean = (torch.tensor(self.vae.config.latents_mean)
                            .view(1, self.vae.config.z_dim, 1, 1, 1).to(latents_vae.device, latents_vae.dtype))
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1).to(latents_vae.device, latents_vae.dtype)
            latents_vae = latents_vae / latents_std + latents_mean
            self.vae.to(device)
            video = self.vae.decode(latents_vae, return_dict=False)[0]
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            nvml.mark_phase("vae_end")
            out["vae"]["decode_s"] = round(time.perf_counter() - vae_start, 2)
            out["vae"]["video_shape"] = list(video.shape)
            out["vae"]["nan_inf"] = bool(torch.isnan(video).any() or torch.isinf(video).any())
            if export_video and output_video_path is not None:
                try:
                    from diffusers.utils import export_to_video
                    frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")
                    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                    export_to_video(frames[0], output_video_path, fps=CANONICAL_FPS)
                    out["vae"]["mp4_ok"] = os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0
                    out["vae"]["output_video"] = output_video_path
                except Exception as exc:
                    out["vae"]["mp4_ok"] = False
                    out["vae"]["export_error"] = str(exc)
            else:
                out["vae"]["mp4_ok"] = None
            del video
            gc.collect()
            out["vae"]["vae_peak_nvml_mb"] = round(nvml.peak_mb, 2)
        else:
            out["vae"]["decode_s"] = None
            out["vae"]["mp4_ok"] = None

        return out


# ---------------------------------------------------------------------------
# Analytic lower bound (§2.4a)
# ---------------------------------------------------------------------------

def estimate_analytic_lb(
    transformer_cfg: Any, height: int, width: int, num_frames: int,
    persistent_weights_mb: float, max_block_weight_mb: float, bytes_per_elem: float,
) -> Dict[str, Any]:
    """Conservative analytic lower bound on live tensors (ESTIMATE, not physical min).

    LB_analitico = pesos_persistentes + max_i(pesos_bloque_i)
                 + max(conjunto_activaciones_simultaneas)
    """
    patch_size = getattr(transformer_cfg, "patch_size", (1, 2, 2))
    dim = (getattr(transformer_cfg, "dim", None)
           or getattr(transformer_cfg, "hidden_size", None) or 1536)
    p_t, p_h, p_w = patch_size
    latent_f = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    pn_f = max(1, latent_f // p_t)
    pn_h = max(1, latent_h // p_h)
    pn_w = max(1, latent_w // p_w)
    tokens = pn_f * pn_h * pn_w

    hidden_b = tokens * dim * bytes_per_elem
    # Tensors that MUST coexist at the narrowest point of a block forward:
    #   hidden_states + residual copy + qkv projection + MLP hidden (4x)
    activations_lb_mb = (2.0 * hidden_b + 3.0 * hidden_b + 4.0 * hidden_b) / MB
    lb = persistent_weights_mb + max_block_weight_mb + activations_lb_mb
    return {
        "method": "analytic estimate (conservative lower bound, NOT a physical minimum)",
        "inputs": {
            "patch_size": list(patch_size), "dim": dim,
            "latent_f": latent_f, "latent_h": latent_h, "latent_w": latent_w,
            "pn_f": pn_f, "pn_h": pn_h, "pn_w": pn_w, "tokens": tokens,
            "bytes_per_elem": bytes_per_elem,
        },
        "pesos_persistentes_mb": round(persistent_weights_mb, 2),
        "max_pesos_bloque_mb": round(max_block_weight_mb, 2),
        "max_activaciones_simultaneas_mb": round(activations_lb_mb, 2),
        "LB_analitico_mb": round(lb, 2),
        "note": ("Attention score matrices are excluded (chunked/paged in practice); "
                 "this deliberately under-estimates, preserving the lower-bound property."),
    }


# ---------------------------------------------------------------------------
# Attribution decomposition (§2.2)
# ---------------------------------------------------------------------------

def _interp(snaps: List[Dict[str, Any]], t: float, key: str) -> Optional[float]:
    pts = [(s["t_perf"], s.get(key)) for s in snaps if s.get(key) is not None]
    if not pts:
        return None
    pts.sort()
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        t0, v0 = pts[i - 1]
        t1, v1 = pts[i]
        if t0 <= t <= t1:
            if t1 == t0:
                return v1
            frac = (t - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return pts[-1][1]


def compute_attribution(
    snapshots: List[Dict[str, Any]], nvml_series: List[Dict[str, Any]],
    peak_t_perf: Optional[float], s0_mb: float, s1_mb: float,
    persistent_weights_mb: float, peak_streamer_resident_mb: float,
) -> Dict[str, Any]:
    """7-category mutually exclusive partition of Delta_induced (§2.2).

    Categories 1,2,4 partition the torch reserved pool at the peak instant.
    5 is a subset of 4 (reported, not additive). 3 is the NVML transient excess
    bounded by CUDA-event windows. 7 is the residual (never split).
    """
    if peak_t_perf is None:
        return {"error": "no NVML peak timestamp"}

    allocated_peak = _interp(snapshots, peak_t_perf, "allocated_mb") or 0.0
    reserved_peak = _interp(snapshots, peak_t_perf, "reserved_mb") or 0.0
    inactive_split = None
    if snapshots:
        nearest = min(snapshots, key=lambda s: abs(s["t_perf"] - peak_t_perf))
        inactive_split = nearest.get("inactive_split_current_mb")

    peak_nvml = max((s["nvml_used_mb"] for s in nvml_series), default=0.0)
    delta_induced = max(0.0, peak_nvml - s0_mb)
    overhead = max(0.0, s1_mb - s0_mb)

    weights_resident = persistent_weights_mb + peak_streamer_resident_mb
    activations = max(0.0, allocated_peak - weights_resident)
    allocator_pool = max(0.0, reserved_peak - allocated_peak)
    fragmentation = max(0.0, inactive_split or 0.0)

    # Category 3: transient NVML excess within event-bounded windows.
    non_torch_increment = max(0.0, delta_induced - overhead - reserved_peak)
    transient_excesses: List[float] = []
    for s in snapshots:
        t = s["t_perf"]
        near = [x["nvml_used_mb"] for x in nvml_series
                if abs(x["t_perf"] - t) <= 0.050]
        if not near:
            continue
        local_reserved = s.get("reserved_mb", reserved_peak)
        excess = max(0.0, max(near) - s0_mb - overhead - local_reserved)
        transient_excesses.append(excess)
    workspace = min(non_torch_increment, max(transient_excesses) if transient_excesses else 0.0)

    cat1 = weights_resident
    cat2 = activations
    cat3 = workspace
    cat4 = allocator_pool
    cat6 = overhead
    attributed = cat1 + cat2 + cat3 + cat4 + cat6
    others = max(0.0, delta_induced - attributed)

    frac_induced = (attributed / delta_induced) if delta_induced > 0 else 0.0
    frac_raw = (attributed / peak_nvml) if peak_nvml > 0 else 0.0

    # Dominant component + traceable F9-1 recommendation (protocol §5.5/§5.6, §6.3).
    labeled = {
        "1_pesos_residentes": cat1,
        "2_activaciones": cat2,
        "3_workspace_kernels": cat3,
        "4_memoria_allocator": cat4,
        "6_cuda_driver_runtime_overhead": cat6,
        "7_otros_transitorios": others,
    }
    dominant_key = max(labeled, key=labeled.get) if labeled else None
    recommendation = {
        "1_pesos_residentes": "Lifetime/streaming de pesos (residencia de bloques, doble buffer, scheduling de prefetch).",
        "2_activaciones": "Lifetime/coexistencia de activaciones (recompute, chunking/tiling de atención, liberación temprana).",
        "3_workspace_kernels": "Reutilización de workspace de kernels (pool de workspaces, cuDNN/cuBLAS).",
        "4_memoria_allocator": "Gestión del pool del allocator (scheduling de empty_cache, trimming de reserved).",
        "6_cuda_driver_runtime_overhead": "Overhead CUDA/driver/runtime (contexto/streams) — margen de acción limitado.",
        "7_otros_transitorios": "Crecimiento no-torch (driver/WDDM/otros): investigar residencia persistente antes de optimizar.",
    }.get(dominant_key)

    return {
        "Peak_NVML_raw_mb": round(peak_nvml, 2),
        "S0_idle_mb": round(s0_mb, 2),
        "Delta_induced_mb": round(delta_induced, 2),
        "Overhead_CUDA_driver_runtime_mb": round(overhead, 2),
        "at_peak_allocated_mb": round(allocated_peak, 2),
        "at_peak_reserved_mb": round(reserved_peak, 2),
        "non_torch_device_increment_mb": round(non_torch_increment, 2),
        "dominant_category": dominant_key,
        "recommended_mechanism_f9_1": recommendation,
        "categories_mb": {
            "1_pesos_residentes": round(cat1, 2),
            "2_activaciones": round(cat2, 2),
            "3_workspace_kernels": round(cat3, 2),
            "4_memoria_allocator": round(cat4, 2),
            "5_fragmentacion_subset_de_4": round(fragmentation, 2),
            "6_cuda_driver_runtime_overhead": round(cat6, 2),
            "7_otros_transitorios": round(others, 2),
        },
        "categories_breakdown_mb": {
            "1_persistent_weights": round(persistent_weights_mb, 2),
            "1_streaming_slot_resident": round(peak_streamer_resident_mb, 2),
        },
        "attributed_sum_1_to_6_mb": round(attributed, 2),
        "fraction_of_Delta_induced": round(frac_induced, 4),
        "fraction_of_Peak_NVML_raw": round(frac_raw, 4),
        "closure_90pct_met": bool(frac_induced >= ATTRIBUTION_CLOSURE_FRACTION),
        "note": ("5 is a subset of 4 and is NOT added; 7 is the residual. "
                 "Sum(1..6) may be < Delta_induced."),
    }


# ---------------------------------------------------------------------------
# Single protocol run (worker)
# ---------------------------------------------------------------------------

def run_single(args: argparse.Namespace, run_index: int) -> Dict[str, Any]:
    print("=" * 80)
    print(f"  F9-0 RUN {run_index}/{args.runs} -- MEMORY PEAK ATTRIBUTION DIAGNOSTIC")
    print("  (Diagnostic, not optimization. F7/F8 remain CLOSED. Gate unchanged.)")
    print("=" * 80)
    print(f"  Workload: {args.width}x{args.height} @ {args.frames}f, {args.steps} steps, "
          f"mode adaptive, seed {args.seed}")
    print(f"  Reference: F7-D5 = {F7_D5_PEAK_NVML_MB} MB (deficit +{F7_D5_DEFICIT_MB} MB) | "
          f"Gate <= {HARD_GATE_VRAM_MB:.0f} MB")

    t_run_start = time.perf_counter()
    result: Dict[str, Any] = {
        "run_index": run_index,
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F9-0",
        "protocol_reference": "docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md",
        "attribution_iteration": args.attribution_iteration,
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": "adaptive", "seed": args.seed,
                     "guidance_scale": args.guidance},
        "governance": ("Observational/reversible diagnostic. Does NOT modify runtime, "
                       "does NOT change the 4,800 MB gate, does NOT reclassify F7-D5/F7 Stage B/F8."),
    }

    # ---- S0: idle baseline (60 s @ 1 Hz), pre-context ----
    s0_stats = sample_stability(seconds=args.s0_seconds, interval_ms=1000.0, record_series=True)
    S0 = round(s0_stats.get("mean_mb", 0.0), 2)
    pre_procs = nvml_process_summary()
    s0_stable = bool(s0_stats.get("spread_mb") is not None
                     and s0_stats["spread_mb"] <= S0_STABILITY_SPREAD_MB)
    compute_enum_ok = bool(pre_procs.get("compute_enumeration_ok", False))
    exclusive = bool(compute_enum_ok
                     and pre_procs.get("exclusive_other_contexts") is True
                     and s0_stable)
    result["s0_stability"] = s0_stats
    result["processes_pre"] = pre_procs
    result["exclusivity"] = {
        "exclusive": exclusive,
        "compute_enumeration_ok": compute_enum_ok,
        "no_other_cuda_contexts": pre_procs.get("exclusive_other_contexts"),
        "s0_stable": s0_stable,
        "s0_spread_mb": s0_stats.get("spread_mb"),
        "definition": ("no other CUDA compute context (NVML nvmlDeviceGetComputeRunningProcesses) "
                       "AND compute enumeration available AND S0 spread <= 150 MB; "
                       "graphics/DWM residency does not invalidate"),
    }
    print(f"\n  S0 idle baseline: mean {S0} MB (spread {s0_stats.get('spread_mb')} MB, "
          f"stable={s0_stable}) | other CUDA contexts: "
          f"{'none' if pre_procs.get('exclusive_other_contexts') else 'PRESENT'}")

    nvml = NVMLSampler(device_index=0, interval_ms=args.nvml_interval_ms,
                       safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    nvml.mark_phase("s0_end")

    # ---- CUDA context init + deterministic first kernel ----
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    _dummy = torch.empty((10 * 1024 * 1024 // 4,), dtype=torch.float32, device=args.device)
    torch.cuda.synchronize(args.device)
    del _dummy
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    nvml.mark_phase("cuda_warmup_end")
    print("  CUDA context initialized + deterministic warm-up freed.")

    # ---- Control A (pre-model-load) ----
    control_a = None
    if not args.skip_control:
        control_a = run_control_workload(mb=args.control_mb, device=args.device)
        gc.collect()
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)
        print(f"  [Control A] delta_alloc={control_a.get('delta_alloc_mb')} MB "
              f"(proportional={control_a.get('responds_proportionally')})")
    nvml.mark_phase("control_a_end")

    # ---- S1: operational baseline (pre-model-load) ----
    S1 = round(nvml_device_mb(), 2)
    meminfo_s1 = cuda_mem_getinfo_mb()
    overhead_mb = round(S1 - S0, 2)
    result["s1_operational_pre_model_load"] = {
        "S1_mb": S1, "S0_mb": S0, "overhead_cuda_driver_runtime_mb": overhead_mb,
        "cudaMemGetInfo": meminfo_s1,
    }
    print(f"  S1 operational (pre-model-load): {S1} MB | "
          f"Overhead_CUDA/Driver/Runtime = {overhead_mb} MB")

    pipeline = None
    probe = None
    control_b = None
    s3_stats: Dict[str, Any] = {}
    error_occurred: Optional[str] = None
    persistent_weights_mb = 0.0
    max_block_weight_mb = 0.0
    dit_peak_mb = 0.0
    vae_peak_mb = None
    peak_t_perf: Optional[float] = None
    dit_nvml_series: List[Dict[str, Any]] = []
    analytic_lb: Dict[str, Any] = {}

    try:
        pipeline = F9AttributionPipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
        gc.collect()
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)

        # Persistent (non-block) weights resident after pipeline load.
        persistent_weights_mb = round(torch.cuda.memory_allocated(args.device) / MB, 2)
        max_block_weight_mb = round(
            max((module_param_bytes(b, torch.float16) for b in pipeline.blocks), default=0) / MB, 2)
        analytic_lb = estimate_analytic_lb(
            pipeline.transformer.config, args.height, args.width, args.frames,
            persistent_weights_mb, max_block_weight_mb,
            torch.tensor([], dtype=torch.float16).element_size())

        # DiT loop peak is isolated from the VAE peak.
        nvml.reset_peak()
        nvml.mark_phase("workload_start")
        probe = pipeline.run_attribution(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode="adaptive", nvml=nvml,
            output_video_path=(args.output_video if args.export_video else None),
            export_video=args.export_video and not args.no_vae,
        )
        dit_peak_mb = probe.get("dit_peak_nvml_mb", 0.0)
        peak_t_perf = probe.get("dit_peak_t_perf")
        dit_nvml_series = probe.get("dit_nvml_series", [])
        vae_peak_mb = (probe.get("vae") or {}).get("vae_peak_nvml_mb")
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F9-0 ABORT] Exception during execution: {error_occurred}")
    finally:
        pass

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT] NVML exceeded {SAFE_ABORT_NVML_MB:.0f} MB.")

    # ---- S3: post-workload (60 s @ 1 Hz) + Control B ----
    if error_occurred is None:
        try:
            gc.collect()
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(args.device)
            nvml.mark_phase("s3_start")
            s3_stats = sample_stability(seconds=args.s3_seconds, interval_ms=1000.0, record_series=True)
            nvml.mark_phase("s3_end")
            print(f"  S3 post-workload: mean {s3_stats.get('mean_mb')} MB "
                  f"(spread {s3_stats.get('spread_mb')} MB)")
        except Exception as exc:
            print(f"[WARN] S3 sampling failed: {exc}")
        if not args.skip_control:
            control_b = run_control_workload(mb=args.control_mb, device=args.device)
            gc.collect()
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(args.device)
            nvml.mark_phase("control_b_end")
            print(f"  [Control B] delta_alloc={control_b.get('delta_alloc_mb')} MB")

    nvml.stop()
    achieved_hz = nvml.achieved_hz()
    resolution_ok = bool(achieved_hz is not None and achieved_hz >= NVML_MIN_ACCEPTABLE_HZ)
    peak_window = nvml.peak_window(t_perf=peak_t_perf)

    peak_alloc = torch.cuda.max_memory_allocated() / MB
    peak_reserved = torch.cuda.max_memory_reserved() / MB
    peak_ram = get_process_ram_mb()
    post_procs = nvml_process_summary()

    attribution = compute_attribution(
        snapshots=(probe or {}).get("snapshots", []),
        nvml_series=dit_nvml_series,
        peak_t_perf=peak_t_perf, s0_mb=S0, s1_mb=S1,
        persistent_weights_mb=persistent_weights_mb,
        peak_streamer_resident_mb=(probe or {}).get("peak_streamer_resident_mb", 0.0),
    ) if probe else {"error": "workload not executed"}

    min_allocated_obs = None
    snaps = (probe or {}).get("snapshots", [])
    if snaps:
        min_allocated_obs = round(min(s["allocated_mb"] for s in snaps), 2)
    lb_operativo = None
    if analytic_lb.get("LB_analitico_mb") is not None:
        lb_operativo = round(analytic_lb["LB_analitico_mb"] + overhead_mb, 2)

    # ---- §2.5 observer-effect cross-check (option a: document, no extra run) ----
    dit_torch_alloc = (probe or {}).get("dit_torch_peak_alloc_mb")
    dit_torch_reserved = (probe or {}).get("dit_torch_peak_reserved_mb")
    polling = nvml.polling_cost_stats()
    mean_cost = polling.get("mean_ms")
    consistent = bool(
        dit_torch_reserved is not None
        and dit_peak_mb >= dit_torch_reserved
        and mean_cost is not None and mean_cost < 5.0
    )
    observer_cross_check = {
        "purpose": ("§2.5 observer-effect methodological cross-check. Does NOT modify PASS/FAIL "
                    "criteria nor the preregistration."),
        "Peak_NVML_raw_dit_mb": dit_peak_mb,
        "torch_max_memory_allocated_dit_mb": dit_torch_alloc,
        "torch_max_memory_reserved_dit_mb": dit_torch_reserved,
        "nvml_achieved_hz": achieved_hz,
        "nvml_polling_cost": polling,
        "device_wide_ge_process_reserved": bool(
            dit_torch_reserved is not None and dit_peak_mb >= dit_torch_reserved),
        "polling_negligible_vs_interval": bool(
            mean_cost is not None and mean_cost < 5.0),
        "consistent_with_torch_counters": consistent,
        "interpretation": (
            "NVML `nvmlDeviceGetMemoryInfo` is read-only and allocates no GPU memory; the per-sample "
            "polling cost is negligible vs the 20 ms interval. The device-wide peak is expected to be "
            ">= the process `torch` reserved pool (NVML includes the idle/OS/WDDM baseline not present "
            "in torch counters). Agreement between `Peak_NVML_raw`, `max_memory_allocated` and "
            "`max_memory_reserved` indicates the 50 Hz polling is not artificially inflating the peak."),
    }

    # ---- Per-run verdict (§5) ----
    frac = attribution.get("fraction_of_Delta_induced")
    if error_occurred is not None:
        verdict = "INSTRUMENT INVALID"
        reason = f"execution error: {error_occurred}"
    elif not exclusive:
        verdict = "INSTRUMENT INVALID"
        reason = "exclusivity failed (other CUDA contexts or unstable S0)"
    elif not resolution_ok:
        verdict = "INSTRUMENT INVALID"
        reason = f"NVML resolution {achieved_hz} Hz < {NVML_MIN_ACCEPTABLE_HZ} Hz (POSIBLE SUBESTIMACION DEL PICO)"
    elif frac is not None and frac >= ATTRIBUTION_CLOSURE_FRACTION:
        verdict = "CLOSED - ATTRIBUTED"
        reason = f"{frac*100:.1f}% of Delta_induced attributed; peak window localized"
    else:
        verdict = "ATTRIBUTION INCOMPLETE - DOCUMENTED"
        reason = f"{'n/a' if frac is None else f'{frac*100:.1f}%'} of Delta_induced attributed (<90%)"

    result.update({
        "control_A": control_a,
        "control_B": control_b,
        "s3_stability": s3_stats,
        "persistent_weights_mb": persistent_weights_mb,
        "max_block_weight_mb": max_block_weight_mb,
        "LB_analitico": analytic_lb,
        "Min_allocated_obs_mb": min_allocated_obs,
        "LB_operativo_mb": lb_operativo,
        "memory_peaks_mb": {
            "Peak_NVML_raw_dit_mb": dit_peak_mb,
            "VAE_peak_nvml_mb": vae_peak_mb,
            "peak_torch_alloc": round(peak_alloc, 2),
            "peak_torch_reserved": round(peak_reserved, 2),
            "peak_process_ram": round(peak_ram, 2),
        },
        "observer_effect_cross_check": observer_cross_check,
        "nvml_instrument": {
            "requested_hz": nvml.requested_hz,
            "achieved_hz": achieved_hz,
            "resolution_ok": resolution_ok,
            "min_acceptable_hz": NVML_MIN_ACCEPTABLE_HZ,
            "polling_cost": nvml.polling_cost_stats(),
            "n_series_samples": len(nvml.series),
            "peak_window": peak_window,
            "phase_marks": nvml.phase_marks,
            "series": nvml.export_series(downsample=args.series_downsample),
            "series_downsample": args.series_downsample,
        },
        "per_step": {
            "step_times_s": (probe or {}).get("step_times_s"),
            "step_end_reserved_mb": (probe or {}).get("step_end_reserved_mb"),
            "step_end_alloc_mb": (probe or {}).get("step_end_alloc_mb"),
            "seam_count": (probe or {}).get("seam_count"),
            "step_start_count": (probe or {}).get("step_start_count"),
        },
        "snapshots": (probe or {}).get("snapshots", []),
        "cuda_events": (probe or {}).get("cuda_events", []),
        "vae": (probe or {}).get("vae", {}),
        "attribution": attribution,
        "processes_post": post_procs,
        "comparison": {
            "f7_d5_peak_nvml_mb": F7_D5_PEAK_NVML_MB,
            "f7_d5_deficit_mb": F7_D5_DEFICIT_MB,
            "f7_d5_peak_reserved_mb": F7_D5_PEAK_RESERVED_MB,
            "hard_gate_mb": HARD_GATE_VRAM_MB,
        },
        "verdict": {
            "execution_completed": error_occurred is None,
            "exclusive": exclusive,
            "resolution_ok": resolution_ok,
            "closure_90pct_met": attribution.get("closure_90pct_met"),
            "result_class": verdict,
            "reason": reason,
            "error_detail": error_occurred,
        },
    })

    t_run = time.perf_counter() - t_run_start
    print("\n" + "=" * 80)
    print(f"  F9-0 RUN {run_index} SCORECARD")
    print("=" * 80)
    print(f"  S0 idle                       [MEASURED] : {S0} MB (spread {s0_stats.get('spread_mb')})")
    print(f"  S1 operational (pre-load)     [MEASURED] : {S1} MB")
    print(f"  Overhead CUDA/driver/runtime  [DERIVED]  : {overhead_mb} MB")
    print(f"  Peak NVML (DiT loop)          [MEASURED] : {dit_peak_mb} MB (F7-D5 {F7_D5_PEAK_NVML_MB})")
    print(f"  VAE peak (separate)           [MEASURED] : {vae_peak_mb} MB")
    print(f"  Delta_induced (peak - S0)     [DERIVED]  : {attribution.get('Delta_induced_mb')} MB")
    print(f"  Attributed sum (1..6)         [DERIVED]  : {attribution.get('attributed_sum_1_to_6_mb')} MB")
    print(f"  Fraction of Delta_induced     [DERIVED]  : {attribution.get('fraction_of_Delta_induced')}")
    print(f"  Fraction of Peak_NVML_raw     [DERIVED]  : {attribution.get('fraction_of_Peak_NVML_raw')}")
    print(f"  LB_analitico                  [ESTIMATE] : {analytic_lb.get('LB_analitico_mb')} MB")
    print(f"  Min_allocated_obs             [OBSERVED] : {min_allocated_obs} MB")
    print(f"  LB_operativo                  [DERIVED]  : {lb_operativo} MB")
    print(f"  NVML achieved                 [MEASURED] : {achieved_hz} Hz")
    print(f"  Wall-clock                    [OBSERVED] : {t_run:.1f} s")
    print(f"  VERDICT                       [DERIVED]  : {verdict}")
    print(f"  reason                                   : {reason}")
    print("=" * 80 + "\n")

    result["wall_clock_s"] = round(t_run, 2)
    return result


def run_worker_mode(args: argparse.Namespace) -> None:
    result = run_single(args, run_index=args.run_index)
    os.makedirs(os.path.dirname(args.run_out), exist_ok=True)
    with open(args.run_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f">> Run telemetry saved to: {args.run_out}")
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# Campaign coordinator (N independent clean-process runs + cooldown)
# ---------------------------------------------------------------------------

def _cross_run_stats(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n, "mean": round(mean, 2), "median": round(statistics.median(vals), 2),
        "min": round(min(vals), 2), "max": round(max(vals), 2),
        "std": round(std, 2), "ci95_half_width": round(ci95, 2),
        "spread": round(max(vals) - min(vals), 2),
    }


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def pick(path: List[str]) -> List[float]:
        out = []
        for r in runs:
            cur: Any = r
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    fracs = pick(["attribution", "fraction_of_Delta_induced"])
    deltas = pick(["attribution", "Delta_induced_mb"])
    peaks = pick(["memory_peaks_mb", "Peak_NVML_raw_dit_mb"])
    s0s = pick(["s0_stability", "mean_mb"])
    overheads = pick(["s1_operational_pre_model_load", "overhead_cuda_driver_runtime_mb"])
    valid = [r for r in runs if r.get("verdict", {}).get("result_class") == "CLOSED - ATTRIBUTED"]
    any_closure = bool(valid)

    if valid:
        campaign_verdict = "CLOSED - ATTRIBUTED"
    elif runs and all(r.get("verdict", {}).get("result_class") == "INSTRUMENT INVALID" for r in runs):
        campaign_verdict = "INSTRUMENT INVALID"
    else:
        campaign_verdict = "ATTRIBUTION INCOMPLETE - DOCUMENTED"

    return {
        "n_runs": len(runs),
        "n_valid_exclusive": len(valid),
        "campaign_verdict": campaign_verdict,
        "fraction_of_Delta_induced": _cross_run_stats(fracs),
        "Delta_induced_mb": _cross_run_stats(deltas),
        "Peak_NVML_raw_dit_mb": _cross_run_stats(peaks),
        "S0_idle_mb": _cross_run_stats(s0s),
        "overhead_mb": _cross_run_stats(overheads),
        "closure_90pct_any_run": bool(any_closure),
    }


def run_campaign(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  PHASE F9-0 -- PREREGISTERED MEMORY PEAK ATTRIBUTION DIAGNOSTIC")
    print("  Campaign: N independent clean-process runs with cooldown.")
    print(f"  Protocol: docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md")
    print("=" * 80)
    print(f"  Runs: {args.runs} | Steps: {args.steps} | {args.width}x{args.height} @ {args.frames}f")
    print(f"  Cooldown between runs: {args.cooldown_seconds:.0f} s")
    print(f"  Frozen reference F7-D5 = {F7_D5_PEAK_NVML_MB} MB (deficit +{F7_D5_DEFICIT_MB} MB)")
    print("  Governance: NO execution without explicit Director authorization.")

    py_exe = sys.executable
    runs: List[Dict[str, Any]] = []
    execution_order: List[Dict[str, Any]] = []

    for i in range(1, args.runs + 1):
        run_out = os.path.join(args.run_output_dir, f"f9_0_run_{i:02d}.json")
        print(f"\n[Coordinator] >>> LAUNCHING RUN {i}/{args.runs} (clean process) <<<")
        cmd = [
            py_exe, __file__, "--worker-mode",
            "--run-index", str(i),
            "--steps", str(args.steps), "--frames", str(args.frames),
            "--width", str(args.width), "--height", str(args.height),
            "--seed", str(args.seed), "--guidance", str(args.guidance),
            "--device", args.device,
            "--s0-seconds", str(args.s0_seconds), "--s3-seconds", str(args.s3_seconds),
            "--nvml-interval-ms", str(args.nvml_interval_ms),
            "--control-mb", str(args.control_mb),
            "--attribution-iteration", str(args.attribution_iteration),
            "--series-downsample", str(args.series_downsample),
            "--run-out", run_out,
        ]
        if args.skip_control:
            cmd.append("--skip-control")
        if args.no_vae:
            cmd.append("--no-vae")
        if args.export_video:
            cmd.append("--export-video")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        wall = round(time.perf_counter() - t0, 2)
        execution_order.append({"run": i, "returncode": proc.returncode, "wall_s": wall,
                                "timestamp": datetime.datetime.now().isoformat()})
        if proc.returncode != 0 or not os.path.exists(run_out):
            print(f"[Coordinator FATAL] Run {i} failed (code {proc.returncode}). Aborting campaign.")
            break
        with open(run_out, "r", encoding="utf-8") as f:
            runs.append(json.load(f))

        if i < args.runs:
            print(f"\n[Coordinator] >>> COOLDOWN {args.cooldown_seconds:.0f}s (WDDM page recovery) <<<")
            remaining = float(args.cooldown_seconds)
            while remaining > 0:
                print(f"   Cooldown remaining: {remaining:5.0f}s | NVML physical: {nvml_device_mb():.1f} MB")
                step = min(10.0, remaining)
                time.sleep(step)
                remaining -= step

    aggregate = _aggregate(runs)
    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F9-0 (Preregistered Memory Peak Attribution Diagnostic)",
        "protocol_reference": "docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md",
        "charter_reference": "docs/F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md",
        "attribution_iteration": args.attribution_iteration,
        "governance": ("Diagnostic, not optimization. Does NOT modify runtime, does NOT change the "
                       "4,800 MB gate, does NOT reclassify F7-D5 / F7 Stage B / F8."),
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": "adaptive", "seed": args.seed},
        "execution_order": execution_order,
        "aggregate": aggregate,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\n>> Campaign telemetry saved to: {args.output}")

    if args.report:
        write_report(args, telemetry)
        print(f">> Report saved to: {args.report}")

    print("=" * 80)
    print(f"  F9-0 CAMPAIGN VERDICT: {aggregate.get('campaign_verdict')}")
    print("=" * 80 + "\n")


def write_report(args: argparse.Namespace, telemetry: Dict[str, Any]) -> None:
    runs = telemetry.get("runs", [])
    agg = telemetry.get("aggregate", {})
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F9-0 — Preregistered Memory Peak Attribution Diagnostic — Report\n\n")
        w(f"**Date:** {telemetry['timestamp']}  \n")
        w(f"**Protocol:** [`docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md`]"
          f"(F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md)  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, {args.steps} DiT steps, "
          f"mode `adaptive`, seed {args.seed}  \n")
        w(f"**Frozen reference:** F7-D5 = {F7_D5_PEAK_NVML_MB} MB (deficit +{F7_D5_DEFICIT_MB} MB)  \n")
        w(f"**Attribution iteration:** {args.attribution_iteration}  \n")
        w(f"**Campaign verdict:** **{agg.get('campaign_verdict')}**  \n\n")
        w("---\n\n## 1. Cross-run summary\n\n")
        w(f"- Independent runs: {agg.get('n_runs')} (exclusive: {agg.get('n_valid_exclusive')})\n")
        for key in ("Peak_NVML_raw_dit_mb", "S0_idle_mb", "Delta_induced_mb",
                    "overhead_mb", "fraction_of_Delta_induced"):
            st = agg.get(key, {})
            if st:
                w(f"- {key}: median {st.get('median')} | mean {st.get('mean')} "
                  f"| min {st.get('min')} | max {st.get('max')} | std {st.get('std')} "
                  f"| 95% CI ±{st.get('ci95_half_width')}\n")
        w("\n## 2. Per-run attribution (7 mutually exclusive categories)\n\n")
        w("| Run | Peak NVML | S0 | Delta_induced | Cat1 | Cat2 | Cat3 | Cat4 | Cat6 | Cat7 | Attributed | Fraction | Verdict |\n")
        w("| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-- |\n")
        for r in runs:
            a = r.get("attribution", {})
            c = a.get("categories_mb", {})
            mp = r.get("memory_peaks_mb", {})
            v = r.get("verdict", {})
            w(f"| {r.get('run_index')} | {mp.get('Peak_NVML_raw_dit_mb')} | {a.get('S0_idle_mb')} "
              f"| {a.get('Delta_induced_mb')} | {c.get('1_pesos_residentes')} | {c.get('2_activaciones')} "
              f"| {c.get('3_workspace_kernels')} | {c.get('4_memoria_allocator')} "
              f"| {c.get('6_cuda_driver_runtime_overhead')} | {c.get('7_otros_transitorios')} "
              f"| {a.get('attributed_sum_1_to_6_mb')} | {a.get('fraction_of_Delta_induced')} "
              f"| {v.get('result_class')} |\n")
        w("\n> Category 5 (fragmentation) is a subset of category 4 and is NOT additive. "
          "Category 7 is the residual and is never split.\n")
        w("\n## 3. Lower bounds (per run)\n\n")
        w("| Run | LB_analitico (MB) | Min_allocated_obs (MB) | LB_operativo (MB) | Overhead (MB) |\n")
        w("| :--: | :--: | :--: | :--: | :--: |\n")
        for r in runs:
            lb = r.get("LB_analitico", {})
            overhead = r.get("s1_operational_pre_model_load", {}).get("overhead_cuda_driver_runtime_mb")
            w(f"| {r.get('run_index')} | {lb.get('LB_analitico_mb')} | {r.get('Min_allocated_obs_mb')} "
              f"| {r.get('LB_operativo_mb')} | {overhead} |\n")
        w("\n- `LB_analitico` = pesos_persistentes + max_i(pesos_bloque_i) + max(activaciones_simultaneas) (ESTIMATE)\n")
        w("- `Min_allocated_obs` = minimum observed `torch.cuda.memory_allocated` at synced points\n")
        w("- `LB_operativo` = `LB_analitico` + `Overhead_CUDA/Driver/Runtime`\n")
        w("\n## 4. Observer-effect cross-check (§2.5, option a)\n\n")
        w("| Run | Peak NVML (DiT) | torch max_allocated | torch max_reserved | NVML Hz | Polling mean (ms) | Consistent |\n")
        w("| :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n")
        for r in runs:
            oc = r.get("observer_effect_cross_check", {})
            pc = oc.get("nvml_polling_cost", {})
            w(f"| {r.get('run_index')} | {oc.get('Peak_NVML_raw_dit_mb')} "
              f"| {oc.get('torch_max_memory_allocated_dit_mb')} "
              f"| {oc.get('torch_max_memory_reserved_dit_mb')} "
              f"| {oc.get('nvml_achieved_hz')} | {pc.get('mean_ms')} "
              f"| {oc.get('consistent_with_torch_counters')} |\n")
        w("\n> NVML polling is read-only and allocates no GPU memory; per-sample cost is negligible "
          "vs the 20 ms interval. This cross-check is methodological evidence only and does not "
          "modify PASS/FAIL criteria or the preregistration.\n")
        w("\n## 5. Dominant component and F9-1 recommendation\n\n")
        for r in runs:
            a = r.get("attribution", {})
            w(f"- Run {r.get('run_index')}: dominant = `{a.get('dominant_category')}` → "
              f"{a.get('recommended_mechanism_f9_1')}\n")
        w("\n## 6. Governance\n\n")
        w("F9-0 is observational and reversible: it does not modify the runtime, does not change "
          "the 4,800 MB gate, and does not reclassify F7-D5 / F7 Stage B / F8. F7 and F8 remain CLOSED. "
          "The next mechanism (F9-1) is selected solely from this diagnostic.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Marley Phase F9-0 -- Preregistered Memory Peak Attribution Diagnostic")
    p.add_argument("--worker-mode", action="store_true", help="Internal: run a single protocol run")
    p.add_argument("--run-index", type=int, default=1, help="Internal: run ordinal")
    p.add_argument("--run-out", type=str, default="logs/f9_0_run_01.json",
                   help="Internal: per-run output JSON path")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of independent runs (N>=5)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--s0-seconds", type=float, default=DEFAULT_S0_SECONDS)
    p.add_argument("--s3-seconds", type=float, default=DEFAULT_S3_SECONDS)
    p.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    p.add_argument("--nvml-interval-ms", type=float, default=NVML_INTERVAL_MS)
    p.add_argument("--control-mb", type=float, default=DEFAULT_CONTROL_MB)
    p.add_argument("--skip-control", action="store_true",
                   help="Skip A/B controls (NOT recommended; disables instrument validation)")
    p.add_argument("--attribution-iteration", type=int, default=1, choices=[1, 2],
                   help="Instrumentation iteration (max 2 per protocol)")
    p.add_argument("--series-downsample", type=int, default=1,
                   help="Downsample factor for the stored NVML series (1 = full)")
    p.add_argument("--no-vae", action="store_true", help="Skip the separate VAE decode phase")
    p.add_argument("--export-video", action="store_true", help="Export the VAE-decoded mp4")
    p.add_argument("--output-video", type=str, default="logs/f9_0_attribution.mp4")
    p.add_argument("--run-output-dir", type=str, default="logs")
    p.add_argument("--output", type=str, default="logs/f9_0_attribution_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F9_0_ATTRIBUTION_REPORT_01.md",
                   help="Markdown report path (empty string disables)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("=" * 80)
        print("  F9-0 -- DRY-RUN (parameter verification; NO inference, NO GPU workload)")
        print("=" * 80)
        print(f"  Protocol: docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md")
        print(f"  Runs: {args.runs} | Cooldown: {args.cooldown_seconds:.0f}s | "
              f"Workload: {args.width}x{args.height} @ {args.frames}f / {args.steps} steps")
        print(f"  S0: {args.s0_seconds:.0f}s @ 1 Hz | S3: {args.s3_seconds:.0f}s @ 1 Hz | "
              f"NVML: {args.nvml_interval_ms:.0f} ms ({1000.0/args.nvml_interval_ms:.0f} Hz)")
        print(f"  Controls A/B: ~{args.control_mb:.0f} MB each "
              f"({'SKIPPED (not recommended)' if args.skip_control else 'enabled'})")
        print(f"  Instrumentation iteration: {args.attribution_iteration}/2")
        print(f"  Attribution: 7 mutually exclusive categories; closure >= 90% of Delta_induced")
        print(f"  Frozen reference: F7-D5 = {F7_D5_PEAK_NVML_MB} MB (deficit +{F7_D5_DEFICIT_MB} MB)")
        print(f"  Outputs: {args.output} | report: {args.report or '(disabled)'}")
        print(f"  Estimated campaign wall-clock: ~{args.runs * 45} min + cooldowns "
              f"(30-step 720p ~44 min/run)")
        print("[DRY-RUN] Parameters valid. Awaiting explicit Director authorization.")
        return

    if args.worker_mode:
        run_worker_mode(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
