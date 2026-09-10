"""
f7_stage_b_full_validation.py
=============================
Phase F7 -- Stage B: Re-evaluation 720p (30 steps, 33 frames, 1280x720)
Preregistered Protocol Reference: docs/F7_720P_PREREGISTERED_PROTOCOL_01.md §6

Authorized Execution Protocol:
  1. S0 idle baseline (60 s @ 1 Hz)
  2. torch.cuda.init() + warm-up 10 MB + empty_cache()
  3. Control A (~500 MB known workload) + empty_cache()
  4. Load Wan2.1 Pipeline (CPU offload, DiT fp16, VAE bf16 tiled, prompt embeds)
  5. S1 operational baseline (post-init)
  6. Wan2.1 Denoise (30 adaptive steps with step-start drain & cond->uncond seam release)
  7. VAE Decode (33 frames @ 720p spatially tiled 256x256)
  8. Release + empty_cache()
  9. S3 post-workload observation (120 s @ 1 Hz)
 10. Control B (~500 MB known workload) + empty_cache()
 11. S4 final observation (60 s @ 1 Hz)

Gates & Metrics:
  - Gate A (Physical Safety): Peak NVML <= 4,800.0 MB
  - Gate B (Incremental Efficiency): Delta S0 (S2 - S0) <= 3,850.0 MB
  - Safe Abort: NVML > 5,800.0 MB (per Protocol §7)
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
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline
from marley.core.adaptive import AdaptiveEngine
from diffusers.utils import export_to_video

# Preregistered protocol constants
HARD_GATE_PHYSICAL_VRAM_MB = 4800.0
HARD_GATE_INCREMENTAL_MB = 3850.0
SAFE_ABORT_NVML_MB = 5800.0
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


def get_process_host_memory_mb() -> Dict[str, float]:
    p = psutil.Process(os.getpid())
    mem = p.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024 * 1024), 2),
        "vms_mb": round(mem.vms / (1024 * 1024), 2),
    }


class NVMLSampler:
    """High-frequency physical VRAM sampler (20 ms) with safe abort guard."""

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
        out = {"nvml_used_mb": 0.0, "gpu_temp_c": None, "gpu_clock_mhz": None, "gpu_util_pct": None}
        if not self._available or self._handle is None:
            return out
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            out["nvml_used_mb"] = round(info.used / (1024 * 1024), 2)
            try:
                out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            try:
                out["gpu_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
            except Exception:
                pass
            try:
                out["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            except Exception:
                pass
        except Exception:
            pass
        return out

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb


def nvml_device_mb() -> float:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)


def cuda_mem_getinfo_mb() -> Optional[Dict[str, float]]:
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "free_mb": round(free_b / (1024 * 1024), 2),
            "total_mb": round(total_b / (1024 * 1024), 2),
            "used_mb": round((total_b - free_b) / (1024 * 1024), 2),
        }
    except Exception:
        return None


def sample_stability(seconds: float, interval_ms: float = 1000.0, record_series: bool = True) -> Dict[str, Any]:
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
    out: Dict[str, Any] = {"requested_mb": mb}
    nvml_before = nvml_device_mb()
    meminfo_before = cuda_mem_getinfo_mb()
    num_elements = int((mb * 1024 * 1024) / 4)
    t = torch.empty(num_elements, dtype=torch.float32, device=device)
    t.fill_(1.0)
    torch.cuda.synchronize(device)
    nvml_peak = nvml_device_mb()
    meminfo_peak = cuda_mem_getinfo_mb()
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    del t
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    nvml_after = nvml_device_mb()
    meminfo_after = cuda_mem_getinfo_mb()

    out["nvml_before_mb"] = round(nvml_before, 2)
    out["nvml_peak_mb"] = round(nvml_peak, 2)
    out["nvml_after_mb"] = round(nvml_after, 2)
    out["nvml_delta_mb"] = round(nvml_peak - nvml_before, 2)
    out["torch_allocated_mb"] = round(alloc_mb, 2)
    out["meminfo_before"] = meminfo_before
    out["meminfo_peak"] = meminfo_peak
    out["meminfo_after"] = meminfo_after
    return out


class StageBPipeline(MarleyEndToEndPipeline):
    """Full 30-step pipeline with surgical cond->uncond seam release and step-start drain."""

    def run_stage_b_denoise(
        self, prompt: str, negative_prompt: str, height: int, width: int,
        num_frames: int, num_steps: int, guidance_scale: float, seed: int,
        mode: str, nvml: NVMLSampler, output_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_global = time.perf_counter()
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {}

        # 1. Text embeddings
        t_enc_start = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        out["prompt_enc_s"] = round(time.perf_counter() - t_enc_start, 2)

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

        streamer = AdaptiveEngine(
            blocks=self.blocks, device=device, dtype=dtype, window=1,
            pressure_sampler=pressure_sampler,
        )

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
        _meminfo_peak = None

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
                step_start_count += 1

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

                # Surgical cond->uncond seam release
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                seam_count += 1

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

                step_s = time.perf_counter() - ts
                step_times.append(step_s)
                snap = _snap()
                step_end_reserved.append(snap["reserved_mb"])
                step_end_alloc.append(snap["alloc_mb"])
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

        denoise_s = time.perf_counter() - denoise_start
        out["seam_count"] = seam_count
        out["step_start_count"] = step_start_count
        out["step_times_s"] = [round(x, 2) for x in step_times]
        out["step_end_reserved_mb"] = step_end_reserved
        out["step_end_alloc_mb"] = step_end_alloc
        out["meminfo_workload_peak"] = _meminfo_peak
        out["denoise_time_s"] = round(denoise_s, 2)
        out["cadence_avg_s"] = round(sum(step_times) / len(step_times), 2) if step_times else None

        # VAE Decode (Tiled bfloat16)
        out["vae"] = {}
        if output_video_path is not None:
            print("[Stage B] Starting VAE decode (33 frames, 720p, tiled bf16)...")
            vae_start = time.perf_counter()
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
            out["vae"]["decode_s"] = round(time.perf_counter() - vae_start, 2)
            out["vae"]["video_shape"] = list(video.shape)
            out["vae"]["nan_inf"] = bool(torch.isnan(video).any() or torch.isinf(video).any())
            try:
                frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                export_to_video(frames[0], output_video_path, fps=CANONICAL_FPS)
                out["vae"]["mp4_ok"] = os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0
                out["vae"]["output_video"] = output_video_path
                print(f"[Stage B] Video exported successfully to: {output_video_path}")
            except Exception as exc:
                out["vae"]["mp4_ok"] = False
                out["vae"]["export_error"] = str(exc)
        else:
            out["vae"]["decode_s"] = None
            out["vae"]["mp4_ok"] = False

        out["total_wall_s"] = round(time.perf_counter() - t_global, 2)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7 Stage B -- Full 30-step 720p Re-evaluation")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--s0-seconds", type=float, default=60.0, help="S0 baseline observation duration (s)")
    p.add_argument("--s3-seconds", type=float, default=120.0, help="S3 post-workload observation duration (s)")
    p.add_argument("--s4-seconds", type=float, default=60.0, help="S4 final observation duration (s)")
    p.add_argument("--control-mb", type=float, default=500.0, help="Size of known CUDA control (MB)")
    p.add_argument("--no-vae", action="store_true", help="Skip VAE decode/export")
    p.add_argument("--output", type=str, default="logs/f7_stage_b_full_validation_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_STAGE_B_FULL_VALIDATION_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true", help="Perform sanity check without model execution")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7: STAGE B FULL RE-EVALUATION (30 STEPS @ 720p)")
    print("  Protocol Reference: docs/F7_720P_PREREGISTERED_PROTOCOL_01.md §6")
    print("=" * 80)
    print(f"  Target: {args.width}x{args.height} | {args.frames} frames | {args.steps} steps | Seed: {args.seed}")
    print(f"  Gate A (Physical VRAM): <= {HARD_GATE_PHYSICAL_VRAM_MB:.1f} MB | Gate B (Net Delta): <= {HARD_GATE_INCREMENTAL_MB:.1f} MB")
    print(f"  Safe Abort Threshold:  > {SAFE_ABORT_NVML_MB:.1f} MB")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY-RUN] Validating environment and arguments:")
        print(f"  Python: {sys.executable}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  NVML instant: {nvml_device_mb():.1f} MB")
        print("[DRY-RUN] Configuration verified. Exiting.")
        return

    out_mp4 = None if args.no_vae else f"logs/f7_stage_b_{args.width}x{args.height}_{args.frames}f_{args.steps}st_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7 Stage B (30-step 720p Full Re-evaluation)",
        "protocol_reference": "docs/F7_720P_PREREGISTERED_PROTOCOL_01.md §6",
        "parameters": vars(args),
        "host_system": {
            "platform": sys.platform,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
    }

    # 1. S0 idle baseline (60 s @ 1 Hz)
    print(f"\n[Phase 1] Sampling S0 idle baseline ({args.s0_seconds:.0f} s @ 1 Hz)...")
    s0_stats = sample_stability(args.s0_seconds, interval_ms=1000.0, record_series=True)
    s0_mean = s0_stats["mean_mb"]
    s0_spread = s0_stats["spread_mb"]
    print(f"  S0 Baseline: mean {s0_mean:.1f} MB (min {s0_stats['min_mb']:.1f} / max {s0_stats['max_mb']:.1f} / spread {s0_spread:.1f} MB)")
    telemetry["s0_stats"] = s0_stats

    # 2. CUDA init + warm-up (10 MB)
    print("\n[Phase 2] Initializing CUDA context + 10 MB warm-up...")
    torch.cuda.init()
    _warmup = torch.empty(int(10 * 1024 * 1024 / 4), dtype=torch.float32, device=args.device)
    _warmup.fill_(0.0)
    del _warmup
    gc.collect()
    torch.cuda.synchronize(args.device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)

    # 3. Control A (~500 MB)
    print(f"\n[Phase 3] Running Control A ({args.control_mb:.0f} MB)...")
    control_a = run_control_workload(args.control_mb, device=args.device)
    print(f"  Control A delta: {control_a['nvml_delta_mb']:.1f} MB (allocated: {control_a['torch_allocated_mb']:.1f} MB)")
    telemetry["control_a"] = control_a

    # 4. Start Continuous NVML Sampler (20 ms)
    sampler = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    sampler.start()
    torch.cuda.reset_peak_memory_stats(args.device)

    # 5. Load Pipeline & S1 Operational Baseline
    print("\n[Phase 4] Loading Wan2.1 Pipeline components...")
    pipe = StageBPipeline(
        device=args.device, dit_dtype=torch.float16,
        vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16,
    )
    s1_mb = nvml_device_mb()
    s1_meminfo = cuda_mem_getinfo_mb()
    print(f"  S1 Operational: {s1_mb:.1f} MB | cudaMemGetInfo: {s1_meminfo}")
    telemetry["s1_mb"] = s1_mb
    telemetry["s1_meminfo"] = s1_meminfo

    # 6. Execute Full 30-step Denoise + VAE Decode
    print(f"\n[Phase 5] Executing 30-step Denoise @ {args.width}x{args.height} / {args.frames}f...")
    workload_res = None
    exec_error = None
    try:
        workload_res = pipe.run_stage_b_denoise(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode=args.mode, nvml=sampler, output_video_path=out_mp4,
        )
    except Exception as exc:
        exec_error = str(exc)
        print(f"\n[EXECUTION ERROR] Exception during Stage B: {exec_error}")
    finally:
        s2_peak_nvml = sampler.stop()

    telemetry["workload_execution"] = workload_res
    telemetry["execution_error"] = exec_error
    telemetry["s2_peak_nvml_mb"] = s2_peak_nvml
    print(f"\n[Workload Finished] Peak Physical VRAM (NVML): {s2_peak_nvml:.1f} MB")

    # 7. Release & empty_cache()
    gc.collect()
    torch.cuda.synchronize(args.device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)

    # 8. S3 Post-workload observation (120 s @ 1 Hz)
    print(f"\n[Phase 6] Sampling S3 post-workload plateau ({args.s3_seconds:.0f} s @ 1 Hz)...")
    s3_stats = sample_stability(args.s3_seconds, interval_ms=1000.0, record_series=True)
    s3_mean = s3_stats["mean_mb"]
    s3_spread = s3_stats["spread_mb"]
    print(f"  S3 Plateau: mean {s3_mean:.1f} MB (min {s3_stats['min_mb']:.1f} / max {s3_stats['max_mb']:.1f} / spread {s3_spread:.1f} MB)")
    telemetry["s3_stats"] = s3_stats

    # 9. Control B (~500 MB)
    print(f"\n[Phase 7] Running Control B ({args.control_mb:.0f} MB)...")
    control_b = run_control_workload(args.control_mb, device=args.device)
    print(f"  Control B delta: {control_b['nvml_delta_mb']:.1f} MB (allocated: {control_b['torch_allocated_mb']:.1f} MB)")
    telemetry["control_b"] = control_b

    # 10. S4 Final observation (60 s @ 1 Hz)
    print(f"\n[Phase 8] Sampling S4 final stability ({args.s4_seconds:.0f} s @ 1 Hz)...")
    s4_stats = sample_stability(args.s4_seconds, interval_ms=1000.0, record_series=True)
    s4_mean = s4_stats["mean_mb"]
    s4_spread = s4_stats["spread_mb"]
    print(f"  S4 Baseline: mean {s4_mean:.1f} MB (min {s4_stats['min_mb']:.1f} / max {s4_stats['max_mb']:.1f} / spread {s4_spread:.1f} MB)")
    telemetry["s4_stats"] = s4_stats

    # Compute Final Evaluation Metrics
    delta_s0 = round(s2_peak_nvml - s0_mean, 2)
    delta_s1 = round(s2_peak_nvml - s1_mb, 2)
    gate_a_pass = bool(s2_peak_nvml <= HARD_GATE_PHYSICAL_VRAM_MB and exec_error is None)
    gate_b_pass = bool(delta_s0 <= HARD_GATE_INCREMENTAL_MB and exec_error is None)

    if gate_a_pass and gate_b_pass:
        classification = "🟢 720p CERTIFIED FULL PASS"
        verdict_summary = "Escenario 1: Both Gate A and Gate B satisfied."
    elif gate_a_pass and not gate_b_pass:
        classification = "🟡 FUNCTIONAL / INEFFICIENT"
        verdict_summary = "Escenario 2: Gate A passed (physically safe), but Gate B failed (incremental overhead)."
    elif not gate_a_pass and gate_b_pass:
        classification = "🔴 GATE A FAIL / 🟢 B FAVORABLE"
        verdict_summary = "Escenario 3: Algorithmic delta favorable, but physical device peak exceeded 4,800 MB."
    else:
        classification = "🔴 COMPLETE REJECTION"
        verdict_summary = "Escenario 4: Both Gate A and Gate B failed."

    telemetry["scorecard"] = {
        "gate_a_physical_limit_mb": HARD_GATE_PHYSICAL_VRAM_MB,
        "gate_a_observed_peak_mb": s2_peak_nvml,
        "gate_a_pass": gate_a_pass,
        "gate_b_incremental_limit_mb": HARD_GATE_INCREMENTAL_MB,
        "gate_b_observed_delta_s0_mb": delta_s0,
        "gate_b_pass": gate_b_pass,
        "delta_s1_mb": delta_s1,
        "classification": classification,
        "verdict_summary": verdict_summary,
    }

    # Save telemetry JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\n>> Telemetry saved to: {args.output}")

    # Generate Markdown Report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7 Stage B — 720p Full Re-evaluation Report (30 Steps)\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Protocol Reference:** [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md) §6  \n")
        w(f"**Classification:** **{classification}**  \n\n")
        w("---\n\n")
        w("## 1. Executive Scorecard\n\n")
        w(f"> **Official Matrix Verdict:** {verdict_summary}\n\n")
        w("| Gate / Metric | Pre-registered Limit | Observed Value | Result |\n")
        w("| :--- | :---: | :---: | :---: |\n")
        w(f"| **Gate A (Physical VRAM)** | $\\le {HARD_GATE_PHYSICAL_VRAM_MB:.1f}$ MB | **{s2_peak_nvml:.1f} MB** | {'🟢 PASS' if gate_a_pass else '🔴 FAIL'} |\n")
        w(f"| **Gate B (Net Delta $\\Delta_{{S0}}$)** | $\\le {HARD_GATE_INCREMENTAL_MB:.1f}$ MB | **{delta_s0:.1f} MB** | {'🟢 PASS' if gate_b_pass else '🔴 FAIL'} |\n")
        w(f"| **Operational Delta $\\Delta_{{S1}}$** | N/A (Analytical) | **{delta_s1:.1f} MB** | Observed |\n")
        w(f"| **VAE Decode & MP4** | 33 frames valid, 0 NaNs | {(workload_res or {}).get('vae', {}).get('mp4_ok')} | {'🟢 OK' if (workload_res or {}).get('vae', {}).get('mp4_ok') else '🔴 FAILED'} |\n\n")
        w("## 2. Milestone Telemetry Breakdown\n\n")
        w(f"- **S0 Idle Baseline:** {s0_mean:.1f} MB (spread: {s0_spread:.1f} MB)\n")
        w(f"- **Control A Delta:** {control_a['nvml_delta_mb']:.1f} MB\n")
        w(f"- **S1 Operational Baseline:** {s1_mb:.1f} MB\n")
        w(f"- **S2 Workload Peak:** {s2_peak_nvml:.1f} MB\n")
        w(f"- **S3 Post-Workload Plateau:** {s3_mean:.1f} MB (spread: {s3_spread:.1f} MB)\n")
        w(f"- **Control B Delta:** {control_b['nvml_delta_mb']:.1f} MB\n")
        w(f"- **S4 Final Baseline:** {s4_mean:.1f} MB (spread: {s4_spread:.1f} MB)\n\n")
        w("## 3. Epistemological and Engineering Governance\n\n")
        if gate_a_pass and gate_b_pass:
            w("The 30-step full E2E run confirms that the adaptive runtime with cond->uncond seam release and step-start draining sustains the complete diffusion schedule within the physical VRAM gate of 4,800 MB.\n")
        else:
            w(f"The run failed to satisfy both gates simultaneously. Gate A observed {s2_peak_nvml:.1f} MB vs 4,800.0 MB limit. Gate B observed {delta_s0:.1f} MB vs 3,850.0 MB limit.\n")

    print(f">> Report saved to: {args.report}")
    print("=" * 80)
    print(f"  OFFICIAL CLASSIFICATION: {classification}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
