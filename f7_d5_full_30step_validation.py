"""
f7_d5_full_30step_validation.py
================================
Phase F7-D5 (full 30-step E2E) -- Validation of the cond->uncond seam release
+ step-start drain across the COMPLETE denoising schedule + VAE.

Workload: 1280x720, 33 frames, 30 adaptive steps, seed 42.
Intervention (unchanged from the F7-D4 favorable configuration, "seam_and_step_start"):
  - empty_cache at every cond->uncond seam within each step;
  - empty_cache (step-start drain) before the cond pass of every step.
No other optimization. Runtime untouched (subclass seam).

Goal (F7-D5 integration milestone):
  Confirm that over 30 FULL continuous steps the Peak Físico NVML stays strictly
  <= 4,800.0 MB with nominal cadence (~60 s/step, total ~32-35 min), VAE decode
  completes, and 0 NaNs. FAVORABLE authorizes formal integration of the seam
  release into marley/pipeline/end_to_end.py gated to resolutions > 480p.

Kill gates (ROADMAP F7-D5): Peak NVML > 4,800.0 MB, OOM, NaN/Inf, or
wall-clock > 45 min => confine 720p to experimental / low-res focus.

Evidence: MEASURED / OBSERVED / DERIVED / HYPOTHESIS.
Governance: isolated runner + spec only until this milestone is FAVORABLE.
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

HARD_GATE_VRAM_MB = 4800.0
WALLCLOCK_KILL_GATE_S = 45 * 60          # 45 min upper bound (ROADMAP F7-D5)
SAFE_ABORT_NVML_MB = 5050.0
CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
CANONICAL_FPS = 16
DEFAULT_STEPS = 30
F7_0_RESERVED_MB = 5894.0
D3_PEAK_NVML_MB = 4403.5
D3_PEAK_RESERVED_MB = 3576.0
D4_PEAK_NVML_MB = 4708.9
D4_PEAK_RESERVED_MB = 3776.0


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
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.peak_mb = info.used / (1024 * 1024)
        except Exception as exc:
            print(f"[WARN] NVML init failed: {exc}", file=sys.stderr)

    def start(self) -> None:
        if not self._available:
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


class FullE2ESeamReleasePipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched). Replicates generate() with empty_cache
    at every cond->uncond seam AND a step-start drain before the cond pass."""

    def run_validation(self, prompt, negative_prompt, height, width, num_frames,
                       num_steps, guidance_scale, seed, mode, nvml,
                       output_video_path) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {}
        t_global = time.perf_counter()

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # encode
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # prep
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
        seam_release_count = 0
        step_start_release_count = 0
        denoise_start = time.perf_counter()

        try:
            for step_idx, t in enumerate(timesteps):
                ts = time.perf_counter()
                # F7-D5 (D4 favorable config): drain allocator carryover at step start
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                step_start_release_count += 1
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

                # F7-D3 INTERVENTION: controlled release at cond->uncond seam
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                seam_release_count += 1

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
                print(f"   Step [{step_idx+1}/{num_steps}] {step_s:.2f}s | end Reserved={snap['reserved_mb']:.1f} "
                      f"NVML={snap['nvml_mb']:.1f} (peak so far {nvml.peak_mb:.1f})")

        finally:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
            torch.cuda.empty_cache()
            self.transformer.blocks = transformer_blocks_orig

        denoise_s = time.perf_counter() - denoise_start
        out["seam_release_count"] = seam_release_count
        out["step_start_release_count"] = step_start_release_count
        out["step_times_s"] = [round(x, 2) for x in step_times]
        out["step_end_reserved_mb"] = step_end_reserved
        out["denoise_time_s"] = round(denoise_s, 2)
        out["cadence_avg_s"] = round(sum(step_times) / len(step_times), 2) if step_times else None

        # ---- VAE decode (tiled bf16, F5-A rule) ----
        out["vae"] = {}
        if output_video_path is not None:
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
            # export
            try:
                frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                export_to_video(frames[0], output_video_path, fps=CANONICAL_FPS)
                out["vae"]["mp4_ok"] = os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0
                out["vae"]["output_video"] = output_video_path
            except Exception as exc:
                out["vae"]["mp4_ok"] = False
                out["vae"]["export_error"] = str(exc)
        else:
            out["vae"]["decode_s"] = None
            out["vae"]["mp4_ok"] = False

        out["total_wall_s"] = round(time.perf_counter() - t_global, 2)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7-D5 -- full 30-step E2E seam-release validation")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-vae", action="store_true", help="Skip VAE decode/export")
    p.add_argument("--output", type=str, default="logs/f7_d5_full_30step_validation_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1
    out_mp4 = None if args.no_vae else f"logs/f7_d5_{args.width}x{args.height}_{args.frames}f_{args.steps}st_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D5 (FULL): 30-STEP E2E SEAM-RELEASE VALIDATION")
    print("  (Intervention = empty_cache at step start + cond->uncond seam, per F7-D4 favorable)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: {args.steps}")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Latent: [1,16,{latent_f},{latent_h},{latent_w}]")
    print(f"  Hard Gate: <= {HARD_GATE_VRAM_MB:.0f} MB | Wallclock Kill Gate: <= {WALLCLOCK_KILL_GATE_S/60:.0f} min | Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB")
    print(f"  D3 single-step ref: NVML {D3_PEAK_NVML_MB} MB, Reserved {D3_PEAK_RESERVED_MB} MB")
    print(f"  D4 10-step ref:     NVML {D4_PEAK_NVML_MB} MB, Reserved {D4_PEAK_RESERVED_MB} MB")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print(f"  {args.steps} denoise steps, each with step-start + cond->uncond empty_cache; VAE={'on' if not args.no_vae else 'off'}")
        print(f"  Estimated runtime ~ {args.steps * 60 / 60:.0f} min denoise + ~70 s VAE (based on D4 cadence).")
        print("[DRY-RUN] Validated. Awaiting execution authorization.")
        return

    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    probe = None
    error_occurred: Optional[str] = None
    t_start = time.perf_counter()
    try:
        pipeline = FullE2ESeamReleasePipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
        print(f"\n>> Executing F7-D5 ({args.steps} steps @ {args.width}x{args.height}, {args.frames}f)...")
        probe = pipeline.run_validation(
            prompt=args.prompt, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode=args.mode, nvml=nvml, output_video_path=out_mp4)
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D5 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = nvml.stop()

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT OPERATIVO] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB (proteccion OOM).")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()

    step_end_res = (probe or {}).get("step_end_reserved_mb", [])
    accumulation = None
    if len(step_end_res) >= 2:
        accumulation = round(step_end_res[-1] - step_end_res[0], 1)
    first_res = step_end_res[0] if step_end_res else None
    last_res = step_end_res[-1] if step_end_res else None

    vae_ok = bool((probe or {}).get("vae", {}).get("mp4_ok", False))
    nan_inf = bool((probe or {}).get("vae", {}).get("nan_inf", False))
    if probe and "vae" in probe and probe["vae"].get("decode_s") is None and args.no_vae:
        vae_ok = None

    gate_pass = (peak_nvml <= HARD_GATE_VRAM_MB and not nan_inf
                 and (vae_ok is not False) and t_total <= WALLCLOCK_KILL_GATE_S)
    if gate_pass:
        result = "FAVORABLE (30-step E2E peak <=4800, VAE ok, wallclock <=45min)"
    elif peak_nvml <= HARD_GATE_VRAM_MB and not nan_inf and (vae_ok is not False) and t_total > WALLCLOCK_KILL_GATE_S:
        result = "PARTIAL (mem OK but wallclock exceeded 45min)"
    else:
        result = "PARTIAL/NEGATIVE"

    print("\n" + "=" * 80)
    print("  PHASE F7-D5 (FULL): SCORECARD")
    print("=" * 80)
    print(f"  Peak NVML fisico        [OBSERVED] : {peak_nvml:7.1f} MB (D4 10-step: {D4_PEAK_NVML_MB:.1f})")
    print(f"  Peak Reserved           [OBSERVED] : {peak_reserved:7.1f} MB (D4: {D4_PEAK_RESERVED_MB:.1f})")
    print(f"  Peak Allocated          [OBSERVED] : {peak_alloc:7.1f} MB")
    print(f"  Host RSS                [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total        [OBSERVED] : {t_total:7.2f} s ({t_total/60:.2f} min)")
    if probe:
        print(f"  Denoise time            [OBSERVED] : {probe.get('denoise_time_s')} s")
        print(f"  Cadence avg             [OBSERVED] : {probe.get('cadence_avg_s')} s/step")
        print(f"  Seam releases           [OBSERVED] : {probe.get('seam_release_count')}")
        print(f"  Step-start releases     [OBSERVED] : {probe.get('step_start_release_count')}")
        print(f"  Step1-end Reserved      [OBSERVED] : {first_res} MB")
        print(f"  Step{args.steps}-end Reserved     [OBSERVED] : {last_res} MB")
    if accumulation is not None:
        print(f"  Reserved accumulation   [DERIVED]  : {accumulation:+6.1f} MB (last - first step-end)")
    if probe and "vae" in probe:
        v = probe["vae"]
        print(f"  VAE decode              [OBSERVED] : {v.get('decode_s')} s | NaN/Inf: {v.get('nan_inf')} | mp4: {v.get('mp4_ok')}")
    print(f"  Result                   [DERIVED]  : {result}")
    print(f"  Abort/Error             : {error_occurred or 'none'}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D5 (full 30-step E2E seam-release validation)",
        "evidence_class": "All measured/observed; derived computed",
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": args.mode, "seed": args.seed,
                     "guidance_scale": args.guidance},
        "intervention": "empty_cache at step start AND cond->uncond seam (F7-D4 favorable config), 30 full steps + VAE",
        "release_strategy": "seam_and_step_start",
        "per_step": {"step_times_s": (probe or {}).get("step_times_s"),
                     "step_end_reserved_mb": step_end_res},
        "denoise_time_s": (probe or {}).get("denoise_time_s"),
        "cadence_avg_s": (probe or {}).get("cadence_avg_s"),
        "seam_release_count": (probe or {}).get("seam_release_count"),
        "step_start_release_count": (probe or {}).get("step_start_release_count"),
        "reserved_accumulation_mb": accumulation,
        "memory_peaks_mb": {"peak_nvml_used": round(peak_nvml, 1),
                            "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_torch_reserved": round(peak_reserved, 1),
                            "peak_process_ram": round(peak_ram, 1)},
        "vae": (probe or {}).get("vae", {}),
        "comparison": {"d3_single_step_peak_nvml_mb": D3_PEAK_NVML_MB,
                       "d3_single_step_peak_reserved_mb": D3_PEAK_RESERVED_MB,
                       "d4_10step_peak_nvml_mb": D4_PEAK_NVML_MB,
                       "d4_10step_peak_reserved_mb": D4_PEAK_RESERVED_MB,
                       "f7_0_baseline_reserved_mb": F7_0_RESERVED_MB},
        "result_class": result,
        "verdict": {"execution_completed": error_occurred is None,
                    "hard_gate_4800mb_met": peak_nvml <= HARD_GATE_VRAM_MB,
                    "wallclock_45min_gate_met": t_total <= WALLCLOCK_KILL_GATE_S,
                    "nan_pass": not nan_inf,
                    "mp4_pass": vae_ok,
                    "error_detail": error_occurred},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7-D5 (Full) 30-Step End-to-End Seam-Release Validation -- Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, {args.steps} steps, mode `{args.mode}`  \n")
        w("**Intervention:** empty_cache at every step start AND cond->uncond seam "
          "(single-variable discipline from F7-D3/D4 favorable config).\n\n")
        w("---\n\n## 1. Result\n\n")
        w(f"- Peak NVML: {peak_nvml:.1f} MB (D4 10-step: {D4_PEAK_NVML_MB:.1f})\n")
        w(f"- Peak Reserved: {peak_reserved:.1f} MB (D4: {D4_PEAK_RESERVED_MB:.1f})\n")
        w(f"- Reserved accumulation (last-first step-end): {accumulation} MB\n")
        w(f"- Wall-clock total: {t_total:.1f} s ({t_total/60:.2f} min) | Kill gate {WALLCLOCK_KILL_GATE_S/60:.0f} min\n")
        if probe:
            w(f"- Denoise: {probe.get('denoise_time_s')} s; cadence {probe.get('cadence_avg_s')} s/step; "
              f"seam releases {probe.get('seam_release_count')}; step-start releases {probe.get('step_start_release_count')}\n")
        if probe and "vae" in probe:
            v = probe["vae"]
            w(f"- VAE decode: {v.get('decode_s')} s; NaN/Inf {v.get('nan_inf')}; mp4 {v.get('mp4_ok')}\n")
        w(f"- Classification: **{result}**\n")
        w("\n## 2. Per-step end-of-step Reserved (MB)\n\n")
        for i, r in enumerate(step_end_res):
            w(f"- step {i+1}: {r}\n")
        w("\n## 3. Governance\n\n")
        w("Full 30-step E2E causal validation in an isolated runner. If FAVORABLE, "
          "authorizes formal integration of the seam release into "
          "`marley/pipeline/end_to_end.py` gated to resolutions > 480p, preserving the "
          "480p baseline immutable (F6).\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
