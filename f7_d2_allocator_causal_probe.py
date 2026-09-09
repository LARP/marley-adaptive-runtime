"""
f7_d2_allocator_causal_probe.py
===============================
Phase F7-D2 -- Allocator Causal Release Probe (experimental diagnostic).

Authorized by: awaiting Director execution authorization (spec 2026-09-09).
Workload: 1280x720, 33 frames, a SINGLE DiT denoising step (adaptive mode),
seed 42, guidance 5.0. No full denoise, no VAE decode.

Question (single variable):
  Does a controlled release of the PyTorch caching allocator pool
  (drop dead refs -> gc -> synchronize -> empty_cache -> synchronize) at the
  post-denoise boundary produce a significant, reproducible reduction of
  PyTorch Reserved and physical NVML?

Governance:
  * Only this file + spec are modified. The runtime (MarleyEndToEndPipeline,
    VAE, streamers, AdaptiveEngine, diffusers, end_to_end.py) is NOT modified.
  * Seam technique reused from F7-D1: subclass the pipeline to replicate the
    stage logic so a controlled release can be inserted at the DiT->VAE
    boundary without editing the base class.
  * Independent variable = the empty_cache+gc release only. No weight
    residency changes, no VAE, no full 5-step denoise.
  * Evidence honesty: every printed number is tagged
    MEASURED / OBSERVED / DERIVED / HYPOTHESIS. Dry-run performs NO inference.
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

HARD_GATE_VRAM_MB = 4800.0
SAFE_ABORT_NVML_MB = 6050.0      # near demonstrated physical tolerance (~6.1 GB), protect from OOM
CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
F7_0_BASELINE_NVML_MB = 6058.5
F7_D1_NVML_MB = 6088.4
F7_0_RESERVED_MB = 5894.0
F7_0_ALLOC_MB = 2187.5


def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class ContinuousNVMLSampler:
    """High-frequency hardware VRAM sampler plus temp/clock/util (OBSERVED)."""

    def __init__(self, device_index: int = 0, interval_ms: float = 25.0,
                 safe_abort_mb: Optional[float] = None) -> None:
        self.interval_s = interval_ms / 1000.0
        self.device_index = device_index
        self.safe_abort_mb = safe_abort_mb
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        self._handle = None
        self.peak_mb = 0.0
        self.peak_timestamp: Optional[str] = None
        self.safe_abort_triggered = False
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
                    if self.safe_abort_mb is not None and used_mb > self.safe_abort_mb:
                        self.safe_abort_triggered = True
            except Exception:
                pass
            time.sleep(self.interval_s)

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


class AllocatorCausalPipeline(MarleyEndToEndPipeline):
    """
    Subclasses MarleyEndToEndPipeline (non-invasive, mirroring F7-D1 technique)
    to run a SINGLE DiT denoising step and then expose the DiT->VAE boundary so
    the controlled allocator release can be applied and measured.
    """

    def run_single_step_and_probe(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        guidance_scale: float,
        seed: int,
        mode: str,
        release_reps: int,
        monitor: ContinuousNVMLSampler,
    ) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype

        probe: Dict[str, Any] = {
            "phases": {},
            "boundary": {},
        }
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ---- Phase 1: prompt encoding ------------------------------------
        t0 = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=torch.device("cpu"),
            dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        probe["phases"]["prompt_encoding_s"] = round(time.perf_counter() - t0, 2)

        # ---- Phase 2: latents & scheduler ---------------------------------
        t0 = time.perf_counter()
        self.pipe.scheduler.set_timesteps(5, device=device)
        timesteps = self.pipe.scheduler.timesteps
        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        probe["phases"]["dit_prep_s"] = round(time.perf_counter() - t0, 2)
        probe["phases"]["latent_shape"] = list(latents.shape)

        # ---- Phase 3: streamer --------------------------------------------
        def pressure_sampler() -> float:
            return monitor.sample_instant()["nvml_used_mb"]

        if mode == "adaptive":
            streamer = AdaptiveEngine(
                blocks=self.blocks,
                device=device,
                dtype=dtype,
                window=1,
                pressure_sampler=pressure_sampler,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # ---- Phase 4: exactly ONE denoising step (inner scope) ------------
        torch.set_grad_enabled(False)
        batch_size, num_channels, num_frames_in, h_in, w_in = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        post_patch_num_frames = num_frames_in // p_t
        post_patch_height = h_in // p_h
        post_patch_width = w_in // p_w

        transformer_blocks_orig = self.transformer.blocks
        self.transformer.blocks = nn.ModuleList([])

        def _one_step(t: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
            """Runs a single cond+uncond DiT pass; all locals die on return."""
            latent_model_input = lat.to(dtype)
            rotary_emb = self.transformer.rope(latent_model_input)
            hidden_states = self.transformer.patch_embedding(latent_model_input)
            hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
            timestep = t.expand(latent_model_input.shape[0])
            temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
                timestep, prompt_embeds, None
            )
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

            hidden_states_cond = hidden_states.clone()
            self._execute_dit_custom_forward(
                streamer=streamer, mode=mode,
                hidden_states=hidden_states_cond,
                encoder_hidden_states=enc_hidden_states,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
                num_steps=1, num_frames=num_frames,
            )
            shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states_cond.device)
            scale = scale.to(hidden_states_cond.device)
            hidden_states_cond = (self.transformer.norm_out(hidden_states_cond.float()) * (1 + scale) + shift).type_as(hidden_states_cond)
            hidden_states_cond = self.transformer.proj_out(hidden_states_cond)
            hidden_states_cond = hidden_states_cond.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            ).permute(0, 7, 1, 4, 2, 5, 3, 6)
            noise_pred_cond = hidden_states_cond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            temb_uncond, timestep_proj_uncond, enc_hidden_states_uncond, _ = self.transformer.condition_embedder(
                timestep, negative_prompt_embeds, None
            )
            timestep_proj_uncond = timestep_proj_uncond.unflatten(1, (6, -1))
            hidden_states_uncond = hidden_states.clone()
            self._execute_dit_custom_forward(
                streamer=streamer, mode=mode,
                hidden_states=hidden_states_uncond,
                encoder_hidden_states=enc_hidden_states_uncond,
                timestep_proj=timestep_proj_uncond,
                rotary_emb=rotary_emb,
                num_steps=1, num_frames=num_frames,
            )
            shift_u, scale_u = (self.transformer.scale_shift_table.to(temb_uncond.device) + temb_uncond.unsqueeze(1)).chunk(2, dim=1)
            shift_u = shift_u.to(hidden_states_uncond.device)
            scale_u = scale_u.to(hidden_states_uncond.device)
            hidden_states_uncond = (self.transformer.norm_out(hidden_states_uncond.float()) * (1 + scale_u) + shift_u).type_as(hidden_states_uncond)
            hidden_states_uncond = self.transformer.proj_out(hidden_states_uncond)
            hidden_states_uncond = hidden_states_uncond.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            ).permute(0, 7, 1, 4, 2, 5, 3, 6)
            noise_pred_uncond = hidden_states_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            return self.pipe.scheduler.step(noise_pred, t, lat).prev_sample

        t_step_start = time.perf_counter()
        latents = _one_step(timesteps[0], latents)
        step_s = time.perf_counter() - t_step_start
        torch.cuda.synchronize(device)
        probe["phases"]["single_step_s"] = round(step_s, 2)

        # Restore blocks / release streamer (mirrors runtime finally path).
        try:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
        finally:
            self.transformer.blocks = transformer_blocks_orig

        def _snapshot(label: str) -> Dict[str, Any]:
            alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
            stats = torch.cuda.memory_stats(device)
            inst = monitor.sample_instant()
            snap = {
                "label": label,
                "timestamp": datetime.datetime.now().isoformat(),
                "nvml_used_mb": inst["nvml_used_mb"],
                "torch_alloc_mb": round(alloc_mb, 2),
                "torch_reserved_mb": round(reserved_mb, 2),
                "allocator_delta_mb": round(reserved_mb - alloc_mb, 2),
                "segment_current": stats.get("segment.all.current", 0),
                "segment_allocated": stats.get("segment.all.allocated", 0),
                "segment_freed": stats.get("segment.all.freed", 0),
                "host_rss_mb": round(get_process_ram_mb(), 2),
                "gpu_temp_c": inst["gpu_temp_c"],
                "gpu_clock_mhz": inst["gpu_clock_mhz"],
                "gpu_util_pct": inst["gpu_util_pct"],
            }
            probe["boundary"][label] = snap
            print(
                f"   [BOUNDARY {label}] NVML={snap['nvml_used_mb']:7.1f} MB | "
                f"Alloc={snap['torch_alloc_mb']:7.1f} | Reserved={snap['torch_reserved_mb']:7.1f} | "
                f"Delta={snap['allocator_delta_mb']:7.1f} | Segs(cur/alloc/freed)="
                f"{snap['segment_current']}/{snap['segment_allocated']}/{snap['segment_freed']}"
            )
            return snap

        # Boundary probe: retained state, then repeated empty_cache+gc releases.
        gc.collect()
        torch.cuda.synchronize(device)
        _snapshot("retained")

        for rep in range(release_reps):
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            _snapshot(f"released_rep{rep + 1}")

        return probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Runtime Phase F7-D2 -- Allocator Causal Release Probe")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--release-reps", type=int, default=2)
    p.add_argument("--output", type=str, default="logs/f7_d2_allocator_causal_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D2_ALLOCATOR_CAUSAL_PROBE_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true", help="Validate params/env WITHOUT inference or results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D2: ALLOCATOR CAUSAL RELEASE PROBE")
    print("  (Minimal monovariable causal probe. Results tagged MEASURED/OBSERVED/DERIVED/HYPOTHESIS)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: 1 (single)")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Guidance: {args.guidance}")
    print(f"  Latent shape: [1, 16, {latent_f}, {latent_h}, {latent_w}]")
    print(f"  Boundary release reps: {args.release_reps} (empty_cache + gc only)")
    print(f"  Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB (physical, protects against OOM)")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference, NO results.")
        print("  Seam: subclass of MarleyEndToEndPipeline (runtime untouched).")
        print("  Single-variable: empty_cache + gc at the post-denoise boundary only.")
        print("  No weight-residency change, no VAE decode, no full denoise.")
        print("  Boundary snapshots: retained -> released_rep1..N (before/after).")
        print("[DRY-RUN] Parameters validated. Ready for real execution upon authorization.")
        return

    monitor = ContinuousNVMLSampler(device_index=0, interval_ms=25.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    probe_data: Optional[Dict[str, Any]] = None
    error_occurred: Optional[str] = None
    abort_kind: Optional[str] = None
    t_start = time.perf_counter()

    try:
        pipeline = AllocatorCausalPipeline(
            device=args.device,
            dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16,
            text_dtype=torch.bfloat16,
        )
        print(f"\n>> Executing F7-D2 single DiT step @ {args.width}x{args.height} ({args.frames}f)...")
        probe_data = pipeline.run_single_step_and_probe(
            prompt=args.prompt,
            negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            guidance_scale=args.guidance,
            seed=args.seed,
            mode=args.mode,
            release_reps=args.release_reps,
            monitor=monitor,
        )
    except Exception as exc:
        abort_kind = "RUNTIME_ERROR"
        error_occurred = str(exc)
        print(f"\n[F7-D2 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = monitor.stop()

    if monitor.safe_abort_triggered:
        abort_kind = "SAFE_ABORT_OPERATIONAL"
        print(f"\n[SAFE ABORT OPERATIVO] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB (proteccion OOM).")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram = get_process_ram_mb()

    # ---- Derived deltas across the retained->released snapshots ----------
    boundary = (probe_data or {}).get("boundary", {})
    derived: Dict[str, Any] = {}
    if "retained" in boundary and "released_rep1" in boundary:
        ret = boundary["retained"]
        rel = boundary["released_rep1"]
        derived["delta_reserved_mb"] = round(ret["torch_reserved_mb"] - rel["torch_reserved_mb"], 2)
        derived["delta_nvml_mb"] = round(ret["nvml_used_mb"] - rel["nvml_used_mb"], 2)
        derived["segments_freed_between"] = max(0, rel["segment_freed"] - ret["segment_freed"])
    deltas = [boundary[k] for k in boundary if k.startswith("released_")]
    nvml_released_vals = [d["nvml_used_mb"] for d in deltas]
    derived["nvml_released_reproducibility_range_mb"] = (
        [round(min(nvml_released_vals), 2), round(max(nvml_released_vals), 2)] if nvml_released_vals else None
    )

    # Classification helper text (interpretive, not an invented gate).
    interp = "n/a"
    if "delta_reserved_mb" in derived:
        dr = derived["delta_reserved_mb"]
        dn = derived.get("delta_nvml_mb", 0.0)
        if dr > 100 and dn > 150:
            interp = "Case A candidate (reserved AND NVML fall together)"
        elif dr > 100 and dn <= 150:
            interp = "Case B candidate (reserved returns but NVML stays high)"
        elif dr > 100:
            interp = "Case C candidate (partial)"
        else:
            interp = "Case D candidate (pool not releasable at this boundary)"

    print("\n" + "=" * 80)
    print("  PHASE F7-D2: BOUNDARY SCORECARD")
    print("=" * 80)
    print(f"  Peak NVML fisico        [OBSERVED] : {peak_nvml:7.1f} MB (Gate<=4800 reference)")
    print(f"  PyTorch Allocated peak  [OBSERVED] : {peak_alloc:7.1f} MB (F7-0: {F7_0_ALLOC_MB})")
    print(f"  PyTorch Reserved peak   [OBSERVED] : {peak_reserved:7.1f} MB (F7-0: {F7_0_RESERVED_MB})")
    print(f"  Host RSS                [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total        [OBSERVED] : {t_total:7.2f} s ({t_total/60:.2f} min)")
    if "delta_reserved_mb" in derived:
        print(f"  Delta Reserved (rel1)   [DERIVED]  : {derived['delta_reserved_mb']:+7.1f} MB")
        print(f"  Delta NVML (rel1)       [DERIVED]  : {derived['delta_nvml_mb']:+7.1f} MB")
        print(f"  Segments freed          [DERIVED]  : {derived['segments_freed_between']}")
    if derived.get("nvml_released_reproducibility_range_mb"):
        print(f"  NVML released spread    [DERIVED]  : {derived['nvml_released_reproducibility_range_mb']} MB")
    print(f"  Interpretation           [DERIVED]  : {interp}")
    print(f"  Abort                   : {abort_kind or 'none'}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D2 (Allocator Causal Release Probe)",
        "evidence_class": "All measured/observed; interpretation labelled DERIVED",
        "workload": {
            "resolution": f"{args.width}x{args.height}",
            "frames": args.frames,
            "steps_executed": 1,
            "mode": args.mode,
            "seed": args.seed,
            "guidance_scale": args.guidance,
            "release_reps": args.release_reps,
        },
        "phases_s": (probe_data or {}).get("phases", {}),
        "boundary_snapshots": boundary,
        "derived_deltas": derived,
        "interpretation": interp,
        "memory_peaks_mb": {
            "peak_nvml_used": round(peak_nvml, 1),
            "peak_nvml_timestamp": monitor.peak_timestamp,
            "peak_torch_alloc": round(peak_alloc, 1),
            "peak_torch_reserved": round(peak_reserved, 1),
            "peak_process_ram": round(peak_ram, 1),
        },
        "comparison_vs_baselines": {
            "f7_0_baseline_nvml_mb": F7_0_BASELINE_NVML_MB,
            "f7_d1_nvml_mb": F7_D1_NVML_MB,
            "f7_0_reserved_mb": F7_0_RESERVED_MB,
            "f7_0_alloc_mb": F7_0_ALLOC_MB,
        },
        "verdict": {
            "execution_completed": error_occurred is None,
            "abort_kind": abort_kind,
            "hard_gate_4800mb_reference": peak_nvml <= HARD_GATE_VRAM_MB,
            "error_detail": error_occurred,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    # ---- Report (English, honest labels) ---------------------------------
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7-D2 Allocator Causal Release Probe -- Pilot Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, 1 DiT step, mode `{args.mode}`  \n\n")
        w("**Evidence:** `MEASURED` instrumented; `OBSERVED` sampled; `DERIVED` computed; `HYPOTHESIS` not demonstrated.\n\n")
        w("---\n\n## 1. Boundary Snapshots (OBSERVED)\n\n")
        w("| Snapshot | NVML (MB) | Alloc (MB) | Reserved (MB) | Delta (MB) | Seg(cur/alloc/freed) | RAM (MB) |\n")
        w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for label, snap in boundary.items():
            w(f"| {label} | {snap['nvml_used_mb']} | {snap['torch_alloc_mb']} | {snap['torch_reserved_mb']} | "
              f"{snap['allocator_delta_mb']} | {snap['segment_current']}/{snap['segment_allocated']}/{snap['segment_freed']} | "
              f"{snap['host_rss_mb']} |\n")
        w("\n## 2. Derived Deltas\n\n")
        w(f"- Delta Reserved (retained vs released_rep1): **{derived.get('delta_reserved_mb', 'n/a')} MB**\n")
        w(f"- Delta NVML (retained vs released_rep1): **{derived.get('delta_nvml_mb', 'n/a')} MB**\n")
        w(f"- Segments freed between snapshots: **{derived.get('segments_freed_between', 'n/a')}**\n")
        w(f"- NVML reproducibility spread (released reps): **{derived.get('nvml_released_reproducibility_range_mb', 'n/a')}**\n")
        w(f"- Interpretation: **{interp}**\n")
        w("\n## 3. Memory Peaks (OBSERVED)\n\n")
        w(f"- Peak physical NVML: {peak_nvml:.1f} MB (F7-0 reference {F7_0_BASELINE_NVML_MB:.1f} MB)\n")
        w(f"- Peak PyTorch Allocated: {peak_alloc:.1f} MB (F7-0 {F7_0_ALLOC_MB:.1f} MB)\n")
        w(f"- Peak PyTorch Reserved: {peak_reserved:.1f} MB (F7-0 {F7_0_RESERVED_MB:.1f} MB)\n")
        w("\n## 4. Interpretation & Next Decision\n\n")
        w("This is a diagnostic causal micro-test at the DiT->VAE boundary. It does NOT by itself authorize "
          "any permanent runtime change. The interpretation above informs the Director whether H1 (allocator "
          "retention as a physical driver) gains causal support or must be revised toward a WDDM/driver-residency "
          "explanation.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
