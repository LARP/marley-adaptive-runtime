"""
f7_d2b_allocator_live_boundary_probe.py
========================================
Phase F7-D2b -- Allocator LIVE-boundary & in-step profile probe (complementary).

Builds on F7-D2 (which showed the QUIESCENT seam holds ~no releasable pool).
This probe answers the Director's Option-2 question:

  Over the single DiT step, how much of the Reserved pool (~5.8 GB) is
  unavoidable SIMULTANEOUS LIVE demand vs allocator segment growth that a
  stream/order change might reclaim? And what is the seam state when the
  VAE-transition tensors are HELD LIVE (not dropped) rather than released?

Authorized by: awaiting Director authorization (2026-09-09).
Workload: 1280x720, 33 frames, ONE DiT step (adaptive), seed 42, guidance 5.0.
No VAE decode, no full denoise.

Evidence: MEASURED / OBSERVED / DERIVED / HYPOTHESIS tagged throughout.
Governance: runner + spec only. Runtime untouched (subclass seam as F7-D1/D2).
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
SAFE_ABORT_NVML_MB = 6050.0
CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
F7_0_BASELINE_NVML_MB = 6058.5
F7_0_RESERVED_MB = 5894.0
F7_0_ALLOC_MB = 2187.5


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


class TorchMemorySeriesSampler:
    """
    Background sampler logging a time series of torch allocated/reserved/free-pool
    plus an optional NVML sampler read, tagged with the active phase window.
    """

    def __init__(self, device: torch.device, interval_ms: float = 8.0,
                 nvml: Optional[NVMLSampler] = None) -> None:
        self.device = device
        self.interval_s = interval_ms / 1000.0
        self.nvml = nvml
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._phase = "init"
        self._phase_lock = threading.Lock()
        self.series: List[Dict[str, Any]] = []

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                nvml = self.nvml.sample_instant()["nvml_used_mb"] if self.nvml else 0.0
                with self._phase_lock:
                    phase = self._phase
                self.series.append({
                    "t_ms": round(len(self.series) * self.interval_s * 1000.0, 1),
                    "phase": phase,
                    "alloc_mb": round(alloc, 2),
                    "reserved_mb": round(reserved, 2),
                    "free_pool_mb": round(reserved - alloc, 2),
                    "nvml_mb": round(nvml, 2),
                })
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class AllocatorLiveProbePipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched) for the live-boundary + in-step profile."""

    def run_probe(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        guidance_scale: float,
        seed: int,
        mode: str,
        nvml: NVMLSampler,
        mem_series: TorchMemorySeriesSampler,
    ) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {"phases": {}, "windows": [], "seam": {}}

        generator = torch.Generator(device="cpu").manual_seed(seed)

        mem_series.set_phase("prompt_encode")
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        mem_series.set_phase("latent_prep")
        self.pipe.scheduler.set_timesteps(5, device=device)
        timesteps = self.pipe.scheduler.timesteps
        latents = self.pipe.prepare_latents(
            batch_size=1, num_channels_latents=self.transformer.config.in_channels,
            height=height, width=width, num_frames=num_frames,
            dtype=torch.float32, device=device, generator=generator,
        )
        out["phases"]["latent_shape"] = list(latents.shape)

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

        def _window(phase: str):
            mem_series.set_phase(phase)

        # --- Step body (inline, phase-tagged) -------------------------------
        t_step_start = time.perf_counter()
        lat = latents
        _window("rope_patch_embed")
        latent_model_input = lat.to(dtype)
        rotary_emb = self.transformer.rope(latent_model_input)
        hidden_states = self.transformer.patch_embedding(latent_model_input)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        timestep = timesteps[0].expand(latent_model_input.shape[0])
        temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
            timestep, prompt_embeds, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # --- COND 30-block pass ---
        _window("cond_blocks_30")
        hidden_states_cond = hidden_states.clone()
        self._execute_dit_custom_forward(
            streamer=streamer, mode=mode, hidden_states=hidden_states_cond,
            encoder_hidden_states=enc_hidden_states, timestep_proj=timestep_proj,
            rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames,
        )
        _window("cond_unpatchify")
        shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states_cond.device)
        scale = scale.to(hidden_states_cond.device)
        hidden_states_cond = (self.transformer.norm_out(hidden_states_cond.float()) * (1 + scale) + shift).type_as(hidden_states_cond)
        hidden_states_cond = self.transformer.proj_out(hidden_states_cond)
        hidden_states_cond = hidden_states_cond.reshape(
            batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
        ).permute(0, 7, 1, 4, 2, 5, 3, 6)
        noise_pred_cond = hidden_states_cond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # --- UNCOND 30-block pass ---
        _window("uncond_prep")
        temb_u, timestep_proj_u, enc_hidden_states_u, _ = self.transformer.condition_embedder(
            timestep, negative_prompt_embeds, None
        )
        timestep_proj_u = timestep_proj_u.unflatten(1, (6, -1))
        hidden_states_uncond = hidden_states.clone()
        _window("uncond_blocks_30")
        self._execute_dit_custom_forward(
            streamer=streamer, mode=mode, hidden_states=hidden_states_uncond,
            encoder_hidden_states=enc_hidden_states_u, timestep_proj=timestep_proj_u,
            rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames,
        )
        _window("uncond_unpatchify")
        shift_u, scale_u = (self.transformer.scale_shift_table.to(temb_u.device) + temb_u.unsqueeze(1)).chunk(2, dim=1)
        shift_u = shift_u.to(hidden_states_uncond.device)
        scale_u = scale_u.to(hidden_states_uncond.device)
        hidden_states_uncond = (self.transformer.norm_out(hidden_states_uncond.float()) * (1 + scale_u) + shift_u).type_as(hidden_states_uncond)
        hidden_states_uncond = self.transformer.proj_out(hidden_states_uncond)
        hidden_states_uncond = hidden_states_uncond.reshape(
            batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
        ).permute(0, 7, 1, 4, 2, 5, 3, 6)
        noise_pred_uncond = hidden_states_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        _window("combine_scheduler_step")
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = self.pipe.scheduler.step(noise_pred, timesteps[0], latents).prev_sample
        step_s = time.perf_counter() - t_step_start
        torch.cuda.synchronize(device)
        out["phases"]["single_step_s"] = round(step_s, 2)
        _window("seam_live_held")

        # --- Seam A: LIVE-HELD (references to VAE-transition tensors kept) ----
        # Hold latents + prompt embeds (as the runtime does when entering VAE).
        live_holder: List[Any] = [latents, prompt_embeds, negative_prompt_embeds]
        gc.collect()
        torch.cuda.synchronize(device)
        seam_live = {
            "label": "seam_live_held",
            "nvml_mb": nvml.sample_instant()["nvml_used_mb"],
            "alloc_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
            "free_pool_mb": round((torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / (1024 * 1024), 2),
        }
        out["seam"]["live_held"] = seam_live
        print(f"   [SEAM live_held] NVML={seam_live['nvml_mb']:.1f} Alloc={seam_live['alloc_mb']:.1f} "
              f"Res={seam_live['reserved_mb']:.1f} FreePool={seam_live['free_pool_mb']:.1f}")

        # --- Seam B: RELEASED (drop refs + empty_cache) ----------------------
        mem_series.set_phase("seam_release")
        try:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
        finally:
            self.transformer.blocks = transformer_blocks_orig
        live_holder.clear()
        del noise_pred, noise_pred_cond, noise_pred_uncond, hidden_states_cond, hidden_states_uncond
        del hidden_states, rotary_emb, temb, temb_u
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        seam_rel = {
            "label": "seam_released",
            "nvml_mb": nvml.sample_instant()["nvml_used_mb"],
            "alloc_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
            "free_pool_mb": round((torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / (1024 * 1024), 2),
        }
        out["seam"]["released"] = seam_rel
        print(f"   [SEAM released  ] NVML={seam_rel['nvml_mb']:.1f} Alloc={seam_rel['alloc_mb']:.1f} "
              f"Res={seam_rel['reserved_mb']:.1f} FreePool={seam_rel['free_pool_mb']:.1f}")

        _window("done")
        return out


def _window_stats(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-phase max alloc / reserved / free-pool over the time series."""
    phases: Dict[str, Dict[str, float]] = {}
    for s in series:
        p = s["phase"]
        if p not in phases:
            phases[p] = {"alloc_mb": 0.0, "reserved_mb": 0.0, "free_pool_mb": 0.0, "nvml_mb": 0.0, "n": 0}
        for k in ("alloc_mb", "reserved_mb", "free_pool_mb", "nvml_mb"):
            phases[p][k] = max(phases[p][k], s[k])
        phases[p]["n"] += 1
    return phases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7-D2b -- Allocator live-boundary & in-step profile")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default="logs/f7_d2b_allocator_live_boundary_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D2B_ALLOCATOR_LIVE_BOUNDARY_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D2b: ALLOCATOR LIVE-BOUNDARY & IN-STEP PROFILE")
    print("  (Complement to F7-D2. Results tagged MEASURED/OBSERVED/DERIVED/HYPOTHESIS)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: 1 (single)")
    print(f"  Latent: [1, 16, {latent_f}, {latent_h}, {latent_w}] | Mode: {args.mode} | Seed: {args.seed}")
    print(f"  Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB (physical, OOM protection)")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print("  Windows: rope/patch -> cond_blocks -> cond_unpatchify -> uncond_blocks")
        print("           -> uncond_unpatchify -> combine/scheduler -> seam_live_held -> seam_released")
        print("  Seam A keeps VAE-transition tensors LIVE; Seam B drops them + empty_cache.")
        print("[DRY-RUN] Validated. Awaiting execution authorization.")
        return

    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    mem_series = TorchMemorySeriesSampler(torch.device(args.device), interval_ms=8.0, nvml=nvml)
    mem_series.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    probe: Optional[Dict[str, Any]] = None
    error_occurred: Optional[str] = None
    t_start = time.perf_counter()
    try:
        pipeline = AllocatorLiveProbePipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16,
        )
        print(f"\n>> Executing F7-D2b single DiT step @ {args.width}x{args.height} ({args.frames}f)...")
        probe = pipeline.run_probe(
            prompt=args.prompt, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            guidance_scale=args.guidance, seed=args.seed, mode=args.mode,
            nvml=nvml, mem_series=mem_series,
        )
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D2b ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = nvml.stop()
        mem_series.stop()

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT OPERATIVO] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB (proteccion OOM).")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()

    win_stats = _window_stats(mem_series.series)
    series = mem_series.series
    # Persist full series in a compact form.
    full_series = [dict(s) for s in series]

    seam = (probe or {}).get("seam", {})
    delta_res = None
    delta_nvml = None
    if "live_held" in seam and "released" in seam:
        delta_res = round(seam["live_held"]["reserved_mb"] - seam["released"]["reserved_mb"], 2)
        delta_nvml = round(seam["live_held"]["nvml_mb"] - seam["released"]["nvml_mb"], 2)

    print("\n" + "=" * 80)
    print("  PHASE F7-D2b: SCORECARD")
    print("=" * 80)
    print(f"  Peak NVML fisico        [OBSERVED] : {peak_nvml:7.1f} MB")
    print(f"  Peak Allocated          [OBSERVED] : {peak_alloc:7.1f} MB (F7-0: {F7_0_ALLOC_MB})")
    print(f"  Peak Reserved           [OBSERVED] : {peak_reserved:7.1f} MB (F7-0: {F7_0_RESERVED_MB})")
    print(f"  Host RSS                [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total        [OBSERVED] : {t_total:7.2f} s ({t_total/60:.2f} min)")
    for ph, st in win_stats.items():
        print(f"  phase[{ph:22s}] max alloc={st['alloc_mb']:6.1f} res={st['reserved_mb']:6.1f} "
              f"freePool={st['free_pool_mb']:6.1f} nvml={st['nvml_mb']:6.1f} (n={st['n']})")
    if delta_res is not None:
        print(f"  Seam delta Reserved (live->released): {delta_res:+.1f} MB")
        print(f"  Seam delta NVML      (live->released): {delta_nvml:+.1f} MB")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D2b (Allocator live-boundary & in-step profile)",
        "evidence_class": "All measured/observed; interpretation DERIVED",
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps_executed": 1, "mode": args.mode, "seed": args.seed,
                     "guidance_scale": args.guidance},
        "phases_s": (probe or {}).get("phases", {}),
        "seam": seam,
        "seam_delta_mb": {"reserved": delta_res, "nvml": delta_nvml},
        "phase_max_windows": win_stats,
        "time_series": full_series,
        "memory_peaks_mb": {"peak_nvml_used": round(peak_nvml, 1), "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_torch_reserved": round(peak_reserved, 1), "peak_process_ram": round(peak_ram, 1)},
        "verdict": {"execution_completed": error_occurred is None,
                    "peak_nvml_mb": round(peak_nvml, 1),
                    "hard_gate_4800mb_reference": peak_nvml <= HARD_GATE_VRAM_MB,
                    "error_detail": error_occurred},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7-D2b Allocator Live-Boundary & In-Step Profile -- Pilot Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, 1 DiT step, mode `{args.mode}`  \n\n")
        w("---\n\n## 1. Per-Phase Maxima (OBSERVED, during single step)\n\n")
        w("| Phase | Max Alloc (MB) | Max Reserved (MB) | Max FreePool (MB) | Max NVML (MB) | Samples |\n")
        w("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for ph, st in win_stats.items():
            w(f"| {ph} | {st['alloc_mb']} | {st['reserved_mb']} | {st['free_pool_mb']} | {st['nvml_mb']} | {st['n']} |\n")
        w("\n## 2. Seam (OBSERVED)\n\n")
        if "live_held" in seam:
            lh = seam["live_held"]
            w(f"- **live_held:** NVML {lh['nvml_mb']} MB | Alloc {lh['alloc_mb']} MB | Reserved {lh['reserved_mb']} MB | FreePool {lh['free_pool_mb']} MB\n")
        if "released" in seam:
            rl = seam["released"]
            w(f"- **released:** NVML {rl['nvml_mb']} MB | Alloc {rl['alloc_mb']} MB | Reserved {rl['reserved_mb']} MB | FreePool {rl['free_pool_mb']} MB\n")
        if delta_res is not None:
            w(f"- Delta Reserved (live_held -> released): **{delta_res:+.1f} MB**\n")
            w(f"- Delta NVML (live_held -> released): **{delta_nvml:+.1f} MB**\n")
        w("\n## 3. Memory Peaks (OBSERVED)\n\n")
        w(f"- Peak physical NVML: {peak_nvml:.1f} MB\n")
        w(f"- Peak Allocated: {peak_alloc:.1f} MB\n")
        w(f"- Peak Reserved: {peak_reserved:.1f} MB\n")
        w("\n## 4. Interpretation (DERIVED)\n\n")
        w("Phase maxima show whether Reserved tracks Allocated closely (unavoidable simultaneous live "
          "demand) or carries a persistent large free-pool cushion (potential stream/order reclaim). The "
          "live_held vs released seam quantifies the VAE-transition state. This is diagnostic; it does not "
          "authorize runtime changes.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
