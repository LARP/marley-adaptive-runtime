"""
f7_d3_cond_uncond_seam_release.py
==================================
Phase F7-D3 -- Controlled cond->uncond seam release (causal, single variable).

Question (one hypothesis):
  Does a controlled empty_cache release at the natural cond->uncond seam prevent
  or reduce the second Reserved/NVML growth of the uncond pass, keeping the peak
  under the 4,800 MB hard gate?

Workload identical to F7-D2c: 1280x720, 33 frames, ONE DiT step (adaptive),
seed 42. No VAE, no full denoise.

Single independent variable: gc.collect() + synchronize + empty_cache +
synchronize, inserted in the RUNNER at the cond->uncond seam. Runtime untouched
(subclass seam technique as F7-D2/D2b/D2c). Per-block hooks reused to observe the
uncond Reserved series.

Governance: runner + spec only. Does NOT authorize runtime modification.
Evidence: MEASURED / OBSERVED / DERIVED / HYPOTHESIS.
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
# D2c baseline (no seam release)
D2C_PEAK_NVML_MB = 6006.2
D2C_PEAK_RESERVED_MB = 5682.0
D2C_RES_AFTER_COND_MB = 3046.0


def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class NVMLSampler:
    """High-frequency physical VRAM sampler."""

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


class BlockMemoryProbe:
    """Per-block forward hooks (observational)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.current_pass = "cond"
        self.records: List[Dict[str, Any]] = []
        self._handles: List[Any] = []

    def _snap(self) -> Tuple[float, float, float, int, int, int]:
        stats = torch.cuda.memory_stats(self.device)
        a = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        r = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        return (a, r, r - a, int(stats.get("segment.all.current", 0)),
                int(stats.get("segment.all.allocated", 0)), int(stats.get("segment.all.freed", 0)))

    def _pre(self, bi):
        def hook(m, args):
            a, r, f, segc, sega, segf = self._snap()
            self.records.append({"block": bi, "pass": self.current_pass, "phase": "pre",
                                 "alloc_mb": round(a, 2), "reserved_mb": round(r, 2),
                                 "free_pool_mb": round(f, 2), "seg_current": segc,
                                 "seg_allocated": sega, "seg_freed": segf})
        return hook

    def _post(self, bi):
        def hook(m, args, output):
            a, r, f, segc, sega, segf = self._snap()
            self.records.append({"block": bi, "pass": self.current_pass, "phase": "post",
                                 "alloc_mb": round(a, 2), "reserved_mb": round(r, 2),
                                 "free_pool_mb": round(f, 2), "seg_current": segc,
                                 "seg_allocated": sega, "seg_freed": segf})
        return hook

    def attach(self, blocks: List[nn.Module]) -> None:
        for i, b in enumerate(blocks):
            self._handles.append(b.register_forward_pre_hook(self._pre(i)))
            self._handles.append(b.register_forward_hook(self._post(i)))

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class SeamReleasePipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched) for the cond->uncond seam release causal test."""

    def run_test(self, prompt, negative_prompt, height, width, num_frames,
                 guidance_scale, seed, mode, nvml) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {"seam": {}}

        generator = torch.Generator(device="cpu").manual_seed(seed)
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            device=torch.device("cpu"), dtype=self.text_dtype,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        self.pipe.scheduler.set_timesteps(5, device=device)
        timesteps = self.pipe.scheduler.timesteps
        latents = self.pipe.prepare_latents(
            batch_size=1, num_channels_latents=self.transformer.config.in_channels,
            height=height, width=width, num_frames=num_frames,
            dtype=torch.float32, device=device, generator=generator,
        )
        out["latent_shape"] = list(latents.shape)

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

        probe = BlockMemoryProbe(device)
        probe.attach(self.blocks)

        def _pass(pass_name, prompt_emb):
            probe.current_pass = pass_name
            latent_model_input = latents.to(dtype)
            rotary_emb = self.transformer.rope(latent_model_input)
            hidden_states = self.transformer.patch_embedding(latent_model_input)
            hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
            timestep = timesteps[0].expand(latent_model_input.shape[0])
            temb, timestep_proj, enc_hidden_states, _ = self.transformer.condition_embedder(
                timestep, prompt_emb, None
            )
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
            self._execute_dit_custom_forward(
                streamer=streamer, mode=mode, hidden_states=hidden_states,
                encoder_hidden_states=enc_hidden_states, timestep_proj=timestep_proj,
                rotary_emb=rotary_emb, num_steps=1, num_frames=num_frames,
            )
            shift, scale = (self.transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.transformer.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, pn_f, pn_h, pn_w, p_t, p_h, p_w, -1
            ).permute(0, 7, 1, 4, 2, 5, 3, 6)
            return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        def _snapshot(label):
            stats = torch.cuda.memory_stats(device)
            inst = nvml.sample_instant()
            snap = {
                "label": label,
                "nvml_mb": inst["nvml_used_mb"],
                "alloc_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
                "free_pool_mb": round((torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / (1024 * 1024), 2),
                "seg_current": stats.get("segment.all.current", 0),
                "seg_allocated": stats.get("segment.all.allocated", 0),
                "seg_freed": stats.get("segment.all.freed", 0),
                "ram_mb": round(get_process_ram_mb(), 2),
            }
            out["seam"][label] = snap
            print(f"   [SEAM {label:12s}] NVML={snap['nvml_mb']:7.1f} Alloc={snap['alloc_mb']:7.1f} "
                  f"Res={snap['reserved_mb']:7.1f} FreePool={snap['free_pool_mb']:7.1f} "
                  f"Segs(cur/alloc/freed)={snap['seg_current']}/{snap['seg_allocated']}/{snap['seg_freed']}")
            return snap

        t_step_start = time.perf_counter()
        # COND pass
        noise_pred_cond = _pass("cond", prompt_embeds)
        torch.cuda.synchronize(device)
        # SEAM: pre-release snapshot
        gc.collect()
        _snapshot("pre_release")
        # INTERVENTION (single variable): release pool at cond->uncond seam
        t_rel_start = time.perf_counter()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        rel_ms = (time.perf_counter() - t_rel_start) * 1000.0
        _snapshot("post_release")
        out["seam"]["release_ms"] = round(rel_ms, 2)
        # UNCOND pass
        noise_pred_uncond = _pass("uncond", negative_prompt_embeds)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = self.pipe.scheduler.step(noise_pred, timesteps[0], latents).prev_sample
        step_s = time.perf_counter() - t_step_start
        torch.cuda.synchronize(device)
        out["single_step_s"] = round(step_s, 2)

        try:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
        finally:
            self.transformer.blocks = transformer_blocks_orig
        probe.detach()
        out["records"] = probe.records
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7-D3 -- cond->uncond seam release (causal)")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default="logs/f7_d3_cond_uncond_seam_release_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D3_COND_UNCOND_SEAM_RELEASE_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D3: COND->UNCOND SEAM RELEASE (CAUSAL)")
    print("  (Single variable = empty_cache at cond->uncond seam. Runtime untouched)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: 1")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Latent: [1,16,{latent_f},{latent_h},{latent_w}]")
    print(f"  Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB")
    print(f"  D2c reference (no release): peak NVML {D2C_PEAK_NVML_MB} MB, peak Reserved {D2C_PEAK_RESERVED_MB} MB")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print("  cond pass -> [pre_release snapshot] -> empty_cache -> [post_release] -> uncond pass")
        print("  Per-block hooks record the uncond Reserved series (does it re-grow to ~5.7 GB?).")
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
        pipeline = SeamReleasePipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16,
        )
        print(f"\n>> Executing F7-D3 @ {args.width}x{args.height} ({args.frames}f) with cond->uncond release...")
        probe = pipeline.run_test(
            prompt=args.prompt, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            guidance_scale=args.guidance, seed=args.seed, mode=args.mode, nvml=nvml,
        )
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D3 ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = nvml.stop()

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT OPERATIVO] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB (proteccion OOM).")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()

    records: List[Dict[str, Any]] = (probe or {}).get("records", [])
    seam = (probe or {}).get("seam", {})

    # Per-block uncond series after release
    uncond_res = []
    for b in range(30):
        post = None
        for r in records:
            if r["pass"] == "uncond" and r["phase"] == "post" and r["block"] == b:
                post = r["reserved_mb"]
        pre = None
        for r in records:
            if r["pass"] == "uncond" and r["phase"] == "pre" and r["block"] == b:
                pre = r["reserved_mb"]
        if pre is not None:
            uncond_res.append((b, pre, post))
    res_after_uncond_b0 = uncond_res[0][2] if uncond_res else None

    # re-growth check: does reserved climb back toward D2C level?
    regrowth = None
    if res_after_uncond_b0 is not None:
        regrowth = res_after_uncond_b0 - D2C_RES_AFTER_COND_MB

    # result classification
    if peak_nvml <= HARD_GATE_VRAM_MB:
        result = "FAVORABLE (peak <= 4800)"
    elif peak_nvml < D2C_PEAK_NVML_MB - 200:
        result = "PARTIAL (peak down but > 4800)"
    else:
        result = "NEGATIVE (peak ~ unchanged)"

    print("\n" + "=" * 80)
    print("  PHASE F7-D3: SEAM RELEASE SCORECARD")
    print("=" * 80)
    print(f"  Peak NVML fisico        [OBSERVED] : {peak_nvml:7.1f} MB  (D2c no-release: {D2C_PEAK_NVML_MB:.1f})")
    print(f"  Peak Reserved           [OBSERVED] : {peak_reserved:7.1f} MB  (D2c: {D2C_PEAK_RESERVED_MB:.1f})")
    print(f"  Peak Allocated          [OBSERVED] : {peak_alloc:7.1f} MB  (F7-0: {F7_0_ALLOC_MB})")
    print(f"  Reserved after uncond b0 [DERIVED] : {res_after_uncond_b0} MB (regrowth {regrowth})")
    print(f"  Host RSS                [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total        [OBSERVED] : {t_total:7.2f} s")
    if "release_ms" in seam:
        print(f"  empty_cache duration     [OBSERVED] : {seam['release_ms']:.2f} ms")
    print(f"  Result                   [DERIVED]  : {result}")
    for lab in ("pre_release", "post_release"):
        if lab in seam:
            s = seam[lab]
            print(f"  {lab:12s}: NVML {s['nvml_mb']:.1f} Res {s['reserved_mb']:.1f} FreePool {s['free_pool_mb']:.1f}")
    print("  - uncond per-block post-reserved:")
    for b, pre, post in uncond_res:
        print(f"     blk{b:02d} pre {pre:6.0f} -> post {post if post else -1:.0f}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D3 (cond->uncond seam release, causal)",
        "evidence_class": "All measured/observed; derived deltas computed",
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps_executed": 1, "mode": args.mode, "seed": args.seed,
                     "guidance_scale": args.guidance},
        "single_step_s": (probe or {}).get("single_step_s"),
        "seam": seam,
        "per_block_records": records,
        "uncond_reserved_series": [{"block": b, "pre_mb": pre, "post_mb": post} for b, pre, post in uncond_res],
        "reserved_after_uncond_block0_mb": res_after_uncond_b0,
        "regrowth_mb": regrowth,
        "memory_peaks_mb": {"peak_nvml_used": round(peak_nvml, 1),
                            "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_torch_reserved": round(peak_reserved, 1),
                            "peak_process_ram": round(peak_ram, 1)},
        "comparison": {"d2c_no_release_peak_nvml_mb": D2C_PEAK_NVML_MB,
                       "d2c_no_release_peak_reserved_mb": D2C_PEAK_RESERVED_MB,
                       "f7_0_baseline_nvml_mb": F7_0_BASELINE_NVML_MB},
        "result_class": result,
        "verdict": {"execution_completed": error_occurred is None,
                    "hard_gate_4800mb_met": peak_nvml <= HARD_GATE_VRAM_MB,
                    "error_detail": error_occurred},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f">> Telemetry saved to: {args.output}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F7-D3 Cond->Uncond Seam Release -- Causal Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, 1 DiT step, mode `{args.mode}`  \n")
        w("**Variable:** controlled empty_cache at the cond->uncond seam (only).\n\n")
        w("---\n\n## 1. Seam snapshots (OBSERVED)\n\n")
        w("| Snapshot | NVML (MB) | Alloc (MB) | Reserved (MB) | FreePool (MB) | Seg(cur/alloc/freed) |\n")
        w("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for lab in ("pre_release", "post_release"):
            if lab in seam:
                s = seam[lab]
                w(f"| {lab} | {s['nvml_mb']} | {s['alloc_mb']} | {s['reserved_mb']} | {s['free_pool_mb']} | "
                  f"{s['seg_current']}/{s['seg_allocated']}/{s['seg_freed']} |\n")
        w(f"\nempty_cache duration: {seam.get('release_ms', 'n/a')} ms\n")
        w("\n## 2. Uncond per-block Reserved after release (DERIVED)\n\n")
        w("| Pass | Blk | PreRes (MB) | PostRes (MB) |\n")
        w("| :--- | :---: | :---: | :---: |\n")
        for b, pre, post in uncond_res:
            w(f"| uncond | {b:02d} | {pre:.0f} | {post if post else -1:.0f} |\n")
        w("\n## 3. Result\n\n")
        w(f"- Peak NVML: {peak_nvml:.1f} MB (D2c no-release: {D2C_PEAK_NVML_MB:.1f})\n")
        w(f"- Peak Reserved: {peak_reserved:.1f} MB (D2c: {D2C_PEAK_RESERVED_MB:.1f})\n")
        w(f"- Reserved after uncond block0: {res_after_uncond_b0} MB (regrowth vs cond end: {regrowth} MB)\n")
        w(f"- Classification: **{result}**\n")
        w("\n## 4. Governance\n\n")
        w("Causal experiment in an isolated runner. Does NOT authorize runtime modification.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
