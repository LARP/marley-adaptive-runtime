"""
f7_d2c_per_block_allocator_diagnostic.py
=========================================
Phase F7-D2c -- Per-block allocator growth diagnostic (OBSERVATIONAL).

Strictly observational: NO variable is changed. The purpose is to explain,
not modify, the Reserved/FreePool growth during the two 30-block passes
(cond, uncond) of a single DiT step at 1280x720/33f.

Answers the consultant's Escenario A/B/C question:
  How does Reserved/FreePool evolve per block across cond and uncond, and is
  the growth attributable to (A) correct reuse / necessary growth,
  (B) new avoidable reservations, or (C) fragmentation / poor reuse?

Method (non-invasive):
  * Subclass MarleyEndToEndPipeline (F7-D1 seam technique); runtime untouched.
  * Register torch forward PRE/POST hooks on the 30 DiT blocks FROM THE RUNNER
    (the blocks are nn.Module objects invoked by the frozen streamer, so the
    hooks fire without editing any runtime file).
  * Run a SINGLE cond+uncond denoising step via AdaptiveEngine (same as F7-D2b).
  * Each hook records torch allocated/reserved/free-pool + CUDA segment counters
    (segment.all.allocated/freed/current) around every block forward.

Constraints honored: no modification of MarleyEndToEndPipeline,
BudgetedAsyncStreamer, INT8BudgetedStreamer, AdaptiveEngine, VAE, attention,
frames, resolution, offload strategy, or allocator policy. Single variable: NONE.

Evidence tags: MEASURED / OBSERVED / DERIVED / HYPOTHESIS.
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
        out = {"nvml_used_mb": 0.0, "gpu_temp_c": None, "gpu_clock_mhz": None}
        if not self._available or self._handle is None:
            return out
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            out["nvml_used_mb"] = round(info.used / (1024 * 1024), 2)
            out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        return out

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self.peak_mb


class BlockMemoryProbe:
    """
    Holds the forward hooks that record per-block allocator state. Attached to
    the nn.Module blocks from the runner; NOT a modification of the runtime.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.current_pass = "cond"
        self.records: List[Dict[str, Any]] = []
        self._handles: List[Any] = []
        # weight MB per block (static params) computed at attach time
        self.block_weight_mb: List[float] = []

    def _stats(self) -> Tuple[float, float, float, int, int, int]:
        stats = torch.cuda.memory_stats(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        return (
            alloc,
            reserved,
            reserved - alloc,
            int(stats.get("segment.all.current", 0)),
            int(stats.get("segment.all.allocated", 0)),
            int(stats.get("segment.all.freed", 0)),
        )

    def _make_pre(self, block_idx: int):
        def hook(module, args):
            a, r, f, segc, sega, segf = self._stats()
            self.records.append({
                "block": block_idx, "pass": self.current_pass, "phase": "pre",
                "alloc_mb": round(a, 2), "reserved_mb": round(r, 2),
                "free_pool_mb": round(f, 2),
                "segment_current": segc, "segment_allocated": sega, "segment_freed": segf,
                "weight_mb": round(self.block_weight_mb[block_idx], 2),
            })
        return hook

    def _make_post(self, block_idx: int):
        def hook(module, args, output):
            a, r, f, segc, sega, segf = self._stats()
            self.records.append({
                "block": block_idx, "pass": self.current_pass, "phase": "post",
                "alloc_mb": round(a, 2), "reserved_mb": round(r, 2),
                "free_pool_mb": round(f, 2),
                "segment_current": segc, "segment_allocated": sega, "segment_freed": segf,
                "weight_mb": round(self.block_weight_mb[block_idx], 2),
            })
        return hook

    def attach(self, blocks: List[nn.Module]) -> None:
        for i, blk in enumerate(blocks):
            w_mb = sum(p.numel() * p.element_size() for p in blk.parameters()) / (1024 * 1024)
            self.block_weight_mb.append(w_mb)
            self._handles.append(blk.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(blk.register_forward_hook(self._make_post(i)))

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class PerBlockDiagnosticPipeline(MarleyEndToEndPipeline):
    """Subclass seam (runtime untouched) for the per-block allocator diagnostic."""

    def run_diagnostic(
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
    ) -> Dict[str, Any]:
        device = self.device
        dtype = self.dit_dtype
        out: Dict[str, Any] = {}

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

        # Attach per-block hooks (observational only).
        probe = BlockMemoryProbe(device)
        probe.attach(self.blocks)

        def _pass(cond_or_uncond: str, prompt_emb: torch.Tensor) -> torch.Tensor:
            probe.current_pass = cond_or_uncond
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

        t0 = time.perf_counter()
        probe.current_pass = "cond"
        noise_pred_cond = _pass("cond", prompt_embeds)
        noise_pred_uncond = _pass("uncond", negative_prompt_embeds)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = self.pipe.scheduler.step(noise_pred, timesteps[0], latents).prev_sample
        step_s = time.perf_counter() - t0
        torch.cuda.synchronize(device)
        out["single_step_s"] = round(step_s, 2)

        # cleanup: restore blocks / release streamer / detach hooks
        try:
            if hasattr(streamer, "restore_all_to_cpu"):
                streamer.restore_all_to_cpu()
            if hasattr(streamer, "release"):
                streamer.release()
        finally:
            self.transformer.blocks = transformer_blocks_orig
        probe.detach()
        out["records"] = probe.records
        out["block_weight_mb"] = probe.block_weight_mb
        return out


def _reserved_at(records: List[Dict[str, Any]], pass_name: str, phase: str, block: int) -> Optional[float]:
    for r in records:
        if r["pass"] == pass_name and r["phase"] == phase and r["block"] == block:
            return r["reserved_mb"]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F7-D2c -- per-block allocator growth diagnostic")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default="logs/f7_d2c_per_block_allocator_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F7_D2C_PER_BLOCK_ALLOCATOR_REPORT_01.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_f = (args.frames - 1) // 4 + 1

    print("=" * 80)
    print("  MARLEY RUNTIME -- PHASE F7-D2c: PER-BLOCK ALLOCATOR GROWTH DIAGNOSTIC")
    print("  (Strictly OBSERVATIONAL. Variable = NONE. Results tagged MEASURED/OBSERVED/DERIVED)")
    print("=" * 80)
    print(f"  Resolution: {args.width}x{args.height} | Frames: {args.frames} | Steps: 1 (single)")
    print(f"  Mode: {args.mode} | Seed: {args.seed} | Latent: [1,16,{latent_f},{latent_h},{latent_w}]")
    print(f"  Safe Abort: > {SAFE_ABORT_NVML_MB:.0f} MB (OOM protection)")

    if args.dry_run:
        print("\n[DRY-RUN] Parameter/environment verification. NO inference.")
        print("  Per-block forward PRE/POST hooks on the 30 DiT blocks (from runner, runtime untouched).")
        print("  Single cond+uncond step via AdaptiveEngine. Records: alloc/reserved/free_pool/segments.")
        print("[DRY-RUN] Validated. Awaiting execution authorization.")
        return

    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = None
    probe: Optional[Dict[str, Any]] = None
    error_occurred: Optional[str] = None
    t_start = time.perf_counter()
    try:
        pipeline = PerBlockDiagnosticPipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16,
        )
        print(f"\n>> Executing F7-D2c single cond+uncond step @ {args.width}x{args.height} ({args.frames}f)...")
        probe = pipeline.run_diagnostic(
            prompt=args.prompt, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            guidance_scale=args.guidance, seed=args.seed, mode=args.mode,
            nvml=nvml,
        )
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F7-D2c ABORT] Exception during execution: {error_occurred}")
    finally:
        t_total = time.perf_counter() - t_start
        peak_nvml = nvml.stop()

    if nvml.safe_abort_triggered:
        print(f"\n[SAFE ABORT OPERATIVO] NVML fisico supero {SAFE_ABORT_NVML_MB:.0f} MB (proteccion OOM).")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()

    records: List[Dict[str, Any]] = (probe or {}).get("records", [])

    # ---- Per-block reserved series (DERIVED from records) ------------------
    def _series(pass_name: str) -> List[Tuple[int, float, float, float, int]]:
        # (block, pre_reserved, post_reserved, post_free_pool, post_seg_current)
        seq = []
        for b in range(30):
            pre = _reserved_at(records, pass_name, "pre", b)
            post = _reserved_at(records, pass_name, "post", b)
            fp = None
            for r in records:
                if r["pass"] == pass_name and r["phase"] == "post" and r["block"] == b:
                    fp = r["free_pool_mb"]
            if pre is not None and post is not None and fp is not None:
                seq.append((b, pre, post, fp, int(post - pre)))
        return seq

    cond_seq = _series("cond")
    uncond_seq = _series("uncond")

    res_cond_start = cond_seq[0][1] if cond_seq else 0.0
    res_cond_end = cond_seq[-1][2] if cond_seq else 0.0
    res_uncond_end = uncond_seq[-1][2] if uncond_seq else 0.0
    growth_cond = res_cond_end - res_cond_start
    growth_uncond = res_uncond_end - res_cond_end  # additional during uncond
    # reutilization: if uncond starts by NOT growing beyond cond end, reuse is happening
    # count uncond blocks where post_reserved increased vs cond_end plateau
    reuse_metric = 0.0
    if uncond_seq and cond_seq:
        reuse_metric = growth_uncond  # ideally near 0 if fully reusing cond segments

    # segment allocated delta across whole run (new segments ever allocated)
    seg_allocated_total = 0
    seg_freed_total = 0
    if records:
        seg_allocated_total = records[-1]["segment_allocated"]
        seg_freed_total = records[-1]["segment_freed"]

    print("\n" + "=" * 80)
    print("  PHASE F7-D2c: PER-BLOCK GROWTH SUMMARY")
    print("=" * 80)
    print(f"  Peak NVML fisico        [OBSERVED] : {peak_nvml:7.1f} MB")
    print(f"  Peak Allocated          [OBSERVED] : {peak_alloc:7.1f} MB (F7-0: {F7_0_ALLOC_MB})")
    print(f"  Peak Reserved           [OBSERVED] : {peak_reserved:7.1f} MB (F7-0: {F7_0_RESERVED_MB})")
    print(f"  Host RSS                [OBSERVED] : {peak_ram:7.1f} MB")
    print(f"  Wall-clock total        [OBSERVED] : {t_total:7.2f} s ({t_total/60:.2f} min)")
    print(f"  Reserved start cond     [DERIVED]  : {res_cond_start:7.1f} MB")
    print(f"  Reserved end cond       [DERIVED]  : {res_cond_end:7.1f} MB")
    print(f"  Reserved end uncond     [DERIVED]  : {res_uncond_end:7.1f} MB")
    print(f"  Growth during cond      [DERIVED]  : {growth_cond:+7.1f} MB (30 blocks)")
    print(f"  Growth during uncond    [DERIVED]  : {growth_uncond:+7.1f} MB (over cond plateau)")
    print(f"  Segments ever allocated [DERIVED]  : {seg_allocated_total}")
    print(f"  Segments ever freed     [DERIVED]  : {seg_freed_total}")
    print("  - per-block (cond):")
    for b, pre, post, fp, d in cond_seq:
        print(f"     blk{b:02d} res {pre:6.0f}->{post:6.0f} d{d:+5.0f} freePool {fp:6.0f}")
    print("  - per-block (uncond):")
    for b, pre, post, fp, d in uncond_seq:
        print(f"     blk{b:02d} res {pre:6.0f}->{post:6.0f} d{d:+5.0f} freePool {fp:6.0f}")
    print("=" * 80 + "\n")

    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F7-D2c (per-block allocator growth diagnostic, observational)",
        "evidence_class": "All measured/observed; derived deltas computed",
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps_executed": 1, "mode": args.mode, "seed": args.seed,
                     "guidance_scale": args.guidance},
        "single_step_s": (probe or {}).get("single_step_s"),
        "per_block_records": records,
        "reserved_series": {
            "cond_start_mb": round(res_cond_start, 1), "cond_end_mb": round(res_cond_end, 1),
            "uncond_end_mb": round(res_uncond_end, 1),
            "growth_cond_mb": round(growth_cond, 1), "growth_uncond_over_cond_mb": round(growth_uncond, 1),
            "uncond_reuse_signal_mb": round(reuse_metric, 1),
        },
        "segment_totals": {"allocated": seg_allocated_total, "freed": seg_freed_total},
        "memory_peaks_mb": {"peak_nvml_used": round(peak_nvml, 1),
                            "peak_torch_alloc": round(peak_alloc, 1),
                            "peak_torch_reserved": round(peak_reserved, 1),
                            "peak_process_ram": round(peak_ram, 1)},
        "verdict": {"execution_completed": error_occurred is None,
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
        w("# F7-D2c Per-Block Allocator Growth Diagnostic -- Report\n\n")
        w(f"**Date:** {datetime.datetime.now().isoformat()}  \n")
        w(f"**Workload:** {args.width}x{args.height} @ {args.frames} frames, 1 DiT step, mode `{args.mode}`  \n")
        w("**Variable changed:** NONE (observational).\n\n")
        w("---\n\n## 1. Reserved series per block (DERIVED)\n\n")
        w("| Pass | Reserved start (MB) | Reserved end (MB) | Growth (MB) |\n")
        w("| :--- | :---: | :---: | :---: |\n")
        w(f"| cond (blocks 0..29) | {res_cond_start:.1f} | {res_cond_end:.1f} | {growth_cond:+.1f} |\n")
        w(f"| uncond (blocks 0..29) | {res_cond_end:.1f} | {res_uncond_end:.1f} | {growth_uncond:+.1f} |\n")
        w("\nDetailed per-block (pre/post reserved, post free-pool):\n\n")
        w("| Pass | Blk | PreRes (MB) | PostRes (MB) | FreePool (MB) |\n")
        w("| :--- | :---: | :---: | :---: | :---: |\n")
        for b, pre, post, fp, _ in cond_seq:
            w(f"| cond | {b:02d} | {pre:.0f} | {post:.0f} | {fp:.0f} |\n")
        for b, pre, post, fp, _ in uncond_seq:
            w(f"| uncond | {b:02d} | {pre:.0f} | {post:.0f} | {fp:.0f} |\n")
        w("\n## 2. Reutilization / growth signal\n\n")
        w(f"- Growth during cond (30 blocks): **{growth_cond:+.1f} MB**\n")
        w(f"- Additional growth during uncond over cond plateau: **{growth_uncond:+.1f} MB**\n")
        w(f"- Segments ever allocated / freed: **{seg_allocated_total} / {seg_freed_total}**\n")
        w("\nInterpretive guidance (Escenarios A/B/C): if uncond reuses cond segments, its additional "
          "growth should be near 0 and Reserved should plateau; a large positive uncond growth indicates "
          "the second pass acquires new reserved segments rather than reusing the freed cond ones.\n")
        w("\n## 3. Memory peaks (OBSERVED)\n\n")
        w(f"- Peak physical NVML: {peak_nvml:.1f} MB\n")
        w(f"- Peak Allocated: {peak_alloc:.1f} MB\n")
        w(f"- Peak Reserved: {peak_reserved:.1f} MB\n")
        w("\n## 4. Governance\n\n")
        w("Observational diagnostic. Does NOT authorize F7-D3 or any runtime modification.\n")
    print(f">> Report saved to: {args.report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
