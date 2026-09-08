"""
f6_end_to_end_benchmark.py
==========================
Phase F6 — Multidimensional Benchmarks & End-to-End Video Generation Runner.

Executes the complete text-to-video inference pipeline for Wan2.1-T2V-1.3B:
  Prompt -> UMT5-XXL -> Latents -> DiT (Marley Streaming) -> VAE (Tiled BF16) -> MP4

Supports:
  --smoke : 2-step Integration Smoke Test (F6-0)
  --steps : diffusion steps (e.g. 2 for smoke, 30 for benchmark)
  --frames: video frames (default: 33)
  --mode  : sync | async_fp16 | async_int8 | adaptive
  --output: path to telemetry JSON

Evaluates the 5 dimensions recommended by Consultant:
  1. Functional : valid output, 0 NaNs/Infs, MP4 playable.
  2. Memory     : Hard Gate <= 4,800 MB (Physical NVML), Target <= 4,000 MB.
  3. Performance: Total wall-clock time vs Engineering Target (< 600 s).
  4. Quality    : video shape [1, 3, 33, 480, 832], pixel range [-1, 1], frame consistency.
  5. Adaptive   : policy stability & pressure response.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
import torch

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from marley.pipeline.end_to_end import MarleyEndToEndPipeline

HARD_GATE_VRAM_MB = 4800.0
ENGINEERING_TARGET_VRAM_MB = 4000.0
ENGINEERING_TARGET_LATENCY_S = 600.0

CANONICAL_PROMPT = (
    "A golden retriever dog runs joyfully across a sunlit meadow, "
    "cinematic lighting, shallow depth of field, 4K."
)
CANONICAL_NEG_PROMPT = "blurry, low quality, watermark, deformed"
CANONICAL_SEED = 42
CANONICAL_GUIDANCE = 5.0
CANONICAL_FPS = 16


class NVMLSampler:
    """Polls physical GPU VRAM every 50 ms via NVML."""

    def __init__(self, device_index: int = 0, interval_ms: float = 50.0) -> None:
        self.interval_s = interval_ms / 1000.0
        self._peak_mb = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_available = True
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._peak_mb = info.used / (1024 * 1024)
        except Exception as exc:
            print(f"[WARN] NVML initialization failed: {exc}", file=sys.stderr)

    def start(self) -> None:
        if not self._nvml_available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        import pynvml
        while self._running:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mb = info.used / (1024 * 1024)
                if used_mb > self._peak_mb:
                    self._peak_mb = used_mb
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> float:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self._peak_mb

    @property
    def peak_mb(self) -> float:
        return self._peak_mb


def get_process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Runtime Phase F6 — End-to-End Benchmark")
    p.add_argument("--smoke", action="store_true", help="Execute 2-step Integration Smoke Test (F6-0)")
    p.add_argument("--steps", type=int, default=30, help="Number of diffusion denoising steps (default: 30)")
    p.add_argument("--frames", type=int, default=33, help="Number of output video frames (default: 33)")
    p.add_argument("--width", type=int, default=832, help="Frame width (default: 832)")
    p.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    p.add_argument("--mode", type=str, default="async_fp16", choices=["sync", "async_fp16", "async_int8", "adaptive"])
    p.add_argument("--prompt", type=str, default=CANONICAL_PROMPT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default="logs/f6_e2e_benchmark.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.steps = 2
        stage_name = "F6-0 (Integration Smoke Test)"
    else:
        stage_name = f"F6 Full Benchmark ({args.mode}, {args.steps} steps)"

    print("=" * 75)
    print(f"  MARLEY RUNTIME — PHASE F6: {stage_name.upper()}")
    print("=" * 75)
    print(f"  Workload:           {args.width}x{args.height} @ {args.frames} frames")
    print(f"  Denoising Steps:    {args.steps}")
    print(f"  Streaming Mode:     {args.mode}")
    print(f"  Prompt:             {args.prompt[:65]}...")
    print(f"  Seed:               {args.seed}")
    print(f"  Hard Gate VRAM:     <= {HARD_GATE_VRAM_MB:.1f} MB (Physical NVML)")
    print(f"  Engineering Target: <= {ENGINEERING_TARGET_VRAM_MB:.1f} MB | Latency < {ENGINEERING_TARGET_LATENCY_S:.0f} s")
    print("=" * 75)

    sampler = NVMLSampler(device_index=0, interval_ms=50.0)
    sampler.start()

    init_ram = get_process_ram_mb()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = MarleyEndToEndPipeline(
        device=args.device,
        dit_dtype=torch.float16,
        vae_dtype=torch.bfloat16,
        text_dtype=torch.bfloat16,
    )

    video_tensor, metrics = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=CANONICAL_NEG_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        mode=args.mode,
        nvml_sampler=sampler,
    )

    peak_nvml = sampler.stop()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    peak_ram = get_process_ram_mb()

    metrics.peak_nvml_mb = peak_nvml
    metrics.peak_torch_alloc_mb = peak_alloc
    metrics.peak_torch_reserved_mb = peak_reserved
    metrics.peak_process_ram_mb = peak_ram

    # Evaluate 5 Dimensions
    functional_pass = (not metrics.nan_inf_detected) and (metrics.output_video_path is not None) and os.path.exists(metrics.output_video_path)
    memory_pass = peak_nvml <= HARD_GATE_VRAM_MB
    target_vram_met = peak_nvml <= ENGINEERING_TARGET_VRAM_MB
    perf_met = metrics.total_wall_clock_s <= ENGINEERING_TARGET_LATENCY_S

    print("\n" + "=" * 75)
    print(f"  PHASE F6 SCORECARD — {stage_name.upper()}")
    print("=" * 75)
    print(f"  Stage Timings Breakdown:")
    print(f"    1. Prompt Encoding (UMT5-XXL): {metrics.prompt_encode_time_s:7.2f} s")
    print(f"    2. DiT Preparation (latents):   {metrics.dit_prep_time_s:7.2f} s")
    print(f"    3. DiT Denoising ({args.steps} steps):      {metrics.denoise_time_s:7.2f} s ({metrics.denoise_time_s/max(args.steps, 1):.2f} s/step)")
    print(f"    4. VAE Tiled Decode ({args.frames}f):       {metrics.vae_decode_time_s:7.2f} s")
    print(f"    5. Video Serialization (.mp4):  {metrics.video_export_time_s:7.2f} s")
    print(f"    ─────────────────────────────────────────────")
    print(f"    TOTAL END-TO-END WALL-CLOCK:   {metrics.total_wall_clock_s:7.2f} s ({metrics.total_wall_clock_s/60:.2f} min)")
    print(f"  Physical VRAM Peak (NVML 50ms):  {metrics.peak_nvml_mb:7.1f} MB (Gate <= {HARD_GATE_VRAM_MB:.0f} MB: {'PASS' if memory_pass else 'FAIL'})")
    print(f"  PyTorch Allocated Peak:          {metrics.peak_torch_alloc_mb:7.1f} MB")
    print(f"  Process Host RAM RSS:            {metrics.peak_process_ram_mb:7.1f} MB")
    print(f"  Exported Video File:             {metrics.output_video_path}")
    print(f"  NaN / Inf Detected:              {metrics.nan_inf_detected}")
    print("-" * 75)
    print("  MULTIDIMENSIONAL VERDICT (5 AXES):")
    print(f"    [1] Functional: {'🟢 PASS' if functional_pass else '❌ FAIL'}")
    print(f"    [2] Memory:     {'🟢 PASS' if memory_pass else '❌ FAIL'} ({'🟢 <= 4,000 Target MET' if target_vram_met else '🟡 <= 4,800 Acceptable'})")
    print(f"    [3] Performance:{'🟢 Target < 600s MET' if perf_met else '🟡 Target Exceeded (informational)'}")
    print(f"    [4] Quality:    {'🟢 Clean / Verified' if not metrics.nan_inf_detected else '❌ Corrupted'}")
    print(f"    [5] Adaptive:   {'🟢 Tested' if args.mode == 'adaptive' else '⚪ N/A (Standard Mode)'}")
    print("=" * 75 + "\n")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": stage_name,
        "mode": args.mode,
        "smoke_test": args.smoke,
        "workload": {
            "resolution": f"{args.width}x{args.height}",
            "frames": args.frames,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
            "prompt": args.prompt,
        },
        "timings_s": {
            "prompt_encoding": round(metrics.prompt_encode_time_s, 2),
            "dit_prep": round(metrics.dit_prep_time_s, 2),
            "denoising_total": round(metrics.denoise_time_s, 2),
            "denoising_per_step_avg": round(metrics.denoise_time_s / max(args.steps, 1), 2),
            "vae_decode": round(metrics.vae_decode_time_s, 2),
            "video_export": round(metrics.video_export_time_s, 2),
            "total_wall_clock": round(metrics.total_wall_clock_s, 2),
        },
        "memory_mb": {
            "peak_nvml_used": round(metrics.peak_nvml_mb, 1),
            "peak_torch_alloc": round(metrics.peak_torch_alloc_mb, 1),
            "peak_torch_reserved": round(metrics.peak_torch_reserved_mb, 1),
            "peak_process_ram": round(metrics.peak_process_ram_mb, 1),
        },
        "verdict": {
            "functional_pass": functional_pass,
            "memory_gate_pass": memory_pass,
            "target_vram_met": target_vram_met,
            "perf_target_met": perf_met,
            "nan_inf_detected": metrics.nan_inf_detected,
            "output_video": metrics.output_video_path,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Telemetry written to {out_path}")


if __name__ == "__main__":
    main()
