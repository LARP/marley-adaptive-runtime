"""
Marley Runtime - Phase F0 - Reproducible Baseline (REAL MODEL)
================================================================
Executes Wan2.1-T2V-1.3B via Diffusers WanPipeline with standard
sequential CPU offload at 480p / 16 frames / FP16.

Telemetry layers (per ROADMAP §2):
  1. torch.cuda.memory_allocated()   — live PyTorch tensor footprint
  2. torch.cuda.memory_reserved()    — PyTorch caching allocator pool
  3. NVML physical residency          — hardware ground-truth (4.8 GB gate)

Usage:
    .venv\\Scripts\\python.exe f0_baseline_real.py
    .venv\\Scripts\\python.exe f0_baseline_real.py --res 480p --frames 16 --no-offload
    .venv\\Scripts\\python.exe f0_baseline_real.py --res 480p --frames 16 --offload cpu
    .venv\\Scripts\\python.exe f0_baseline_real.py --res 480p --frames 16 --offload model_cpu
"""

import argparse
import datetime
import io
import json
import os
import sys
import time
import traceback

# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

# ---------------------------------------------------------------------------
# NVML setup (nvidia-ml-py, not deprecated pynvml)
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception as _e:
    _NVML_OK = False
    print(f"[WARN] NVML unavailable: {_e}. Physical residency will not be reported.")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_file_path: str = ""


def _init_log(tag: str) -> str:
    global _log_file_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = os.path.join(LOG_DIR, f"f0_real_{tag}_{ts}.log")
    return _log_file_path


def log(msg: str, console: bool = True):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    entry = f"[{ts}] {msg}"
    if console:
        print(entry)
    if _log_file_path:
        with open(_log_file_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")


import threading

# ---------------------------------------------------------------------------
# Continuous NVML Peak Tracker (samples every 50ms in background thread)
# ---------------------------------------------------------------------------
class NVMLPeakTracker:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.peak_mb = 0.0
        self.thread = None

    def _loop(self):
        while not self.stop_event.is_set():
            if _NVML_OK:
                info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                used_mb = info.used / 2**20
                if used_mb > self.peak_mb:
                    self.peak_mb = used_mb
            time.sleep(self.interval_s)

    def start(self):
        if _NVML_OK:
            info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            self.peak_mb = info.used / 2**20
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_mb


# ---------------------------------------------------------------------------
# VRAM telemetry — all 3 layers from ROADMAP §2
# ---------------------------------------------------------------------------
def vram_snapshot(label: str) -> dict:
    snap = {"label": label, "ts": time.monotonic()}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        snap["torch_allocated_mb"]    = torch.cuda.memory_allocated(0) / 2**20
        snap["torch_reserved_mb"]     = torch.cuda.memory_reserved(0)  / 2**20
        snap["torch_max_allocated_mb"]= torch.cuda.max_memory_allocated(0) / 2**20
        snap["cuda_free_mb"]          = free_b  / 2**20
        snap["cuda_total_mb"]         = total_b / 2**20
    if _NVML_OK:
        info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        snap["nvml_used_mb"]  = info.used  / 2**20
        snap["nvml_free_mb"]  = info.free  / 2**20
        snap["nvml_total_mb"] = info.total / 2**20
    return snap


def log_snap(snap: dict):
    lines = [f"  ── VRAM @ {snap['label']} ──"]
    if "torch_allocated_mb" in snap:
        lines.append(
            f"     PyTorch allocated:  {snap['torch_allocated_mb']:7.1f} MB"
            f"  |  reserved: {snap['torch_reserved_mb']:7.1f} MB"
            f"  |  peak-alloc: {snap['torch_max_allocated_mb']:7.1f} MB"
        )
    if "nvml_used_mb" in snap:
        pct = snap["nvml_used_mb"] / snap["nvml_total_mb"] * 100
        gate = "[WARN] ABOVE 4.8 GB GATE" if snap["nvml_used_mb"] > 4915.2 else "[OK] within gate"
        lines.append(
            f"     NVML physical used: {snap['nvml_used_mb']:7.1f} MB"
            f"  |  free: {snap['nvml_free_mb']:7.1f} MB"
            f"  |  {pct:.1f}%  {gate}"
        )
    for l in lines:
        log(l)
    return snap


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
RES_MAP = {
    "480p":  (480, 832),   # height, width — Wan2.1 standard 480p (both div by 16: 30x52)
    "720p":  (720, 1280),  # 720/16=45, 1280/16=80 ✓
    "360p":  (360, 640),   # 360/16=22.5 — avoid, use 480p
}


# ---------------------------------------------------------------------------
# Phase F0 — Real pipeline run
# ---------------------------------------------------------------------------
def run_f0_baseline(
    res: str,
    frames: int,
    dtype_str: str,
    offload: str,
    vae_dtype_str: str = "fp32",
    vae_tiling: bool = False,
):
    """
    Executes Wan2.1-T2V-1.3B via WanPipeline with the requested offload strategy.

    offload values:
        "none"       — no CPU offload (all weights on GPU)
        "cpu"        — enable_sequential_cpu_offload()
        "model_cpu"  — enable_model_cpu_offload()
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    height, width = RES_MAP[res]
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]
    vae_dtype = dtype_map[vae_dtype_str]

    log("=" * 72)
    log("  MARLEY RUNTIME - PHASE F0 - REPRODUCIBLE BASELINE (REAL MODEL)")
    log("=" * 72)
    log(f"  Model:      {MODEL_ID}")
    log(f"  Res:        {res} ({width}x{height})")
    log(f"  Frames:     {frames}")
    log(f"  Dtype:      {dtype_str} ({dtype})")
    log(f"  VAE Dtype:  {vae_dtype_str} ({vae_dtype})")
    log(f"  VAE Tiling: {vae_tiling}")
    log(f"  Offload:    {offload}")
    log(f"  Log:        {_log_file_path}")
    log("=" * 72)

    torch.cuda.reset_peak_memory_stats()
    snap_pre = log_snap(vram_snapshot("pre-load"))

    # ── 1. Load VAE separately with requested precision ─────────────────────
    log(f">> Step 1: Loading VAE ({vae_dtype_str})...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=vae_dtype
    )
    if vae_tiling:
        log("   [VAE TILING] Enabling pipe.vae.enable_tiling()...")
        vae.enable_tiling()
    log(f"   VAE loaded in {time.time()-t0:.1f}s")
    log_snap(vram_snapshot("post-vae-load"))

    # ── 2. Load full pipeline ────────────────────────────────────────────────
    log(">> Step 2: Loading WanPipeline...")
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
    )
    log(f"   Pipeline loaded in {time.time()-t0:.1f}s")
    log_snap(vram_snapshot("post-pipeline-load"))

    # ── 3. Apply offload strategy ────────────────────────────────────────────
    if offload == "cpu":
        log(">> Step 3: Applying enable_sequential_cpu_offload()...")
        pipe.enable_sequential_cpu_offload()
    elif offload == "model_cpu":
        log(">> Step 3: Applying enable_model_cpu_offload()...")
        pipe.enable_model_cpu_offload()
    else:
        log(">> Step 3: No offload — moving pipeline to CUDA:0...")
        pipe = pipe.to("cuda")

    if vae_tiling:
        pipe.vae.enable_tiling()

    log_snap(vram_snapshot("post-offload-setup"))

    # ── 4. Inference ─────────────────────────────────────────────────────────
    PROMPT = (
        "A golden retriever dog runs joyfully across a sunlit meadow, "
        "cinematic lighting, shallow depth of field, 4K."
    )
    NEG_PROMPT = "blurry, low quality, watermark, deformed"

    log(f">> Step 4: Running inference — {frames} frames @ {res}...")
    log(f"   Prompt: {PROMPT[:80]}...")
    t_inf_start = time.time()

    tracker = NVMLPeakTracker(interval_s=0.05)
    tracker.start()

    try:
        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=5.0,
            num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(42),
        )
        t_inf_elapsed = time.time() - t_inf_start
        peak_nvml_real_mb = tracker.stop()
        log(f"   [OK] Inference completed in {t_inf_elapsed:.1f}s")
        snap_post = log_snap(vram_snapshot("post-inference"))
        peak_nvml = max(peak_nvml_real_mb, snap_post.get("nvml_used_mb", 0))
        log(f"   [PEAK NVML DURING INFERENCE]: {peak_nvml:.1f} MB")

        # ── 5. Save output ───────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tiling_tag = "_tiled" if vae_tiling else ""
        out_path = os.path.join(LOG_DIR, f"f0_{res}_{frames}f_{dtype_str}_vae_{vae_dtype_str}{tiling_tag}_{offload}_{ts}.mp4")
        export_to_video(output.frames[0], out_path, fps=16)
        log(f"   Video saved → {out_path}")

        # ── 6. Final summary ─────────────────────────────────────────────────
        summary = {
            "phase": "F0",
            "model": MODEL_ID,
            "res": res,
            "resolution_px": f"{width}x{height}",
            "frames": frames,
            "dtype": dtype_str,
            "vae_dtype": vae_dtype_str,
            "vae_tiling": vae_tiling,
            "offload": offload,
            "inference_time_s": round(t_inf_elapsed, 2),
            "peak_torch_allocated_mb": round(snap_post.get("torch_max_allocated_mb", 0), 1),
            "peak_nvml_used_mb": round(peak_nvml, 1),
            "vram_gate_4800mb_ok": peak_nvml <= 4915.2,
            "output_video": out_path,
            "status": "SUCCESS",
        }

        log("\n" + "=" * 72)
        log("  F0 BASELINE SUMMARY")
        log("=" * 72)
        for k, v in summary.items():
            log(f"  {k:<30} {v}")
        log("=" * 72)

        # Save JSON telemetry
        json_path = out_path.replace(".mp4", "_telemetry.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        log(f"  Telemetry JSON → {json_path}")

        return summary

    except torch.cuda.OutOfMemoryError as oom:
        t_oom = time.time() - t_inf_start
        peak_nvml_real_mb = tracker.stop()
        snap_oom = log_snap(vram_snapshot("OOM-moment"))
        peak_nvml = max(peak_nvml_real_mb, snap_oom.get("nvml_used_mb", 0))
        tiling_tag = "_tiled" if vae_tiling else ""
        log("*" * 72)
        log(">>> GATE F0: CUDA OUT OF MEMORY <<<")
        log(f"    Elapsed before OOM: {t_oom:.2f}s")
        log(f"    Exception: {oom}")
        log(f"    Traceback:\n{traceback.format_exc()}", console=False)
        log("*" * 72)

        summary = {
            "phase": "F0",
            "model": MODEL_ID,
            "res": res,
            "frames": frames,
            "dtype": dtype_str,
            "vae_dtype": vae_dtype_str,
            "vae_tiling": vae_tiling,
            "offload": offload,
            "status": "OOM",
            "oom_elapsed_s": round(t_oom, 2),
            "oom_torch_allocated_mb": round(snap_oom.get("torch_allocated_mb", 0), 1),
            "oom_nvml_used_mb": round(peak_nvml, 1),
            "oom_exception": str(oom)[:300],
        }
        json_path = os.path.join(LOG_DIR, f"f0_OOM_{res}_{frames}f_{dtype_str}_vae_{vae_dtype_str}{tiling_tag}_{offload}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        log(f"  OOM telemetry → {json_path}")
        return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Marley Runtime Phase F0 — Reproducible Baseline (Real Wan2.1 Model)"
    )
    parser.add_argument("--res", default="480p", choices=list(RES_MAP), help="Target resolution")
    parser.add_argument("--frames", type=int, default=17, help="Number of video frames (default: 17; must satisfy (N-1)%%4==0)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision for DiT/Text encoder")
    parser.add_argument("--vae-dtype", default="fp32", choices=["fp32", "bf16", "fp16"], help="Precision for VAE (default: fp32)")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE spatial-temporal tiled decode")
    parser.add_argument(
        "--offload",
        default="cpu",
        choices=["none", "cpu", "model_cpu"],
        help=(
            "Offload strategy: "
            "none=all on GPU, "
            "cpu=sequential CPU offload (safest/lowest VRAM), "
            "model_cpu=model-level CPU offload"
        ),
    )
    args = parser.parse_args()

    tiling_str = "_tiled" if args.vae_tiling else ""
    tag = f"{args.res}_{args.frames}f_{args.dtype}_vae_{args.vae_dtype}{tiling_str}_{args.offload}"
    _init_log(tag)

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**20
    log(f"Device: {gpu_name} | Total VRAM: {total_vram:.0f} MB")

    result = run_f0_baseline(args.res, args.frames, args.dtype, args.offload, args.vae_dtype, args.vae_tiling)

    if result["status"] == "OOM":
        log("\n[WARN] Run ended with OOM. Review telemetry and try --offload cpu or --offload model_cpu.")
        sys.exit(0)
    else:
        log("\n[PASS] Phase F0 baseline PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
