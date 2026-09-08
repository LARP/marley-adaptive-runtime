"""
f5_vae_probe_33f.py
===================
Phase F5 — Canonical 33-frame VAE Decode Probe (F5-A)

Evaluates the existing Wan2.1 VAE (bfloat16 + spatial-temporal tiling 256x256)
on the frozen Canonical F5 Workload (docs/F5_ROADMAP_PLAN_01.md):
  - Resolution: 832x480 (480p)
  - Frames: 33 frames exact
  - Latent shape: (1, 16, 9, 60, 104)  [(33-1)/4 + 1 = 9]
  - Precision: torch.bfloat16
  - Tiling: native enable_tiling() with 256x256 spatial tiles & causal caching

Telemetry layers:
  1. Continuous NVML hardware physical VRAM polling at 50 ms (Hard Gate <= 4,800 MB, Target <= 4,000 MB)
  2. PyTorch memory allocated & reserved (max stats)
  3. Host process RAM (RSS via psutil) & pinned memory tracking
  4. Precise VAE decode wall-clock time (Engineering target <= 150 s)
  5. Tensor integrity check: NaN/Inf scan and output dimension check

Decision Logic (Decision-First Protocol):
  If Peak VRAM <= 4,800 MB and Decode <= 150 s and 0 NaNs:
      -> RETIRE F5-C (Resolved by Existing Tiled VAE) -> Advance to F6
  Else:
      -> Proceed to F5-B (Temporal Characterization 17f -> 25f -> 33f)

Usage:
  .venv\\Scripts\\python.exe f5_vae_probe_33f.py
  .venv\\Scripts\\python.exe f5_vae_probe_33f.py --output logs/f5_vae_probe_33f.json
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
from diffusers import AutoencoderKLWan

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HARD_GATE_VRAM_MB = 4800.0
ENGINEERING_TARGET_VRAM_MB = 4000.0
ENGINEERING_TARGET_LATENCY_S = 150.0

CANONICAL_BATCH = 1
CANONICAL_CHANNELS = 16
CANONICAL_LATENT_FRAMES = 9     # (33 - 1) // 4 + 1
CANONICAL_LATENT_HEIGHT = 60    # 480 // 8
CANONICAL_LATENT_WIDTH = 104    # 832 // 8
CANONICAL_OUT_FRAMES = 33
CANONICAL_OUT_HEIGHT = 480
CANONICAL_OUT_WIDTH = 832


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
    """Returns current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_probe(device_str: str = "cuda:0") -> Dict:
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    print("=" * 75)
    print("  MARLEY RUNTIME — PHASE F5-A: CANONICAL 33-FRAME VAE DECODE PROBE")
    print("=" * 75)
    print(f"  Model ID:           {MODEL_ID} (subfolder='vae')")
    print(f"  Precision:          torch.bfloat16")
    print(f"  Target Resolution:  {CANONICAL_OUT_WIDTH}x{CANONICAL_OUT_HEIGHT} @ {CANONICAL_OUT_FRAMES} frames")
    print(f"  Latent Tensor:      ({CANONICAL_BATCH}, {CANONICAL_CHANNELS}, {CANONICAL_LATENT_FRAMES}, {CANONICAL_LATENT_HEIGHT}, {CANONICAL_LATENT_WIDTH})")
    print(f"  Spatial Tiling:     Enabled (256x256 tiles, causal temporal caching)")
    print(f"  Hard Gate VRAM:     <= {HARD_GATE_VRAM_MB:.1f} MB (Physical NVML)")
    print(f"  Target VRAM:        <= {ENGINEERING_TARGET_VRAM_MB:.1f} MB")
    print(f"  Target Decode Time: <= {ENGINEERING_TARGET_LATENCY_S:.1f} s")
    print("=" * 75)

    sampler = NVMLSampler(device_index=device.index or 0, interval_ms=50.0)
    sampler.start()

    init_ram_mb = get_process_ram_mb()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Step 1: Load VAE
    print("\n>> [1/4] Loading AutoencoderKLWan in bfloat16...")
    t_load_start = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()
    vae.enable_tiling()
    torch.cuda.synchronize(device)
    t_load_elapsed = time.perf_counter() - t_load_start
    print(f"   [OK] VAE loaded & tiling enabled in {t_load_elapsed:.2f}s")
    post_load_nvml = sampler.peak_mb
    post_load_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    print(f"   VRAM post-load: NVML={post_load_nvml:.1f} MB | PyTorch Alloc={post_load_alloc:.1f} MB")

    # Step 2: Prepare Canonical Latent
    print("\n>> [2/4] Allocating Canonical Latent Tensor...")
    torch.manual_seed(42)
    latents = torch.randn(
        CANONICAL_BATCH,
        CANONICAL_CHANNELS,
        CANONICAL_LATENT_FRAMES,
        CANONICAL_LATENT_HEIGHT,
        CANONICAL_LATENT_WIDTH,
        dtype=torch.bfloat16,
        device=device,
    )
    latent_vram_mb = (latents.nelement() * 2) / (1024 * 1024)
    print(f"   Latent shape: {list(latents.shape)} ({latent_vram_mb:.2f} MB)")

    # Step 3: Run Decode
    print("\n>> [3/4] Executing VAE tiled decode (Canonical 33 frames)...")
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t_decode_start = time.perf_counter()
    try:
        decoded_tuple = vae.decode(latents, return_dict=False)
        decoded = decoded_tuple[0]
        torch.cuda.synchronize(device)
        decode_duration_s = time.perf_counter() - t_decode_start
        decode_success = True
        error_msg = None
    except Exception as exc:
        torch.cuda.synchronize(device)
        decode_duration_s = time.perf_counter() - t_decode_start
        decode_success = False
        error_msg = str(exc)
        decoded = None
        print(f"   [FAIL] Decode raised exception: {exc}")

    peak_nvml_mb = sampler.stop()
    peak_torch_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_torch_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    peak_ram_mb = get_process_ram_mb()

    # Step 4: Integrity & Output Verification
    print("\n>> [4/4] Verifying Output Tensor Integrity...")
    nan_count = 0
    inf_count = 0
    out_shape = None
    shape_ok = False

    if decode_success and decoded is not None:
        out_shape = list(decoded.shape)
        # Expected: [batch, channels(3), frames(33), height(480), width(832)]
        expected_shape = [CANONICAL_BATCH, 3, CANONICAL_OUT_FRAMES, CANONICAL_OUT_HEIGHT, CANONICAL_OUT_WIDTH]
        shape_ok = (out_shape == expected_shape)
        nan_count = int(torch.isnan(decoded).sum().item())
        inf_count = int(torch.isinf(decoded).sum().item())
        print(f"   Output Shape:   {out_shape} (Expected: {expected_shape}) -> {'MATCH' if shape_ok else 'MISMATCH'}")
        print(f"   NaNs / Infs:    {nan_count} / {inf_count}")
        print(f"   Output Range:   min={decoded.min().item():.3f}, max={decoded.max().item():.3f}, mean={decoded.mean().item():.3f}")
    else:
        print(f"   Decode failed: {error_msg}")

    # Decision Logic
    hard_gate_pass = (peak_nvml_mb <= HARD_GATE_VRAM_MB) and (nan_count == 0) and (inf_count == 0) and decode_success and shape_ok
    target_vram_met = peak_nvml_mb <= ENGINEERING_TARGET_VRAM_MB
    latency_target_met = decode_duration_s <= ENGINEERING_TARGET_LATENCY_S

    if hard_gate_pass and latency_target_met:
        recommendation = "RETIRE F5-C (Resolved by Existing Tiled VAE) -> Advance to F6"
        status_code = "RETIRE_F5"
    elif hard_gate_pass and not latency_target_met:
        recommendation = "PROCEED TO F5-B (VRAM fits but latency exceeds 150s target -> Characterization 17f->25f->33f)"
        status_code = "PROCEED_F5_B"
    else:
        recommendation = "PROCEED TO F5-C (Hard Gate Trip or OOM -> Temporal Chunker required)"
        status_code = "PROCEED_F5_C"

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(device),
        "total_vram_mb": torch.cuda.get_device_properties(device).total_memory / (1024 * 1024),
        "workload": {
            "model_id": MODEL_ID,
            "dtype": "torch.bfloat16",
            "resolution": f"{CANONICAL_OUT_WIDTH}x{CANONICAL_OUT_HEIGHT}",
            "frames": CANONICAL_OUT_FRAMES,
            "latent_shape": [CANONICAL_BATCH, CANONICAL_CHANNELS, CANONICAL_LATENT_FRAMES, CANONICAL_LATENT_HEIGHT, CANONICAL_LATENT_WIDTH],
            "spatial_tiling": "256x256",
            "temporal_caching": "causal",
        },
        "metrics": {
            "decode_time_s": round(decode_duration_s, 2),
            "peak_nvml_used_mb": round(peak_nvml_mb, 1),
            "peak_torch_allocated_mb": round(peak_torch_alloc_mb, 1),
            "peak_torch_reserved_mb": round(peak_torch_reserved_mb, 1),
            "peak_process_ram_mb": round(peak_ram_mb, 1),
            "init_process_ram_mb": round(init_ram_mb, 1),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "output_shape": out_shape,
            "decode_success": decode_success,
            "error": error_msg,
        },
        "gates": {
            "hard_gate_vram_4800mb": "PASS" if peak_nvml_mb <= HARD_GATE_VRAM_MB else "FAIL",
            "engineering_target_vram_4000mb": "MET" if target_vram_met else "NOT_MET",
            "latency_target_150s": "MET" if latency_target_met else "NOT_MET",
            "integrity_gate_0_nans": "PASS" if (nan_count == 0 and inf_count == 0 and shape_ok) else "FAIL",
            "overall_hard_gate": "PASS" if hard_gate_pass else "FAIL",
        },
        "decision": {
            "status_code": status_code,
            "recommendation": recommendation,
        },
    }

    print("\n" + "=" * 75)
    print("  PHASE F5-A SCORECARD & DECISION")
    print("=" * 75)
    print(f"  VAE Decode Duration:       {decode_duration_s:.2f} s  (Target <= {ENGINEERING_TARGET_LATENCY_S:.0f} s: {'MET' if latency_target_met else 'EXCEEDED'})")
    print(f"  Hardware Peak VRAM (NVML): {peak_nvml_mb:.1f} MB  (Gate <= {HARD_GATE_VRAM_MB:.0f} MB: {'PASS' if peak_nvml_mb <= HARD_GATE_VRAM_MB else 'FAIL'})")
    print(f"  PyTorch Allocated Peak:    {peak_torch_alloc_mb:.1f} MB")
    print(f"  PyTorch Reserved Peak:     {peak_torch_reserved_mb:.1f} MB")
    print(f"  Peak Host Process RAM:     {peak_ram_mb:.1f} MB")
    print(f"  Integrity (NaNs / Infs):   {nan_count} / {inf_count}")
    print("-" * 75)
    print(f"  DECISION VERDICT:          {status_code}")
    print(f"  RECOMMENDED ACTION:        {recommendation}")
    print("=" * 75 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase F5-A: Canonical 33-frame VAE Decode Probe")
    parser.add_argument("--output", type=str, default="logs/f5_vae_probe_33f.json", help="Path to telemetry JSON")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    results = run_probe(device_str=args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Telemetry written to {out_path}")


if __name__ == "__main__":
    main()
