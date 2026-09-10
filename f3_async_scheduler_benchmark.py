"""
f3_async_scheduler_benchmark.py
================================
Phase F3 â€” Budgeted Asynchronous Scheduler Benchmark

Validates and certifies the BudgetedAsyncStreamer via two sequential stages:

  F3-A â€” Scheduler Validation (synthetic activations)
    Proves: scheduler is correct (no race conditions, no NaN/Inf, stable double-buffer).
    Does NOT prove wall-clock speedup on real Wan2.1.

  F3-B â€” Real Performance Certification (real Wan2.1 DiT blocks, representative load)
    Proves: Marley Async reduces EXTERNAL wall-clock denoising time vs. Marley Sync,
    reproducibly. Repeated A/B runs (order alternated, warmup discarded) report
    mean/min/max/std. This stage determines the OFFICIAL F3 PASS/FAIL verdict.

Architecture Notes
------------------
- enable_sequential_cpu_offload() is NOT used. Accelerate's AlignDevicesHook
  intercepts transfers on the default CUDA stream, making async scheduling
  invisible. Instead, BudgetedAsyncStreamer manages block residency directly.
- All 30 DiT blocks remain on CPU. Only the active block occupies GPU VRAM
  (within one of two pre-allocated double-buffer slots).
- Comparison: Marley Sync (Condition A) vs. Marley Async (Condition B).
  Both use identical Marley execution infrastructure â€” only the transfer
  strategy differs. This isolates the async/sync variable cleanly.

Kill Gates (abort conditions)
-----------------------------
  Peak VRAM > 4,800 MB            â†’ ABORT (physical limit of RTX 3050 6 GB)
  Effective overlap < 5%          â†’ F3-A FAIL; F3-B FAIL (measured empirically,
                                    min over all F3-B repetitions)
  Wall-clock speedup < 0%         â†’ F3-B FAIL (async slower than sync in any rep;
                                    speedup uses EXTERNAL wall-clock, not internal time)
  NaN/Inf in any output           â†’ ABORT (scheduler corrupts computation)

Methodology (measurement-integrity amendments)
-----------------------------------------------
  Enmienda 1 â€” Overlap/stall measured with CUDA events per block per step (real GPU
               stall accumulation), never a formula constant. Timeline exported as
               per_block_copy_s / per_block_stall_s / per_block_compute_s.
  Enmienda 2 â€” F3-B runs --f3b-reps repetitions alternating execution order A/B, B/A,
               preceded by a discarded warmup; reports mean/min/max/std.
  Enmienda 3 â€” Official speedup = external wall-clock (perf_counter bracketing each
               condition). Internal total_denoise_time_s retained as diagnostic.

Usage
-----
  python f3_async_scheduler_benchmark.py
  python f3_async_scheduler_benchmark.py --steps 5 --f3a-reps 3
  python f3_async_scheduler_benchmark.py --steps 5 --f3a-reps 3 --f3b-reps 5
  python f3_async_scheduler_benchmark.py --skip-f3a        # F3-B only
  python f3_async_scheduler_benchmark.py --skip-f3b        # F3-A only
  python f3_async_scheduler_benchmark.py --no-f3b-warmup   # no warmup pass

Output
------
  logs/f3_async_scheduler_benchmark.json   â€” machine-readable scorecard
  Console                                  â€” human-readable report

References
----------
  docs/F3_CALIBRATION_CHANGES_02.md           â€” full design rationale
  knowledge acquired through conversation with an AI agent  â€” consultant advisory letters
  marley/ops/async_stream.py               â€” BudgetedAsyncStreamer implementation
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# NVML sampler (50 ms polling, runs in background thread)
# ---------------------------------------------------------------------------

class NVMLSampler:
    """
    Background thread that polls GPU VRAM via pynvml at 50 ms intervals.
    Tracks peak physical VRAM used during a benchmark run.
    """

    def __init__(self, device_index: int = 0, interval_ms: float = 50.0) -> None:
        self.device_index = device_index
        self.interval_s = interval_ms / 1000.0
        self._peak_mb: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._nvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self._nvml_available = True
        except Exception as e:
            print(f"[NVML] pynvml unavailable: {e}. VRAM will be reported via torch.cuda.")

    def _sample_vram_mb(self) -> float:
        if self._nvml_available:
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return info.used / (1024 ** 2)
            except Exception:
                pass
        # Fallback to torch.cuda (reports reserved, not physical)
        return torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)

    def _loop(self) -> None:
        while self._running:
            mb = self._sample_vram_mb()
            if mb > self._peak_mb:
                self._peak_mb = mb
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._peak_mb = self._sample_vram_mb()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop sampler and return peak VRAM in MB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_mb

    @property
    def peak_mb(self) -> float:
        return self._peak_mb


# ---------------------------------------------------------------------------
# Kill Gate evaluator
# ---------------------------------------------------------------------------

KILL_GATE_VRAM_MB = 4800.0
KILL_GATE_OVERLAP_MIN_PCT = 5.0

def check_kill_gates(
    peak_vram_mb: float,
    nan_inf: bool,
    overlap_pct: Optional[float] = None,
    speedup_pct: Optional[float] = None,
    stage: str = "",
) -> bool:
    """
    Evaluate kill gates. Returns True if all gates pass, False if any fails.
    Prints a diagnostic for any failure.
    """
    passed = True

    if nan_inf:
        print(f"[KILL GATE] âŒ {stage}: NaN/Inf detected in outputs. ABORT.")
        passed = False

    if peak_vram_mb > KILL_GATE_VRAM_MB:
        print(f"[KILL GATE] âŒ {stage}: Peak VRAM {peak_vram_mb:.0f} MB > {KILL_GATE_VRAM_MB:.0f} MB limit. ABORT.")
        passed = False

    if overlap_pct is not None and overlap_pct < KILL_GATE_OVERLAP_MIN_PCT:
        print(f"[KILL GATE] âŒ {stage}: Effective overlap {overlap_pct:.1f}% < {KILL_GATE_OVERLAP_MIN_PCT:.0f}% minimum.")
        passed = False

    if speedup_pct is not None and speedup_pct < 0.0:
        print(f"[KILL GATE] âŒ {stage}: Wall-clock speedup {speedup_pct:.1f}% < 0% (Async is SLOWER than Sync).")
        passed = False

    return passed


def interpret_speedup(pct: float) -> str:
    if pct < 0:
        return "âŒ FAIL (regression)"
    elif pct < 5:
        return "âš ï¸ Marginal benefit"
    elif pct < 15:
        return "ðŸŸ¢ Success"
    elif pct < 30:
        return "ðŸŸ¢ Very good"
    else:
        return "ðŸŸ¢ Excellent"


def interpret_overlap(pct: float) -> str:
    if pct < 5:
        return "âŒ FAIL â€” scheduler not overlapping"
    elif pct < 30:
        return "âš ï¸ Minimal overlap"
    elif pct < 70:
        return "ðŸŸ¢ Good overlap"
    else:
        return "ðŸŸ¢ Excellent overlap"


def summarize(values: List[float]) -> Dict:
    """mean / min / max / std over repeated measurements (Enmienda 2)."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    mean = statistics.fmean(values) if hasattr(statistics, "fmean") else sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "min": min(values), "max": max(values), "std": std}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> Dict:
    """
    Main benchmark function. Returns a dict suitable for JSON serialization.
    """
    print("=" * 70)
    print("  Marley Runtime â€” Phase F3 Benchmark")
    print("  Budgeted Asynchronous Scheduler")
    print("=" * 70)

    # --- Device setup ---
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device found. F3 benchmark requires a CUDA GPU.")
        sys.exit(1)

    device = torch.device("cuda", 0)
    dtype = torch.float16
    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    print(f"\nDevice   : {device_name}")
    print(f"VRAM     : {vram_total_mb:.0f} MB total")
    print(f"dtype    : {dtype}")
    print(f"Steps    : {args.steps} (F3-A: {args.f3a_reps} repetitions)")
    print(f"Frames   : {args.frames}")
    print()

    # --- Load ONLY the Wan2.1 DiT transformer (no T5-XXL, no VAE, no scheduler) ---
    # Loading the full WanPipeline also loads T5-XXL (~11 GB RAM) into CUDA-accessible
    # memory (UMA on laptop GPU), causing OOM when the benchmark tries to allocate
    # activations on GPU. We only need the 30 DiT blocks for this benchmark.
    print("[Setup] Loading Wan2.1 transformer blocks (CPU)...")
    t_load_start = time.perf_counter()

    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
        transformer = WanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            subfolder="transformer",
            torch_dtype=dtype,
        )
        # Ensure all components stay on CPU â€” never touch GPU during load
        transformer = transformer.to("cpu")
        # Release any incidental CUDA allocations from the loading process
        torch.cuda.empty_cache()

        blocks = list(transformer.blocks)
        num_blocks = len(blocks)
        print(f"[Setup] Loaded {num_blocks} DiT blocks on CPU. ({time.perf_counter()-t_load_start:.1f}s)")
    except Exception as e:
        print(f"[ERROR] Failed to load Wan2.1 transformer: {e}")
        sys.exit(1)

    # --- Init streamer ---
    print("[Setup] Initializing BudgetedAsyncStreamer (pinning weights, allocating GPU slots)...")
    from marley.ops.async_stream import BudgetedAsyncStreamer, StreamMetrics

    streamer = BudgetedAsyncStreamer(blocks=blocks, device=device, dtype=dtype)

    # Slots consume ~2 Ã— 88.6 MB â‰ˆ 177 MB VRAM for FP16
    slot_vram_mb = 2 * sum(
        p.numel() * 2 for p in blocks[0].parameters()
    ) / (1024**2)
    print(f"[Setup] GPU slots allocated: ~{slot_vram_mb:.0f} MB VRAM")

    # --- Probe activation shapes ---
    print("[Setup] Probing WanTransformerBlock input shapes (CPU)...")
    hidden_states, encoder_hidden_states, temb, rotary_emb = streamer.probe_shapes(
        transformer=transformer,
        latent_shape=(1, 16, 5, 60, 104),   # 480p / 17f baseline
        text_len=226,
        text_dim=4096,
    )
    print(f"[Setup] hidden_states   : {tuple(hidden_states.shape)} {hidden_states.dtype}")
    print(f"[Setup] encoder_hidden  : {tuple(encoder_hidden_states.shape)}")
    print(f"[Setup] temb            : {tuple(temb.shape)}")
    if isinstance(rotary_emb, torch.Tensor):
        print(f"[Setup] rotary_emb      : {tuple(rotary_emb.shape)} {rotary_emb.dtype}")
    elif isinstance(rotary_emb, (tuple, list)):
        shapes = [tuple(x.shape) if isinstance(x, torch.Tensor) else x for x in rotary_emb]
        print(f"[Setup] rotary_emb      : tuple of {len(rotary_emb)} tensors, shapes={shapes}")
    else:
        print(f"[Setup] rotary_emb      : {type(rotary_emb)}")
    print()

    results: Dict = {
        "device": device_name,
        "vram_total_mb": vram_total_mb,
        "dtype": str(dtype),
        "num_blocks": num_blocks,
        "num_steps": args.steps,
        "num_frames": args.frames,
        "slot_vram_mb": slot_vram_mb,
        "activation_shapes": {
            "hidden_states": list(hidden_states.shape),
            "encoder_hidden_states": list(encoder_hidden_states.shape),
            "temb": list(temb.shape),
            "rotary_emb": (
                [list(x.shape) for x in rotary_emb]
                if isinstance(rotary_emb, (tuple, list))
                else list(rotary_emb.shape)
            ),
        },
        "f3a": None,
        "f3b": None,
        "verdict": None,
    }

    nvml = NVMLSampler(device_index=0, interval_ms=50.0)

    # =========================================================================
    # F3-A â€” SCHEDULER VALIDATION
    # =========================================================================
    if not args.skip_f3a:
        print("=" * 70)
        print("  F3-A â€” Scheduler Validation (synthetic activations)")
        print("=" * 70)
        print(f"Running {args.f3a_reps} repetitions Ã— (Sync + Async) Ã— {args.steps} steps Ã— {num_blocks} blocks")
        print("Objective: demonstrate scheduler correctness, not performance.\n")

        f3a_sync_times = []
        f3a_async_times = []
        f3a_nan_detected = False
        f3a_overlap_pcts = []

        for rep in range(args.f3a_reps):
            print(f"  Rep {rep+1}/{args.f3a_reps} â€” Sync...")
            nvml.start()
            m_sync = streamer.execute_sync_loop(
                hidden_states, encoder_hidden_states, temb, rotary_emb,
                num_steps=args.steps, num_frames=args.frames, nan_check=True,
            )
            vram_sync = nvml.stop()
            m_sync.peak_vram_mb = vram_sync
            f3a_sync_times.append(m_sync.total_denoise_time_s)
            if m_sync.nan_inf_detected:
                f3a_nan_detected = True

            print(f"  Rep {rep+1}/{args.f3a_reps} â€” Async...")
            nvml.start()
            m_async = streamer.execute_async_loop(
                hidden_states, encoder_hidden_states, temb, rotary_emb,
                num_steps=args.steps, num_frames=args.frames, nan_check=True,
                sync_baseline_h2d_s=m_sync.total_h2d_transfer_time_s,
            )
            vram_async = nvml.stop()
            m_async.peak_vram_mb = vram_async
            f3a_async_times.append(m_async.total_denoise_time_s)
            f3a_overlap_pcts.append(m_async.effective_overlap_pct)
            if m_async.nan_inf_detected:
                f3a_nan_detected = True

            print(f"    Sync: {m_sync.total_denoise_time_s*1000:.1f} ms | "
                  f"Async: {m_async.total_denoise_time_s*1000:.1f} ms | "
                  f"Overlap: {m_async.effective_overlap_pct:.1f}% | "
                  f"NaN: {f3a_nan_detected}")

        # F3-A summary
        avg_sync = sum(f3a_sync_times) / len(f3a_sync_times)
        avg_async = sum(f3a_async_times) / len(f3a_async_times)
        avg_overlap = sum(f3a_overlap_pcts) / len(f3a_overlap_pcts)
        # Consistency check: std dev of async times (race condition detector)
        consistency_ms = statistics.stdev(f3a_async_times) * 1000 if len(f3a_async_times) > 1 else 0.0

        f3a_pass = check_kill_gates(
            peak_vram_mb=max(m_async.peak_vram_mb, m_sync.peak_vram_mb),
            nan_inf=f3a_nan_detected,
            overlap_pct=avg_overlap,
            stage="F3-A",
        )

        print(f"\nF3-A Results:")
        print(f"  Avg Sync wall-clock   : {avg_sync*1000:.1f} ms")
        print(f"  Avg Async wall-clock  : {avg_async*1000:.1f} ms")
        print(f"  Avg overlap           : {avg_overlap:.1f}% â€” {interpret_overlap(avg_overlap)}")
        print(f"  Output consistency    : Â±{consistency_ms:.1f} ms std dev (low = no race condition)")
        print(f"  NaN/Inf detected      : {f3a_nan_detected}")
        print(f"  F3-A verdict          : {'ðŸŸ¢ PASS' if f3a_pass else 'ðŸ”´ FAIL'}")

        results["f3a"] = {
            "pass": f3a_pass,
            "reps": args.f3a_reps,
            "avg_sync_ms": avg_sync * 1000,
            "avg_async_ms": avg_async * 1000,
            "avg_overlap_pct": avg_overlap,
            "consistency_std_ms": consistency_ms,
            "nan_inf_detected": f3a_nan_detected,
        }

        if not f3a_pass:
            print("\n[F3-A FAILED] Aborting before F3-B.")
            _save_and_exit(results, args)

    # =========================================================================
    # F3-B â€” REAL PERFORMANCE CERTIFICATION
    # =========================================================================
    if not args.skip_f3b:
        print()
        print("=" * 70)
        print("  F3-B â€” Real Wan2.1 DiT Execution Performance Certification")
        print("  Marley Sync (A) vs. Marley Async (B)")
        print("=" * 70)
        print(f"Running {args.steps} denoising steps Ã— {num_blocks} DiT blocks Ã— "
              f"{args.f3b_reps} repetitions (A/B alternated)")
        print("Objective: demonstrate STABLE wall-clock speedup of Marley Async over Marley Sync.\n")

        # --- Local runner for one condition. Returns (metrics, external wall-clock s) ---
        def _run_condition(mode: str, h_in: torch.Tensor) -> Tuple[StreamMetrics, float]:
            if mode == "sync":
                fn = streamer.execute_sync_loop
            else:
                fn = streamer.execute_async_loop
            nvml.start()
            t0 = time.perf_counter()
            mm = fn(
                h_in, encoder_hidden_states, temb, rotary_emb,
                num_steps=args.steps, num_frames=args.frames, nan_check=True,
            )
            wall = time.perf_counter() - t0
            vram = nvml.stop()
            mm.peak_vram_mb = vram
            return mm, wall

        # --- Warmup (discarded): stabilize clocks / driver / caching state ---
        if args.f3b_warmup:
            print("[Warmup] Running one discarded Sync + Async pass to stabilize GPU state...")
            for mode in ("sync", "async"):
                _run_condition(mode, hidden_states.clone())
            print("[Warmup] Done (results discarded).\n")

        # --- Repeated A/B runs, alternating execution order per rep (Enmienda 2) ---
        per_rep: List[Dict] = []
        wall_a_s, wall_b_s = [], []
        internal_a_s, internal_b_s = [], []
        speedups, overlaps, peaks, nan_any = [], [], [], False
        stalls_total_ms: List[float] = []

        for rep in range(args.f3b_reps):
            print(f"--- Rep {rep + 1}/{args.f3b_reps} ---")
            # Independent copies of the initial state for A and B (defensive; Enmienda 3).
            h_a = hidden_states.clone()
            h_b = hidden_states.clone()

            # Alternate order to avoid systematic thermal/caching bias on B.
            do_async_first = (rep % 2 == 1)
            if do_async_first:
                metrics_b, wall_b = _run_condition("async", h_b)
                metrics_a, wall_a = _run_condition("sync", h_a)
                order = "B/A"
            else:
                metrics_a, wall_a = _run_condition("sync", h_a)
                metrics_b, wall_b = _run_condition("async", h_b)
                order = "A/B"

            # Official metric: EXTERNAL wall-clock (Enmienda 3).
            speedup_pct = ((wall_a - wall_b) / wall_a) * 100.0
            overlap_pct = metrics_b.effective_overlap_pct
            peak_vram = max(metrics_a.peak_vram_mb, metrics_b.peak_vram_mb)

            wall_a_s.append(wall_a)
            wall_b_s.append(wall_b)
            internal_a_s.append(metrics_a.total_denoise_time_s)
            internal_b_s.append(metrics_b.total_denoise_time_s)
            speedups.append(speedup_pct)
            overlaps.append(overlap_pct)
            peaks.append(peak_vram)
            nan_any = nan_any or metrics_a.nan_inf_detected or metrics_b.nan_inf_detected
            stalls_total_ms.append(metrics_b.prefetch_latency_s * 1000.0)

            print(f"  Order {order} | Sync wall {wall_a*1000:8.1f} ms | "
                  f"Async wall {wall_b*1000:8.1f} ms | Speedup {speedup_pct:+6.1f}% | "
                  f"Overlap {overlap_pct:5.1f}% | Stall {metrics_b.prefetch_latency_s*1000:6.2f} ms | "
                  f"Peak VRAM {peak_vram:5.0f} MB | NaN {nan_any}")

            row = {
                "rep": rep + 1,
                "order": order,
                "speedup_pct": speedup_pct,
                "overlap_pct": overlap_pct,
                "wall_a_ms": wall_a * 1000.0,
                "wall_b_ms": wall_b * 1000.0,
                "internal_a_ms": metrics_a.total_denoise_time_s * 1000.0,
                "internal_b_ms": metrics_b.total_denoise_time_s * 1000.0,
                "h2d_a_ms": metrics_a.total_h2d_transfer_time_s * 1000.0,
                "h2d_b_ms": metrics_b.total_h2d_transfer_time_s * 1000.0,
                "stall_b_ms": metrics_b.prefetch_latency_s * 1000.0,
                "forced_syncs_a": metrics_a.forced_sync_count,
                "forced_syncs_b": metrics_b.forced_sync_count,
                "peak_vram_mb": peak_vram,
                "nan_inf": metrics_a.nan_inf_detected or metrics_b.nan_inf_detected,
                "timeline": {
                    "per_block_copy_ms": [v * 1000.0 for v in metrics_b.per_block_copy_s],
                    "per_block_stall_ms": [v * 1000.0 for v in metrics_b.per_block_stall_s],
                    "per_block_compute_ms": [v * 1000.0 for v in metrics_b.per_block_compute_s],
                },
            }
            per_rep.append(row)

        # --- Aggregate statistics (Enmienda 2) ---
        agg_speedup = summarize(speedups)
        agg_wall_a = summarize(wall_a_s)
        agg_wall_b = summarize(wall_b_s)
        agg_int_a = summarize(internal_a_s)
        agg_int_b = summarize(internal_b_s)
        agg_overlap = summarize(overlaps)
        peak_vram = max(peaks)
        min_speedup = min(speedups)
        min_overlap = min(overlaps)
        mean_stall_ms = sum(stalls_total_ms) / len(stalls_total_ms)

        # --- Kill gates over the WHOLE battery (worst case per gate) ---
        f3b_pass = check_kill_gates(
            peak_vram_mb=peak_vram,
            nan_inf=nan_any,
            overlap_pct=min_overlap,
            speedup_pct=min_speedup,
            stage="F3-B",
        )

        # --- VRAM rating ---
        if peak_vram <= 4000:
            vram_rating = "ðŸŸ¢ Excellent (â‰¤ 4,000 MB)"
        elif peak_vram <= 4800:
            vram_rating = "ðŸŸ¡ Acceptable (â‰¤ 4,800 MB)"
        else:
            vram_rating = "ðŸ”´ ABORT (> 4,800 MB)"

        print()
        print("=" * 70)
        print("  F3-B A/B VERIFICATION SCORECARD (external wall-clock, official)")
        print("=" * 70)
        print(f"  {'Metric':<34} {'Sync (A)':<22} {'Async (B)':<22}")
        print(f"  {'-'*34} {'-'*22} {'-'*22}")
        print(f"  {'External wall-clock':<34} {agg_wall_a['mean']*1000:>11.1f} ms    {agg_wall_b['mean']*1000:>11.1f} ms")
        print(f"  {'  (min / max)':<34} {agg_wall_a['min']*1000:>11.1f} / {agg_wall_a['max']*1000:<11.1f} ms    {agg_wall_b['min']*1000:>11.1f} / {agg_wall_b['max']*1000:<11.1f} ms")
        print(f"  {'  (std dev)':<34} {agg_wall_a['std']*1000:>11.1f} ms    {agg_wall_b['std']*1000:>11.1f} ms")
        print(f"  {'Internal denoise (diag.)':<34} {agg_int_a['mean']*1000:>11.1f} ms    {agg_int_b['mean']*1000:>11.1f} ms")
        print(f"  {'Effective overlap (real)':<34} {'0.0%':>22} {agg_overlap['mean']:>20.1f}%")
        print(f"  {'  (min / max)':<34} {'':>22} {agg_overlap['min']:>12.1f}% / {agg_overlap['max']:<8.1f}%")
        print(f"  {'Stall total (real, ms)':<34} {'0.00':>22} {mean_stall_ms:>19.2f}")
        print(f"  {'Peak VRAM (NVML, max)':<34} {'':>22} {peak_vram:>18.0f} MB")
        print(f"  {'Forced syncs':<34} {per_rep[0]['forced_syncs_a']:>22} {per_rep[0]['forced_syncs_b']:>22}")
        print(f"  {'NaN/Inf (any rep)':<34} {str(nan_any):>22}")
        print()
        print(f"  Wall-clock speedup (mean) : {agg_speedup['mean']:+.1f}%  "
              f"(min {agg_speedup['min']:+.1f}% / max {agg_speedup['max']:+.1f}% / std Â±{agg_speedup['std']:.1f}%)")
        print(f"    â€” {interpret_speedup(agg_speedup['mean'])}")
        print(f"  Overlap quality (min gate) : {min_overlap:.1f}% â€” {interpret_overlap(min_overlap)}")
        print(f"  Peak VRAM                  : {peak_vram:.0f} MB â€” {vram_rating}")
        print(f"  F3-B verdict               : {'ðŸŸ¢ PASS' if f3b_pass else 'ðŸ”´ FAIL'}")
        print("=" * 70)

        results["f3b"] = {
            "pass": f3b_pass,
            "reps": args.f3b_reps,
            "warmup": bool(args.f3b_warmup),
            "order_pattern": "alternating A/B and B/A per repetition",
            "official_metric": "external wall-clock (perf_counter bracketing each condition)",
            "aggregate": {
                "speedup_pct": agg_speedup,
                "wall_clock_sync_ms": {k: v * 1000.0 for k, v in agg_wall_a.items()},
                "wall_clock_async_ms": {k: v * 1000.0 for k, v in agg_wall_b.items()},
                "internal_sync_ms": {k: v * 1000.0 for k, v in agg_int_a.items()},
                "internal_async_ms": {k: v * 1000.0 for k, v in agg_int_b.items()},
                "overlap_pct": agg_overlap,
                "stall_total_ms_mean": mean_stall_ms,
                "peak_vram_mb": peak_vram,
                "nan_inf_detected": nan_any,
            },
            "per_rep": per_rep,
            "speedup_interpretation": interpret_speedup(agg_speedup["mean"]),
            "overlap_interpretation": interpret_overlap(min_overlap),
            "vram_rating": vram_rating,
        }

    # =========================================================================
    # Overall F3 Verdict
    # =========================================================================
    f3a_pass = results["f3a"]["pass"] if results["f3a"] else True
    f3b_pass = results["f3b"]["pass"] if results["f3b"] else True
    overall_pass = f3a_pass and f3b_pass

    print("=" * 70)
    print("  PHASE F3 OVERALL VERDICT")
    print("=" * 70)
    if results["f3a"]:
        print(f"  F3-A (Scheduler Validation) : {'ðŸŸ¢ PASS' if f3a_pass else 'ðŸ”´ FAIL'}")
    if results["f3b"]:
        print(f"  F3-B (Performance Cert.)    : {'ðŸŸ¢ PASS' if f3b_pass else 'ðŸ”´ FAIL'}")
    print(f"  Phase F3                    : {'ðŸŸ¢ PASS â€” Ready for F3+INT8' if overall_pass else 'ðŸ”´ FAIL â€” Review required'}")
    print("=" * 70)

    results["verdict"] = {
        "f3a_pass": f3a_pass,
        "f3b_pass": f3b_pass,
        "overall_pass": overall_pass,
        "phase": "F3",
        "next_phase": "F3+INT8" if overall_pass else "F3 review required",
    }

    # --- Cleanup ---
    streamer.restore_all_to_cpu()
    streamer.release()

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_exit(results: Dict, args: argparse.Namespace) -> None:
    _save_json(results, args.output)
    sys.exit(1)


def _save_json(results: Dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Output] Results saved to: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase F3 â€” Budgeted Asynchronous Scheduler Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of denoising steps to simulate (default: 30)."
    )
    parser.add_argument(
        "--frames", type=int, default=17,
        help="Number of output video frames (default: 17, the 480p baseline)."
    )
    parser.add_argument(
        "--f3a-reps", type=int, default=5,
        help="Number of F3-A repetitions for race condition detection (default: 5)."
    )
    parser.add_argument(
        "--f3b-reps", type=int, default=5,
        help="Number of F3-B A/B repetitions for reproducibility statistics (default: 5). "
             "Execution order alternates A/B and B/A between repetitions."
    )
    parser.add_argument(
        "--no-f3b-warmup", dest="f3b_warmup", action="store_false",
        help="Disable the discarded warmup pass before measured F3-B repetitions."
    )
    parser.set_defaults(f3b_warmup=True)
    parser.add_argument(
        "--skip-f3a", action="store_true",
        help="Skip F3-A scheduler validation (run F3-B only)."
    )
    parser.add_argument(
        "--skip-f3b", action="store_true",
        help="Skip F3-B performance certification (run F3-A only)."
    )
    parser.add_argument(
        "--output", type=str, default="logs/f3_async_scheduler_benchmark.json",
        help="Output JSON path (default: logs/f3_async_scheduler_benchmark.json)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmark(args)
    _save_json(results, args.output)

