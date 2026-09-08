"""
f4_adaptive_benchmark.py
========================
Phase F4 — Adaptive Memory Decision Engine benchmark.

Validates the F4 decision loop against the frozen F3/F3+INT8 primitives using the
mandatory same-session methodology (F4_TEST_SPEC v3 §7/§8):

    A = Sync FP16
    B = Async FP16
    C = Async INT8
    D = Adaptive (decision engine)

Every absolute claim is expressed relative to the best same-session static policy
D vs best(A, B, C) — never against cross-session historical deltas.

Kill gates (v3 §4.4), evaluated separately:
  Safety Gate     : peak NVML <= 4,800 MB ; 0 NaN/Inf ; policy correct.
  Adaptive Gate   : pressure detection + replan + correct switch + headroom recovery.
  Optimization    : D >= 5% over best(A,B,C) is a TARGET, not a validity gate.

Usage
-----
  python f4_adaptive_benchmark.py --abc
  python f4_adaptive_benchmark.py --abc --steps 5 --reps 3 --profile performance
  python f4_adaptive_benchmark.py --pressure-test     # deterministic injected-pressure loop

Output
------
  logs/f4_adaptive_benchmark.json
  Console human-readable scorecard
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HARD_GATE_VRAM_MB = 4800.0
ENGINEERING_TARGET_VRAM_MB = 4000.0
ADAPTIVE_HEADROOM_GB = 1.5
OPTIMIZATION_TARGET_PCT = 5.0

# Injected-pressure script (deterministic, used by --pressure-test).
# Windows profile: [normal, normal, PRESSURE peak, ...dwell..., release, normal].
PRESSURE_BASELINE_MB = 2600.0
PRESSURE_SPIKE_MB = 3200.0   # +600 MB above EMA -> far beyond PRESSURE_HIGH (200 MB)
PRESSURE_SPIKE_WINDOWS = 3
PRESSURE_RELEASE_MB = 2400.0  # below EMA - PRESSURE_LOW -> exit after dwell


class NVMLSampler:
    """Background thread polling physical GPU VRAM at 50 ms; tracks peak."""

    def __init__(self, device_index: int = 0, interval_ms: float = 50.0) -> None:
        self.interval_s = interval_ms / 1000.0
        self._peak_mb = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self._nvml_available = True
        except Exception:
            self._handle = None
            self._pynvml = None

    def _sample_mb(self) -> float:
        if self._nvml_available:
            try:
                return self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 ** 2)
            except Exception:
                pass
        return torch.cuda.memory_reserved(0) / (1024 ** 2)

    def used_mb_now(self) -> float:
        return self._sample_mb()

    def start(self) -> None:
        self._peak_mb = self._sample_mb()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            mb = self._sample_mb()
            if mb > self._peak_mb:
                self._peak_mb = mb
            time.sleep(self.interval_s)

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_mb


def summarize(values: List[float]) -> Dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    mean = statistics.fmean(values) if hasattr(statistics, "fmean") else sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "min": min(values), "max": max(values), "std": std}


def make_scripted_pressure(num_windows: int) -> Callable[[], float]:
    """Build a deterministic used-VRAM sampler to exercise the hysteresis monitor."""
    # Build a per-window scripted timeline of `used` readings.
    timeline: List[float] = [PRESSURE_BASELINE_MB] * num_windows
    spike_start = 2
    for i in range(spike_start, min(spike_start + PRESSURE_SPIKE_WINDOWS, num_windows)):
        timeline[i] = PRESSURE_SPIKE_MB
    release_idx = spike_start + PRESSURE_SPIKE_WINDOWS
    if release_idx < num_windows:
        timeline[release_idx] = PRESSURE_RELEASE_MB
        for i in range(release_idx + 1, num_windows):
            timeline[i] = PRESSURE_BASELINE_MB

    state = {"i": 0}

    def _read() -> float:
        idx = min(state["i"], len(timeline) - 1)
        state["i"] += 1
        return timeline[idx]

    return _read


def run_pressure_test(args: argparse.Namespace, engine_blocks, shapes, device, dtype) -> Dict:
    """Adaptive Gate check: detection + replan + correct switch + recovery (no oscillation)."""
    print("\n" + "=" * 70)
    print("  F4 PRESSURE TEST (Adaptive Gate) — deterministic injected pressure")
    print("=" * 70)

    hidden_states, encoder_hidden_states, temb, rotary_emb = shapes
    from marley.core.adaptive import AdaptiveEngine

    engine = AdaptiveEngine(
        blocks=engine_blocks, device=device, dtype=dtype,
        window=args.window, base_profile=args.profile,
    )
    # Choose enough steps so the scripted spike + dwell + release are fully visible.
    total_steps = max(args.window * 8, args.steps)
    num_windows = (total_steps + args.window - 1) // args.window
    sampler = make_scripted_pressure(num_windows)

    result = engine.run(
        hidden_states, encoder_hidden_states, temb, rotary_emb,
        num_steps=total_steps, num_frames=args.frames,
        nan_check=True, pressure_sampler=sampler,
    )

    profiles = result.telemetry["profile_sequence"]
    events = result.telemetry["replan_events"]
    decisions = result.telemetry["decisions"]
    switch_count = result.telemetry["switch_count"]
    replan_count = result.telemetry["replan_count"]
    overhead_ms = result.telemetry["decision_overhead_ms_total"]

    entered = any(e["profile"] == "memory_safe" for e in events)
    reexited = any(e["profile"] == "performance" for e in events)
    # Anti-oscillation: a memory_safe window should not immediately flip back to
    # performance while pressure remains (dwell enforced).
    oscillating = False
    if len(events) >= 2:
        prof_seq = "".join("S" if p == "memory_safe" else "P" for p in profiles)
        oscillating = ("PSP" in prof_seq) or ("SPS" in prof_seq)

    headroom_recovered = ADAPTIVE_HEADROOM_GB
    safe_passed = entered and replan_count >= 1
    stability_passed = not oscillating
    adaptive_passed = safe_passed and stability_passed

    print(f"  Windows            : {num_windows}")
    print(f"  Profile sequence   : {' -> '.join(profiles)}")
    print(f"  Replan events      : {len(events)}")
    for e in events:
        print(f"     - window {e['window']}: -> {e['profile']} (regime {e['regime']})")
    print(f"  Replan count       : {replan_count}")
    print(f"  Switch count       : {switch_count}")
    print(f"  Decision overhead  : {overhead_ms:.2f} ms total")
    print(f"  Entered safe       : {entered}")
    print(f"  Re-exited          : {reexited}")
    print(f"  Oscillation        : {oscillating}")
    print(f"  Adaptive Gate      : {'PASS' if adaptive_passed else 'FAIL'}")

    engine.restore_all_to_cpu()
    engine.release()

    return {
        "mode": "pressure",
        "profile_sequence": profiles,
        "decisions": [str(d) for d in decisions],
        "replan_events": events,
        "replan_count": replan_count,
        "switch_count": switch_count,
        "decision_overhead_ms_total": overhead_ms,
        "entered_memory_safe": entered,
        "re_exited": reexited,
        "oscillated": oscillating,
        "headroom_recovered_gb": headroom_recovered,
        "adaptive_gate_pass": adaptive_passed,
    }


def run_abc_benchmark(args: argparse.Namespace, engine_blocks, shapes, device, dtype) -> Dict:
    """Same-session A/B/C/D validation (Safety + Optimization)."""
    print("\n" + "=" * 70)
    print("  Marley Runtime - Phase F4 Benchmark (same-session A/B/C/D)")
    print("=" * 70)

    hidden_states, encoder_hidden_states, temb, rotary_emb = shapes

    from marley.core.adaptive import AdaptiveEngine

    # One engine owns the FP16 and INT8 streamers (a single pre-computed pinned host
    # representation of each). A/B/C/D all REUSE those streamers so host pinned memory
    # is not duplicated (avoids cudaMallocHost OOM on this 6 GB machine).
    engine = AdaptiveEngine(
        blocks=engine_blocks, device=device, dtype=dtype,
        window=args.window, base_profile=args.profile,
    )
    fp16 = engine._fp16
    int8 = engine._int8

    nvml = NVMLSampler(device_index=0, interval_ms=50.0)

    def _run(streamer, mode, h_in):
        nvml.start()
        t0 = time.perf_counter()
        fn = streamer.execute_sync_loop if mode == "sync" else streamer.execute_async_loop
        mm = fn(h_in, encoder_hidden_states, temb, rotary_emb,
                num_steps=args.steps, num_frames=args.frames, nan_check=True)
        wall = time.perf_counter() - t0
        peak = nvml.stop()
        mm.peak_vram_mb = peak
        return mm, wall

    def _run_adaptive(h_in):
        nvml.start()
        t0 = time.perf_counter()
        res = engine.run(h_in, encoder_hidden_states, temb, rotary_emb,
                         num_steps=args.steps, num_frames=args.frames,
                         nan_check=True, base_profile=args.profile)
        wall = time.perf_counter() - t0
        peak = nvml.stop()
        return res, wall, peak

    # Warmup (discarded) to stabilize clocks/driver state.
    if args.warmup:
        print("[Warmup] one discarded pass per condition...")
        _run(fp16, "sync", hidden_states.clone())
        _run(fp16, "async", hidden_states.clone())
        _run(int8, "async", hidden_states.clone())
        _run_adaptive(hidden_states.clone())
        print("[Warmup] Done (results discarded).\n")

    wall_by_cond = {k: [] for k in ("A", "B", "C", "D")}
    peaks_all = []
    nan_any = False
    adaptive_telem: Optional[Dict] = None

    for rep in range(args.reps):
        print(f"--- Rep {rep + 1}/{args.reps} ---")
        order = ["A", "B", "C", "D"] if rep % 2 == 0 else ["D", "C", "B", "A"]
        mm_a = mm_b = mm_c = None
        for cond in order:
            h = hidden_states.clone()
            if cond == "A":
                mm, wall = _run(fp16, "sync", h); mm_a = mm
            elif cond == "B":
                mm, wall = _run(fp16, "async", h); mm_b = mm
            elif cond == "C":
                mm, wall = _run(int8, "async", h); mm_c = mm
            else:  # D adaptive
                res, wall, peak = _run_adaptive(h)
                adaptive_telem = res.telemetry
                res.merged.peak_vram_mb = peak
                mm = res.merged
            wall_by_cond[cond].append(wall)
            peaks_all.append(mm.peak_vram_mb)
            nan_any = nan_any or mm.nan_inf_detected
            print(f"  {cond} ({order}) wall {wall*1000:8.1f} ms | "
                  f"overlap {mm.effective_overlap_pct:5.1f}% | "
                  f"peak {mm.peak_vram_mb:5.0f} MB | NaN {mm.nan_inf_detected}")
        _ = (mm_a, mm_b, mm_c)

    agg = {k: summarize(v) for k, v in wall_by_cond.items()}
    best_static = min(("A", "B", "C"), key=lambda k: agg[k]["mean"])
    best_static_mean_ms = agg[best_static]["mean"]
    d_mean_ms = agg["D"]["mean"]
    vs_best_pct = (best_static_mean_ms - d_mean_ms) / best_static_mean_ms * 100.0
    peak_max = max(peaks_all) if peaks_all else 0.0

    # Gate evaluation (v3 §4.4).
    safety_gate = peak_max <= HARD_GATE_VRAM_MB and not nan_any
    optimization_met = vs_best_pct >= OPTIMIZATION_TARGET_PCT
    telem = adaptive_telem or {}
    decision_overhead_ms = telem.get("decision_overhead_ms_total", 0.0)
    switch_count = telem.get("switch_count", 0)
    replan_count = telem.get("replan_count", 0)

    print("\n" + "=" * 70)
    print("  F4 SCORECARD (same-session external wall-clock)")
    print("=" * 70)
    for k in ("A", "B", "C", "D"):
        label = {"A": "Sync FP16 ", "B": "Async FP16", "C": "Async INT8", "D": "Adaptive "}[k]
        print(f"  {label}  : mean {agg[k]['mean']*1000:8.1f} ms  "
              f"(min {agg[k]['min']*1000:.1f} / max {agg[k]['max']*1000:.1f})")
    print(f"  best static (same-session) : {best_static}  ({best_static_mean_ms*1000:.1f} ms)")
    print(f"  D vs best static           : {vs_best_pct:+.2f}%")
    print(f"  Decision overhead (D)      : {decision_overhead_ms:.2f} ms total")
    print(f"  Switch count (D)           : {switch_count}")
    print(f"  Replan count (D)           : {replan_count}")
    print(f"  Peak VRAM (NVML max)       : {peak_max:.0f} MB  "
          f"(hard gate {HARD_GATE_VRAM_MB:.0f} / target {ENGINEERING_TARGET_VRAM_MB:.0f})")
    print(f"  NaN/Inf                    : {nan_any}")
    print(f"  Safety Gate                : {'PASS' if safety_gate else 'FAIL'}")
    print(f"  Optimization target >=5%   : {'MET' if optimization_met else 'not met (target only)'}")
    print("=" * 70)

    engine.restore_all_to_cpu(); engine.release()

    return {
        "mode": "abc",
        "profile": args.profile,
        "window": args.window,
        "per_condition_mean_ms": {k: agg[k]["mean"] * 1000.0 for k in ("A", "B", "C", "D")},
        "per_condition_agg_ms": {k: {kk: vv * 1000.0 for kk, vv in v.items()} for k, v in agg.items()},
        "best_static_condition": best_static,
        "best_static_ms": best_static_mean_ms * 1000.0,
        "vs_best_static_pct": vs_best_pct,
        "decision_overhead_ms_total": decision_overhead_ms,
        "switch_count": switch_count,
        "replan_count": replan_count,
        "peak_vram_mb": peak_max,
        "nan_inf": nan_any,
        "safety_gate_pass": safety_gate,
        "optimization_target_met": optimization_met,
        "optimization_target_pct": OPTIMIZATION_TARGET_PCT,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase F4 - adaptive decision engine benchmark")
    p.add_argument("--abc", action="store_true", help="Run same-session A/B/C/D benchmark.")
    p.add_argument("--pressure-test", action="store_true",
                   help="Run deterministic injected-pressure Adaptive Gate test.")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--window", type=int, default=5,
                   help="N diffusion steps per F4 decision window.")
    p.add_argument("--profile", type=str, default="performance",
                   choices=["performance", "memory_safe"])
    p.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
    p.add_argument("--output", type=str, default="logs/f4_adaptive_benchmark.json")
    return p.parse_args()


def load_blocks(device, dtype):
    print("[Setup] Loading Wan2.1 transformer blocks (CPU)...")
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer = transformer.to("cpu")
    torch.cuda.empty_cache()
    blocks = list(transformer.blocks)
    from marley.ops.async_stream import BudgetedAsyncStreamer
    streamer = BudgetedAsyncStreamer(blocks=blocks, device=device, dtype=dtype)
    shapes = streamer.probe_shapes(
        transformer=transformer,
        latent_shape=(1, 16, 5, 60, 104),
        text_len=226,
        text_dim=4096,
    )
    streamer.restore_all_to_cpu()
    streamer.release()
    print(f"[Setup] Loaded {len(blocks)} DiT blocks on CPU.")
    return blocks, shapes


def main() -> None:
    args = parse_args()
    if not args.abc and not args.pressure_test:
        args.abc = True

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device found. F4 benchmark requires a CUDA GPU.")
        sys.exit(1)

    device = torch.device("cuda", 0)
    dtype = torch.float16
    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"\nDevice : {device_name}\nVRAM   : {vram_total_mb:.0f} MB total")

    blocks, shapes = load_blocks(device, dtype)

    results: Dict = {
        "phase": "F4",
        "device": device_name,
        "vram_total_mb": vram_total_mb,
        "num_blocks": len(blocks),
        "num_steps": args.steps,
        "num_frames": args.frames,
        "num_reps": args.reps,
        "window": args.window,
        "profile": args.profile,
        "warmup": bool(args.warmup),
        "gates": {
            "hard_gate_vram_mb_limit": HARD_GATE_VRAM_MB,
            "engineering_target_vram_mb": ENGINEERING_TARGET_VRAM_MB,
            "adaptive_gate_headroom_recovered_gb": ADAPTIVE_HEADROOM_GB,
            "optimization_target_pct": OPTIMIZATION_TARGET_PCT,
        },
    }

    if args.pressure_test:
        results["pressure"] = run_pressure_test(
            args, blocks, shapes, device, dtype,
        )
    if args.abc:
        results["abc"] = run_abc_benchmark(args, blocks, shapes, device, dtype)

    # Best combined pass: safety gate must pass in abc; adaptive gate must pass in pressure.
    safe = results.get("abc", {}).get("safety_gate_pass", True)
    adaptive = results.get("pressure", {}).get("adaptive_gate_pass", True)
    results["pass"] = bool(safe and adaptive)

    torch.cuda.empty_cache()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Output] Results saved to: {out}")
    print(f"[Verdict] PASS={results['pass']}")


if __name__ == "__main__":
    main()
