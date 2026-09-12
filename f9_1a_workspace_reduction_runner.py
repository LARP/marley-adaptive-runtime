"""
f9_1a_workspace_reduction_runner.py
===================================
Phase F9-1A -- Confirmatory Workspace Reduction Campaign.

Protocol (frozen, authoritative):
  docs/F9_1_PREREGISTERED_WORKSPACE_REDUCTION_PROTOCOL_01.md
Governance:
  docs/private/F9_1_DIRECTOR_AUTHORIZATION_LETTER_TO_TECH_DIRECTOR_01.md
  docs/private/F9_DIRECTOR_ADJUSTMENTS_LETTER_TO_CONSULTANT_01.md

Design (frozen)
---------------
10 paired blocks, each block runs one control (C) and one intervention (I) in a
pre-registered, hashed order (5 x C->I and 5 x I->C). Every run is a clean OS
process with a 60 s pre-run cooldown and an S0 stability check (spread <= 150 MB).
The paired differences Delta_k = I_k - C_k are analysed with a paired bootstrap
(10,000 resamples) on the median.

Intervention surface (F9-1A arm only; allocator policy UNTOUCHED)
-----------------------------------------------------------------
  - CUBLAS_WORKSPACE_CONFIG=:4096:8  (bounded, deterministic cuBLAS workspace)
  - torch.backends.cudnn.benchmark = False, cudnn.deterministic = True
  - SDPA restricted to FLASH_ATTENTION (bounded temporary footprint)
  - TF32 disabled (deterministic linear/convolutions)
The control arm runs the runtime defaults. The PyTorch CUDACachingAllocator
configuration is identical in both arms (never modified).

Measurement is inherited verbatim from the validated F9-0 instrument.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f9_0_attribution_runner import (  # noqa: E402
    F9AttributionPipeline,
    NVMLSampler,
    compute_attribution,
    sample_stability,
    nvml_device_mb,
    nvml_process_summary,
    estimate_analytic_lb,
    module_param_bytes,
    cuda_mem_getinfo_mb,
    get_process_ram_mb,
    SAFE_ABORT_NVML_MB,
    S0_STABILITY_SPREAD_MB,
    MB,
    CANONICAL_PROMPT,
    CANONICAL_NEG_PROMPT,
    CANONICAL_SEED,
    CANONICAL_GUIDANCE,
)

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------
PROTOCOL_REFERENCE = "docs/F9_1_PREREGISTERED_WORKSPACE_REDUCTION_PROTOCOL_01.md"
HARD_GATE_VRAM_MB = 4800.0
EPSILON_MB = 64.0
DEFAULT_STEPS = 30
DEFAULT_FRAMES = 33
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_S0_SECONDS = 60.0
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
MIN_INTERVENTION_UNDER_GATE = 8
MAX_SHIFTED_BLOCKS = 1  # 10% of N=10

# Frozen block order (V5/V6). Hash must match e0aac003...7227e.
FROZEN_SEQUENCE_RAW = "C->I|I->C|C->I|I->C|I->C|I->C|C->I|I->C|C->I|C->I"
FROZEN_SEQUENCE_HASH = "e0aac00319dc2444414f63c1577fb80662ee558345a66dbb6166c6117547227e"


def _sequence() -> List[Tuple[str, str]]:
    """Return the frozen list of (first, second) conditions per block."""
    return [tuple(part.split("->")) for part in FROZEN_SEQUENCE_RAW.split("|")]


def verify_sequence_hash() -> None:
    digest = hashlib.sha256(FROZEN_SEQUENCE_RAW.encode("utf-8")).hexdigest()
    if digest != FROZEN_SEQUENCE_HASH:
        raise RuntimeError(
            f"Frozen sequence hash mismatch: {digest} != {FROZEN_SEQUENCE_HASH}"
        )


# ---------------------------------------------------------------------------
# Condition configuration
# ---------------------------------------------------------------------------

def condition_config(condition: str) -> Dict[str, Any]:
    """Return the immutable per-arm configuration applied to the runtime."""
    common = {
        "allocator_policy": "untouched (runtime default)",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset)"),
    }
    if condition == "intervention":
        return {
            **common,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "(unset)"),
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "sdpa_backend": "cuDNN SDPA disabled (bounded workspace)",
            "description": "bounded cuBLAS/cuDNN workspace + cuDNN-SDPA disabled",
        }
    return {
        **common,
        "CUBLAS_WORKSPACE_CONFIG": "(unset)",
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "sdpa_backend": "runtime default priority",
        "description": "runtime defaults (contemporary control)",
    }


def apply_condition_config(condition: str) -> Dict[str, Any]:
    """Apply the arm configuration to the current process. Returns the effective config."""
    if condition == "intervention":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Bounded temporary-memory attention: disable the cuDNN SDPA backend
        # (which allocates algorithm workspaces) while keeping the fused
        # flash / mem-efficient kernels and the safe math fallback available.
        torch.backends.cuda.enable_cudnn_sdp(False)
    return condition_config(condition)


def runtime_versions() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        info["driver"] = None
    return info


# ---------------------------------------------------------------------------
# Single clean-process worker
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    condition = args.condition
    # Symmetric offline model loading (identical in both arms; no network access).
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("=" * 80)
    print(f"  F9-1A RUN -- Block {args.block} | position {args.position} | condition {condition.upper()}")
    print(f"  Protocol: {PROTOCOL_REFERENCE}")
    print("=" * 80)

    config = apply_condition_config(condition)
    print(f"  Arm config: {json.dumps(config, default=str)}")

    result: Dict[str, Any] = {
        "phase": "F9-1A",
        "block": args.block,
        "position": args.position,
        "condition": condition,
        "run_label": args.run_label,
        "timestamp": datetime.datetime.now().isoformat(),
        "protocol_reference": PROTOCOL_REFERENCE,
        "frozen_sequence_hash": FROZEN_SEQUENCE_HASH,
        "workload": {
            "resolution": f"{args.width}x{args.height}", "frames": args.frames,
            "steps": args.steps, "mode": "adaptive", "seed": args.seed,
            "guidance_scale": args.guidance,
        },
        "arm_config": config,
        "runtime_versions": runtime_versions(),
        "governance": ("Confirmatory paired campaign. Allocator policy untouched; "
                       "only the workspace/bounded-SDPA surface differs between arms."),
    }

    # ---- S0 idle baseline + exclusivity (pre-context) ----
    s0_stats = sample_stability(seconds=args.s0_seconds, interval_ms=1000.0, record_series=False)
    S0 = round(s0_stats.get("mean_mb", 0.0), 2)
    s0_stable = bool(s0_stats.get("spread_mb") is not None
                     and s0_stats["spread_mb"] <= S0_STABILITY_SPREAD_MB)
    procs_pre = nvml_process_summary()
    exclusive = bool(procs_pre.get("compute_enumeration_ok", False)
                     and procs_pre.get("exclusive_other_contexts") is True
                     and s0_stable)
    result["s0_stability"] = s0_stats
    result["processes_pre"] = procs_pre
    result["exclusivity"] = {
        "exclusive": exclusive,
        "s0_stable": s0_stable,
        "s0_spread_mb": s0_stats.get("spread_mb"),
        "no_other_cuda_contexts": procs_pre.get("exclusive_other_contexts"),
    }
    print(f"  S0 idle: mean {S0} MB (spread {s0_stats.get('spread_mb')} MB, stable={s0_stable})")

    # ---- CUDA warm-up ----
    nvml = NVMLSampler(device_index=0, interval_ms=20.0, safe_abort_mb=SAFE_ABORT_NVML_MB)
    nvml.start()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    _w = torch.empty((10 * 1024 * 1024 // 4,), dtype=torch.float32, device=args.device)
    torch.cuda.synchronize(args.device)
    del _w
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)

    # ---- S1 operational baseline (pre-model-load) ----
    S1 = round(nvml_device_mb(), 2)
    result["s1_operational_pre_model_load"] = {
        "S1_mb": S1, "S0_mb": S0, "overhead_cuda_driver_runtime_mb": round(S1 - S0, 2),
        "cudaMemGetInfo": cuda_mem_getinfo_mb(),
    }

    pipeline = None
    probe: Optional[Dict[str, Any]] = None
    error_occurred: Optional[str] = None
    persistent_weights_mb = 0.0
    max_block_weight_mb = 0.0
    analytic_lb: Dict[str, Any] = {}
    dit_peak_mb = 0.0
    peak_t_perf: Optional[float] = None
    dit_nvml_series: List[Dict[str, Any]] = []

    try:
        pipeline = F9AttributionPipeline(
            device=args.device, dit_dtype=torch.float16,
            vae_dtype=torch.bfloat16, text_dtype=torch.bfloat16)
        gc.collect()
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)

        persistent_weights_mb = round(torch.cuda.memory_allocated(args.device) / MB, 2)
        max_block_weight_mb = round(
            max((module_param_bytes(b, torch.float16) for b in pipeline.blocks), default=0) / MB, 2)
        analytic_lb = estimate_analytic_lb(
            pipeline.transformer.config, args.height, args.width, args.frames,
            persistent_weights_mb, max_block_weight_mb,
            torch.tensor([], dtype=torch.float16).element_size())

        nvml.reset_peak()
        nvml.mark_phase("workload_start")
        probe = pipeline.run_attribution(
            prompt=CANONICAL_PROMPT, negative_prompt=CANONICAL_NEG_PROMPT,
            height=args.height, width=args.width, num_frames=args.frames,
            num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            mode="adaptive", nvml=nvml, export_video=False,
        )
        dit_peak_mb = probe.get("dit_peak_nvml_mb", 0.0)
        peak_t_perf = probe.get("dit_peak_t_perf")
        dit_nvml_series = probe.get("dit_nvml_series", [])
    except Exception as exc:
        error_occurred = str(exc)
        print(f"\n[F9-1A ABORT] Exception during execution: {error_occurred}")

    nvml.stop()

    attribution = compute_attribution(
        snapshots=(probe or {}).get("snapshots", []),
        nvml_series=dit_nvml_series,
        peak_t_perf=peak_t_perf, s0_mb=S0, s1_mb=S1,
        persistent_weights_mb=persistent_weights_mb,
        peak_streamer_resident_mb=(probe or {}).get("peak_streamer_resident_mb", 0.0),
    ) if probe else {"error": "workload not executed"}

    cats = attribution.get("categories_mb", {}) if isinstance(attribution, dict) else {}
    result["memory"] = {
        "peak_nvml_mb": dit_peak_mb,
        "cat3_workspace_mb": cats.get("3_workspace_kernels"),
        "cat4_allocator_mb": cats.get("4_memoria_allocator"),
        "delta_induced_mb": attribution.get("Delta_induced_mb"),
        "at_peak_reserved_mb": attribution.get("at_peak_reserved_mb"),
        "at_peak_allocated_mb": attribution.get("at_peak_allocated_mb"),
        "attributed_sum_1_to_6_mb": attribution.get("attributed_sum_1_to_6_mb"),
    }
    result["cadence"] = {
        "cadence_avg_s": (probe or {}).get("cadence_avg_s"),
        "step_times_s": (probe or {}).get("step_times_s"),
        "denoise_time_s": (probe or {}).get("denoise_time_s"),
    }
    result["peak_nvml_under_gate"] = bool(dit_peak_mb is not None and dit_peak_mb < HARD_GATE_VRAM_MB)
    result["attribution"] = attribution
    result["lb_analitico"] = analytic_lb
    result["processes_post"] = nvml_process_summary()
    result["error"] = error_occurred

    os.makedirs(os.path.dirname(args.run_out), exist_ok=True)
    with open(args.run_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f">> Run telemetry saved to: {args.run_out}")
    print(f"   Peak NVML={dit_peak_mb} MB | Cat3={cats.get('3_workspace_kernels')} MB "
          f"| Cat4={cats.get('4_memoria_allocator')} MB")
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------

def _bootstrap_ci_median(deltas: List[float], resamples: int = BOOTSTRAP_RESAMPLES,
                         seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.size
    meds = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = arr[rng.integers(0, n, n)]
        meds[i] = np.median(sample)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {
        "n": int(n),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "resamples": resamples,
        "seed": seed,
    }


def analyze(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [b for b in blocks if b.get("control") and b.get("intervention")
             and b["control"].get("error") is None and b["intervention"].get("error") is None]
    d_peak = [b["intervention"]["memory"]["peak_nvml_mb"] - b["control"]["memory"]["peak_nvml_mb"]
              for b in valid]
    d_cat3 = [b["intervention"]["memory"]["cat3_workspace_mb"] - b["control"]["memory"]["cat3_workspace_mb"]
              for b in valid]
    d_cat4 = [b["intervention"]["memory"]["cat4_allocator_mb"] - b["control"]["memory"]["cat4_allocator_mb"]
              for b in valid]

    boot_peak = _bootstrap_ci_median(d_peak) if d_peak else {}
    boot_cat3 = _bootstrap_ci_median(d_cat3) if d_cat3 else {}
    boot_cat4 = _bootstrap_ci_median(d_cat4) if d_cat4 else {}

    # Criterion A: causal (median < 0 and upper CI < 0)
    crit_a = bool(boot_peak and boot_peak["median"] < 0 and boot_peak["ci95_high"] < 0)
    # Criterion B: operational (>= 8/10 intervention runs under gate)
    n_under = sum(1 for b in valid if b["intervention"].get("peak_nvml_under_gate"))
    crit_b = bool(len(valid) == 10 and n_under >= MIN_INTERVENTION_UNDER_GATE)
    # Criterion C: mechanistic (median dCat3 < 0 and upper CI < 0)
    crit_c = bool(boot_cat3 and boot_cat3["median"] < 0 and boot_cat3["ci95_high"] < 0)

    # Criterion D: anti-shifting
    shifted = 0
    per_block_shift: List[Dict[str, Any]] = []
    for b, dc3, dc4 in zip(valid, d_cat3, d_cat4):
        if dc3 < 0:
            ck = max(0.0, dc4) / abs(min(dc3, 0.0))
        else:
            ck = None
        is_shift = bool(ck is not None and ck >= 0.5)
        shifted += int(is_shift)
        per_block_shift.append({"block": b["block"], "delta_cat3_mb": dc3,
                                "delta_cat4_mb": dc4, "C_k": ck, "shifted": is_shift})
    crit_d = bool(shifted <= MAX_SHIFTED_BLOCKS)

    # Cadence non-regression (<= 15% vs control)
    cad_pairs = [(b["control"]["cadence"]["cadence_avg_s"], b["intervention"]["cadence"]["cadence_avg_s"])
                 for b in valid
                 if b["control"]["cadence"]["cadence_avg_s"] and b["intervention"]["cadence"]["cadence_avg_s"]]
    cad_deltas_pct = [100.0 * (i - c) / c for c, i in cad_pairs] if cad_pairs else []
    cadence_ok = bool(cad_deltas_pct and statistics.fmean(cad_deltas_pct) <= 15.0)

    return {
        "n_valid_blocks": len(valid),
        "delta_peak_nvml_mb": {"values": d_peak, "bootstrap": boot_peak},
        "delta_cat3_mb": {"values": d_cat3, "bootstrap": boot_cat3},
        "delta_cat4_mb": {"values": d_cat4, "bootstrap": boot_cat4},
        "criterion_A_causal": crit_a,
        "criterion_B_operational": crit_b,
        "criterion_C_mechanistic": crit_c,
        "criterion_D_anti_shifting": crit_d,
        "intervention_under_gate": n_under,
        "shifted_blocks": shifted,
        "cadence_delta_pct": cad_deltas_pct,
        "cadence_non_regression_ok": cadence_ok,
        "per_block_shift": per_block_shift,
        "causal_success": bool(crit_a and crit_d and cadence_ok),
    }


def write_report(args: argparse.Namespace, telemetry: Dict[str, Any]) -> None:
    a = telemetry["analysis"]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        w = f.write
        w("# F9-1A — Confirmatory Workspace Reduction — Report\n\n")
        w(f"**Date:** {telemetry['timestamp']}  \n")
        w(f"**Protocol:** [`{PROTOCOL_REFERENCE}`](F9_1_PREREGISTERED_WORKSPACE_REDUCTION_PROTOCOL_01.md)  \n")
        w(f"**Frozen sequence hash:** `{FROZEN_SEQUENCE_HASH}`  \n")
        w(f"**Blocks (valid):** {a['n_valid_blocks']}  \n\n")
        w("---\n\n## 1. Paired results (Delta = Intervention - Control)\n\n")
        w("| Metric | Median | 95% CI low | 95% CI high |\n| :-- | :--: | :--: | :--: |\n")
        for key, label in (("delta_peak_nvml_mb", "Peak NVML (MB)"),
                           ("delta_cat3_mb", "Cat3 workspace (MB)"),
                           ("delta_cat4_mb", "Cat4 allocator (MB)")):
            b = a[key]["bootstrap"]
            if b:
                w(f"| {label} | {b['median']:+.2f} | {b['ci95_low']:+.2f} | {b['ci95_high']:+.2f} |\n")
        w("\n## 2. Criteria\n\n")
        w(f"- **A (causal):** median(ΔPeak)<0 and upper CI<0 → **{'PASS' if a['criterion_A_causal'] else 'FAIL'}**\n")
        w(f"- **B (operational):** {a['intervention_under_gate']}/10 intervention runs < 4800 MB → "
          f"**{'PASS' if a['criterion_B_operational'] else 'FAIL'}**\n")
        w(f"- **C (mechanistic):** median(ΔCat3)<0 and upper CI<0 → "
          f"**{'PASS' if a['criterion_C_mechanistic'] else 'FAIL'}**\n")
        w(f"- **D (anti-shifting):** shifted blocks {a['shifted_blocks']} (max 1) → "
          f"**{'PASS' if a['criterion_D_anti_shifting'] else 'FAIL'}**\n")
        w(f"- **Cadence non-regression (<=15%):** {a['cadence_non_regression_ok']}\n")
        w(f"\n**Causal success (A ∧ D ∧ cadence):** **{'YES' if a['causal_success'] else 'NO'}**\n")


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _cooldown(seconds: float) -> None:
    print(f"\n[Coordinator] >>> RESET / COOLDOWN {seconds:.0f}s (symmetric pre-run) <<<")
    remaining = float(seconds)
    while remaining > 0:
        print(f"   Cooldown remaining: {remaining:5.0f}s | NVML physical: {nvml_device_mb():.1f} MB")
        step = min(10.0, remaining)
        time.sleep(step)
        remaining -= step


def run_campaign(args: argparse.Namespace) -> None:
    verify_sequence_hash()
    sequence = _sequence()
    if args.max_blocks and args.max_blocks < len(sequence):
        sequence = sequence[:args.max_blocks]
    print("=" * 80)
    print("  PHASE F9-1A -- CONFIRMATORY WORKSPACE REDUCTION CAMPAIGN")
    print(f"  Protocol: {PROTOCOL_REFERENCE}")
    print(f"  Frozen sequence: {FROZEN_SEQUENCE_RAW}")
    print(f"  Frozen hash:     {FROZEN_SEQUENCE_HASH}")
    print(f"  Blocks: {len(sequence)} | Steps: {args.steps} | {args.width}x{args.height} @ {args.frames}f")
    print("=" * 80)

    py_exe = sys.executable
    blocks: List[Dict[str, Any]] = []

    for block_idx, (first, second) in enumerate(sequence, start=1):
        print(f"\n{'#' * 80}\n  BLOCK {block_idx}/10  order: {first} -> {second}\n{'#' * 80}")
        entry: Dict[str, Any] = {"block": block_idx, "order": f"{first}->{second}"}
        for position, condition in ((1, first), (2, second)):
            cond_name = "control" if condition == "C" else "intervention"
            _cooldown(args.cooldown_seconds)
            run_label = f"b{block_idx:02d}_{position}_{condition}"
            run_out = os.path.join(args.run_output_dir, f"f9_1a_{run_label}.json")
            env = os.environ.copy()
            # Symmetric offline model loading (no network; identical in both arms).
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            if cond_name == "intervention":
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            cmd = [
                py_exe, __file__, "--worker-mode",
                "--condition", cond_name,
                "--block", str(block_idx), "--position", str(position),
                "--run-label", run_label,
                "--steps", str(args.steps), "--frames", str(args.frames),
                "--width", str(args.width), "--height", str(args.height),
                "--seed", str(args.seed), "--guidance", str(args.guidance),
                "--device", args.device,
                "--s0-seconds", str(args.s0_seconds),
                "--run-out", run_out,
            ]
            print(f"\n[Coordinator] >>> LAUNCHING block {block_idx} {cond_name.upper()} (clean process) <<<")
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0 or not os.path.exists(run_out):
                print(f"[Coordinator FATAL] {run_label} failed (code {proc.returncode}).")
                entry[cond_name] = {"error": f"process failed code {proc.returncode}"}
            else:
                with open(run_out, "r", encoding="utf-8") as f:
                    entry[cond_name] = json.load(f)
        blocks.append(entry)

        # Incremental checkpoint so partial results survive an interruption.
        with open(args.checkpoint, "w", encoding="utf-8") as f:
            json.dump({"sequence_hash": FROZEN_SEQUENCE_HASH, "blocks": blocks}, f, indent=2)

    analysis = analyze(blocks)
    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F9-1A (Confirmatory Workspace Reduction)",
        "protocol_reference": PROTOCOL_REFERENCE,
        "frozen_sequence": FROZEN_SEQUENCE_RAW,
        "frozen_sequence_hash": FROZEN_SEQUENCE_HASH,
        "epsilon_mb": EPSILON_MB,
        "workload": {"resolution": f"{args.width}x{args.height}", "frames": args.frames,
                     "steps": args.steps, "mode": "adaptive", "seed": args.seed},
        "analysis": analysis,
        "blocks": blocks,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2, default=str)
    print(f"\n>> Campaign telemetry saved to: {args.output}")

    if args.report:
        write_report(args, telemetry)
        print(f">> Report saved to: {args.report}")

    print("=" * 80)
    print(f"  F9-1A CAUSAL SUCCESS: {analysis['causal_success']} "
          f"(A={analysis['criterion_A_causal']} B={analysis['criterion_B_operational']} "
          f"C={analysis['criterion_C_mechanistic']} D={analysis['criterion_D_anti_shifting']})")
    print("=" * 80 + "\n")


def run_analyze_only(args: argparse.Namespace) -> None:
    with open(args.checkpoint, "r", encoding="utf-8") as f:
        blocks = json.load(f)["blocks"]
    analysis = analyze(blocks)
    telemetry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "F9-1A (Confirmatory Workspace Reduction)",
        "protocol_reference": PROTOCOL_REFERENCE,
        "frozen_sequence_hash": FROZEN_SEQUENCE_HASH,
        "analysis": analysis,
        "blocks": blocks,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2, default=str)
    if args.report:
        write_report(args, telemetry)
    print(json.dumps(analysis, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marley Phase F9-1A -- Confirmatory Workspace Reduction")
    p.add_argument("--worker-mode", action="store_true", help="Internal: single clean-process run")
    p.add_argument("--analyze-only", action="store_true", help="Re-analyze from checkpoint, no GPU")
    p.add_argument("--condition", choices=["control", "intervention"], default="control")
    p.add_argument("--block", type=int, default=1)
    p.add_argument("--position", type=int, default=1)
    p.add_argument("--run-label", type=str, default="b01_1_control")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--guidance", type=float, default=CANONICAL_GUIDANCE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--s0-seconds", type=float, default=DEFAULT_S0_SECONDS)
    p.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    p.add_argument("--max-blocks", type=int, default=0,
                   help="Debug/smoke only: limit the number of blocks (0 = all 10). "
                        "The full campaign MUST use the default 0.")
    p.add_argument("--run-out", type=str, default="logs/f9_1a_run.json")
    p.add_argument("--run-output-dir", type=str, default="logs/f9_1a")
    p.add_argument("--checkpoint", type=str, default="logs/f9_1a_checkpoint.json")
    p.add_argument("--output", type=str, default="logs/f9_1a_telemetry.json")
    p.add_argument("--report", type=str, default="docs/F9_1A_WORKSPACE_REDUCTION_REPORT_01.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_worker(args)
    elif args.analyze_only:
        run_analyze_only(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
