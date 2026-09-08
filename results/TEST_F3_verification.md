# Technical Report — F3 Final Verification (Measurement-Corrected)
## Certified Result: Phase F3 PASS · Ready for F3 + INT8

**Date:** 2026-09-08  
**Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,144 MB total)  
**Host:** Windows 11 (WDDM concurrency active)  
**Model:** Wan2.1-T2V-1.3B-Diffusers (30 DiT blocks, FP16)  
**Execution:** `BudgetedAsyncStreamer` double-buffered async scheduler (dual CUDA streams + bidirectional event ownership)  
**Telemetry Artifact:** [`logs/f3_verification_benchmark.json`](../logs/f3_verification_benchmark.json)  
**Methodology:** [`docs/F3_MEASUREMENT_VERIFICATION_03.md`](../docs/F3_MEASUREMENT_VERIFICATION_03.md) · Changes 8–10: [`docs/F3_CALIBRATION_CHANGES_02.md`](../docs/F3_CALIBRATION_CHANGES_02.md)  
**Status:** 🟢 **CERTIFIED PASS** (5-step × 30-block × 5 A/B repetitions)

---

## 1. Purpose

This run executes the measurement-integrity verification protocol agreed with the external
technical consultant (Enmiendas 1–3) to answer:

> **¿El +10,4% es estable, reproducible y correctamente medido?**

Answer established by repeated A/B execution with corrected instrumentation (real
overlap/stall measurement, external wall-clock as official metric, alternating order +
warmup).

---

## 2. Configuration (frozen)

| Variable | Value |
| :--- | :--- |
| Script | `f3_async_scheduler_benchmark.py` |
| Steps / Frames / Precision | 5 · 17 (480p shapes) · FP16 |
| F3-A reps | 3 |
| F3-B reps | 5 (order A/B ↔ B/A alternated) |
| Warmup | 1 discarded Sync + Async pass |
| Model / blocks | Wan2.1-T2V-1.3B · 30 DiT blocks |
| Official metric | External wall-clock (`perf_counter` bracketing each condition) |

---

## 3. Result — External wall-clock speedup (official)

| Metric | Marley Sync (A) | Marley Async (B) |
| :--- | :---: | :---: |
| **Mean wall-clock** | 18,127.4 ms | **16,824.1 ms** |
| (min / max) | 17,761.7 / 18,381.6 ms | 16,490.7 / 17,145.6 ms |
| (std dev) | ±245.9 ms | ±290.6 ms |
| **Speedup mean** | — | **+7.2%** |
| (min / max / std) | — | +6.0% / +8.4% / ±1.1% |

All 5 repetitions were positive (no regression); distribution is tight (σ ±1.1 pp),
confirming a **stable, reproducible improvement**.

Internal vs. external timing agreed within milliseconds (external − internal < ~7 ms each
run), confirming correct wall-clock bracketing.

---

## 4. Result — Effective overlap (measured, not artifact)

**100.0% in all 5 repetitions.** This value is now **empirically backed** by per-block GPU
timelines (`per_rep[*].timeline`):

| Signal (per block) | Measured |
| :--- | :---: |
| **Copy (H2D prefetch)** | ~40–44 ms |
| **Compute (forward)** | ~540–580 ms |
| **Stall (compute idle waiting on copy)** | **0.00 ms** (all blocks, all reps) |
| Stall total | 0.00 ms |

Because each prefetch copy (~43 ms) is ~12× shorter than the block it feeds (~550 ms), the
double-buffer prefetch always completes well before compute needs it → **zero GPU stall**.
The 100% overlap is therefore a genuine measured property of the schedule, not a formula
artifact. The earlier "100%" reported without this instrumentation is superseded by this
measurement.

---

## 5. Result — Kill Gates

| Gate | Threshold | Measured (this run) | Verdict |
| :--- | :---: | :---: | :---: |
| Wall-clock speedup (min over reps) | > 0% | **+6.0%** (mean +7.2%) | 🟢 **PASS** |
| Effective overlap (min over reps) | ≥ 5% | **100.0%** (measured) | 🟢 **PASS** |
| Peak VRAM (NVML, max over reps) | ≤ 4,800 MB | **2,584 MB** (≤ 4,000 excellent) | 🟢 **PASS** |
| NaN / Inf | none | **0 / 0** | 🟢 **PASS** |
| Forced stream syncs | minimize | 150 → **5** (1 per step) | 🟢 **PASS** |
| F3-A consistency | low std dev | ±232.9 ms | 🟢 **PASS** |

---

## 6. A/B Comparison (mean across 5 reps)

| Metric | Marley Sync (A) | Marley Async (B) | Delta |
| :--- | :---: | :---: | :--- |
| Wall-clock (5 steps) | 18,127.4 ms | **16,824.1 ms** | **−1,303.3 ms (+7.2%)** |
| H2D transfer total | ~1,305 ms | ~1,265 ms (GPU-measured) | concurrent with compute |
| Effective overlap | 0.0% | **100.0%** (0 ms stall) | full latency hiding |
| Peak VRAM (NVML) | — | 2,584 MB | within 4,000 MB target |
| Forced syncs | 150 | **5** | −96.7% |
| NaN / Inf | 0 | **0** | clean |

---

## 7. Verdict & Next Steps

- **F3-A (Scheduler Validation):** 🟢 **PASS**
- **F3-B (Real Wan2.1 DiT Execution Performance Certification):** 🟢 **PASS**
- **Phase F3 Overall:** 🟢 **PASS — CERTIFIED**

> The initial single-run figure of +10.4% was an optimistic sample; the corrected,
> reproducible figure is **+7.2% (range +6.0% to +8.4%)**, a stable and correctly measured
> improvement attributable to the async scheduler under an unchanged 4.8 GB budget.

**Next phase:** 🟢 **F3 + INT8** (advance authorized by certified PASS).

```text
F3 (FP16) PASS → F3 + INT8 → F4 Adaptive Decision Engine
```
