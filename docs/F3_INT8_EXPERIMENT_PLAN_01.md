# Experimental Plan — Phase F3 + INT8 (Isolated)
## Freeze Phase F3 (FP16) as Baseline · Quantify the Effect of Reduced Transfer Volume

**Phase:** F3 + INT8 (isolation experiment; **NOT** F4, **NOT** NF4)
**Date:** 2026-09-08
**Status:** ✅ **IMPLEMENTED & MEASURED — 🟢 PASS** (plan executed; results in [`results/TEST_F3_INT8_benchmark.md`](../results/TEST_F3_INT8_benchmark.md))
**Consultant Directive:** knowledge acquired through conversation with an AI agent · Closure letter: F3 certified +7.2% and frozen as baseline.
**Baseline evidence:** [`results/TEST_F3_verification.md`](../results/TEST_F3_verification.md)
**Baseline quantization evidence:** [`results/TEST_F1.7_selective_quantization.md`](../results/TEST_F1.7_selective_quantization.md)
**Implementation:** [`marley/ops/async_stream_int8.py`](../marley/ops/async_stream_int8.py) (`INT8BudgetedStreamer`) · [`f3_int8_scheduler_benchmark.py`](../f3_int8_scheduler_benchmark.py)
**Measured result:** Async-vs-Sync **+13.2%** (min +10.4 / max +15.4, σ ±2.5) · overlap **98.1%** · peak NVML **2,076 MB (−508 vs FP16)** · payload **−49.9%** · dequant cos ≥ 0.9999 · **0/0 NaN-Inf**

---

## 1. Purpose & Experimental Question

Phase F3 (FP16) certified a **reproducible +7.2%** external wall-clock speedup (range
+6.0% to +8.4%, σ ±1.1%) by hiding PCIe weight transfers behind DiT compute (overlap
100% measured, 0 ms stall, 2,584 MB peak NVML, 0/0 NaN-Inf).

F1.7 demonstrated that 8-bit quantization of the DiT **linear projections** (96.4% of
per-block parameters) reduces the per-block PCIe payload from **88.6 MB → 45.9 MB
(-48.2%)**.

> **Experimental question:**
> *What is the isolated effect on wall-clock, overlap, stall, VRAM and numerical
> integrity when we (a) reduce the bytes transferred per DiT block via INT8 linear
> projections AND (b) keep the proven double-buffered async prefetch unchanged?*

The async architecture already hides transfers; the hypothesis to test is whether a
~48% smaller per-block payload converts into a further, separately-quantifiable
wall-clock reduction — or whether the transfer was already fully hidden (in which case
the dominant gain will instead be **lower peak VRAM** and lower PCIe traffic, not time).

---

## 2. Scope Control (freeze discipline)

This experiment changes **exactly one independent variable** relative to the frozen F3 FP16
baseline: **the DiT linear-projection weight precision (INT8 vs FP16)**.

**In scope:**
- INT8 quantization of DiT linear projections (per F1.7, `-48.2%` payload/block).
- Reuse of the exact `BudgetedAsyncStreamer` double-buffer scheduler and its 7-metric
  scorecard with measurement-integrity (Enmiendas 1–3) unchanged.

**Explicitly OUT of scope (must NOT be introduced simultaneously):**
- NF4 / 4-bit (reserved for extreme low-memory mode; requires full temporal-drift verification).
- Phase F4 Adaptive Decision Engine.
- VAE / text-encoder changes.
- Any additional memory trick or scheduler rewrite.

Each modification must carry its own attributable result. This run isolates INT8 only.

---

## 3. Comparison Matrix

| Condition | Precision | Scheduler | Purpose |
| :--- | :--- | :--- | :--- |
| **Baseline (frozen)** | DiT FP16 | Async (F3 certified) | Reference: +7.2%, 2,584 MB |
| **F3 + INT8** | DiT linear projections INT8 | Async (identical) | Measured delta attributable to INT8 only |

> Comparison is **F3+INT8 (Async) vs. frozen F3 (Async)**, and additionally vs.
> **Sync (INT8)** to keep the Sync/Async A/B attribution intact within the same run.

---

## 4. Hypothesis (falsifiable)

Given per-block copy (~43 ms FP16) is already ~12× shorter than block compute (~550 ms),
prefetch is expected to remain fully hidden → overlap stays ~100% and stall ~0 ms.

Two candidate outcomes:

- **H-a (transfer-bound residue):** shrinking payload further shortens a not-fully-hidden
  critical path → additional measurable wall-clock speedup beyond +7.2%.
- **H-b (already hidden):** wall-clock speedup plateaus near +7.2%; the INT8 gain instead
  manifests as lower peak VRAM (2,584 MB → materially lower) and roughly halved H2D bytes.

Both are acceptable, evidence-based outcomes; the plan reports whichever is observed and
does not pre-judge.

---

## 5. Mandatory 7-Metric Scorecard (reuse, unchanged)

1. Total DiT denoising wall-clock time (**external wall-clock = official metric**).
2. Cumulative PCIe H2D transfer time (CUDA events).
3. Effective overlapped compute/transfer duration (**empirical** per-block timeline).
4. Physical peak VRAM via 50 ms NVML sampling (≤ 4,800 MB; target ≤ 4,000 MB).
5. Real measured GPU stall (delayed prefetch).
6. Count of forced stream synchronizations.
7. Throughput (seconds/frame).

Measurement-integrity (Enmiendas 1–3) apply unchanged: per-block timelines, alternating
A/B order with discarded warmup, external wall-clock as official metric, min-over-reps
gates.

---

## 6. Kill Gates

| Gate | Threshold | Verdict basis |
| :--- | :---: | :--- |
| External wall-clock speedup (min over reps) | > 0% | vs. Sync; plus report Async-INT8 vs Async-FP16 |
| Effective overlap (min over reps) | ≥ 5% | measured |
| Peak VRAM (NVML, max over reps) | ≤ 4,800 MB | ≤ 4,000 MB target |
| NaN / Inf | none | 0 / 0 |
| INT8 fidelity | cosine similarity ≥ 0.99 | per F1.7 gate |
| Forced syncs | minimize | expect ~1/step |

If INT8 shows no wall-clock benefit but no regression and lower VRAM, the phase PASSes on
VRAM/traffic grounds with time-neutral speedup (H-b), and the finding is reported
explicitly rather than masked.

---

## 7. Methodology & Required Instrumentation

1. **Quantization path:** quantize each DiT block's linear projections to INT8 with a
   retained per-tensor/per-channel scale, per the F1.7 recipe. Non-linear params remain
   FP16. Validate cosine similarity ≥ 0.99 and 0 NaN before the timed run.
2. **Dequant for compute:** dequantize on the GPU (INT8 host payload → FP16 GPU slot) so
   the **transferred byte volume** is the independent variable while compute precision and
   numerical path remain directly comparable to the FP16 baseline. (This isolates transfer
   volume; full INT8-native matmul is a separate, later question and out of scope.)
3. **Scheduler:** call the identical `BudgetedAsyncStreamer` async/sync loops. Only the
   host payload handed to `_prefetch_block_async` / `_load_block_sync` changes (≈45.9 MB vs
   88.6 MB per block) plus a per-block dequant step on device.
4. **Repetition:** same protocol as F3-B — 5 A/B reps, alternating order, discarded warmup,
   `perf_counter` bracketing, NVML 50 ms sampler.
5. **Telemetry:** per-block timeline (copy/compute/stall), H2D bytes, VRAM peak, NaN/Inf,
   forced syncs, mean/min/max/std.

> **Probe workload:** Wan2.1-T2V-1.3B DiT only (no T5-XXL/VAE), 30 blocks, 480p shapes
> `(1,16,5,60,104)`, text_len 226, dim 4096 — identical to the F3-B probe so results are
> directly comparable. Temporal scaling to 480p/33f is deferred to F4.

---

## 8. Expected Instrumentation Targets (planning estimates, NOT measured)

| Metric | F3 FP16 (certified) | F3 + INT8 (expected) |
| :--- | :---: | :---: |
| Wall-clock speedup vs Sync | +7.2% | ≥ +7.2% (H-a) or ≈ +7.2% (H-b) |
| Per-block H2D payload | 88.6 MB | ~45.9 MB (-48.2%) |
| Per-block copy (approx.) | ~43 ms | ~22 ms |
| Effective overlap | 100% | 100% (expected) |
| Stall | 0 ms | 0 ms (expected) |
| Peak VRAM (NVML) | 2,584 MB | lower (H-b signal) |
| Forced syncs | 5 (1/step) | 5 |
| NaN / Inf | 0 / 0 | 0 / 0 (verified) |

---

## 9. Artifacts (to be produced)

- **Runner:** `f3_int8_scheduler_benchmark.py` (new; derived from `f3_async_scheduler_benchmark.py`).
- **Telemetry:** `logs/f3_int8_benchmark.json`.
- **Report:** `results/TEST_F3_INT8_benchmark.md`.
- Registry row added in [`results/TEST_REGISTRY.md`](../results/TEST_REGISTRY.md).
- ROADMAP phase entry updated with measured verdict.

---

## 10. Decision Gate after Execution

After this run we evaluate, based solely on measured data:

- Does INT8 add measurable wall-clock on top of +7.2% (H-a)? → carry both gains to F4.
- Or does it plateau in time but reduce VRAM/traffic (H-b)? → record VRAM/traffic win as
  the INT8 value-add; carry the adaptive *precision knob* to F4 Performance vs Memory Safe
  profiles.

**Either way:** F4 (Adaptive Decision Engine) is built on top of this isolated result,
not merged with it.

---

## 11. Sequencing (unchanged per consultant)

```text
F3 FP16  (frozen baseline: +7.2%)
   ↓
F3 + INT8  (THIS plan — isolated measurement)
   ↓
Medir: wall-clock · H2D · overlap · stalls · VRAM · integridad
   ↓
F4 Adaptive Decision Engine
```
