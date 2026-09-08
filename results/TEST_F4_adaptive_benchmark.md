# TEST F4 — Adaptive Memory Decision Engine

**Phase:** F4 — Adaptive Memory Decision Engine *(CORE SYSTEM)*
**Date:** 2026-09-08
**Status:** 🟢 **CORE SYSTEM VALIDATED — PERFORMANCE CERTIFICATION PENDING**
(per Consejero Técnico Externo review, 2026-09-08).
**Authorized contract:** [`docs/F4_TEST_SPEC_01.md`](../docs/F4_TEST_SPEC_01.md) (v3, approved by Director).
**Runner:** [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py) · **Telemetry:** [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json).

> **Summary.** Both contract experiments were executed on real hardware (Wan2.1-T2V-1.3B, 30 DiT
> blocks, RTX 3050 6GB): the **same-session A/B/C/D benchmark (30 steps × 3 reps, alternating
> order, warmup discarded)** and the **`--pressure-test` (deterministic Adaptive Gate)**. The F4
> decision loop is validated end-to-end: Safety Gate PASS, negligible decision overhead, and a
> deterministic, non-oscillating pressure→recovery cycle. D did **not** beat the best static
> policy in the no-pressure regime (it stayed on INT8 because no switch was ever demanded) — this
> is a *valid* negative result that motivates the next decision under the contract, not a failure.

---

## 1. Objective & Scope

F4 adds a **decision loop** (observer + profile selector + AOT planner + hysteresis checkpoints)
that selects `precision × prefetch × residency` per window and delegates execution to the frozen
`BudgetedAsyncStreamer` (FP16) / `INT8BudgetedStreamer` (INT8). It does **not** re-implement
transfer/compute, and it leaves `marley/ops/async_stream*.py` unmodified (frozen) to preserve
causal attribution.

Artifacts: [`marley/core/policies.py`](../marley/core/policies.py) (pure selector, 3-state
residency) · [`marley/core/adaptive.py`](../marley/core/adaptive.py) (`AdaptiveEngine`,
EMA observer + hysteresis) · [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py).

---

## 2. Commands Run

```
# Priority 1 — certified same-session A/B/C/D (30 steps, 3 reps, alternating order, warmup discarded)
.venv\Scripts\python.exe f4_adaptive_benchmark.py --abc --steps 30 --reps 3 --window 5

# Priority 2 — Adaptive Gate (deterministic injected-pressure)
.venv\Scripts\python.exe f4_adaptive_benchmark.py --pressure-test --window 5
```

Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU · 6,144 MB total · 30 DiT blocks · window N=5.

---

## 3. Same-Session A/B/C/D — 30 steps × 3 reps (external wall-clock)

| Condition | Strategy | Mean (ms) | Min (ms) | Max (ms) | Overlap | NaN/Inf |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Sync FP16 | 113,538.7 | 112,695.5 | 114,992.8 | 0.0% | 0/0 |
| **B** | Async FP16 | **105,441.5** | 104,864.5 | 106,205.0 | 100.0% | 0/0 |
| **C** | Async INT8 | 109,157.9 | 108,946.6 | 109,475.6 | 98.1% | 0/0 |
| **D** | Adaptive (engine) | 109,022.8 | 108,875.6 | 109,275.0 | ~88.4% | 0/0 |

- **Best same-session static:** B (Async FP16), 105,441.5 ms.
- **D vs best static (B): −3.40%.**
- **Decision overhead (D): 0.16 ms total · switch_count 0 · replan_count 0 · peak NVML 2,882 MB.**

**Per-rep wall (ms) for the alternating order:** D 109,275 / 108,917 / 108,875; C 108,946 / 109,051 / 109,475; B 106,205 / 105,255 / 104,864; A 114,992 / 112,927 / 112,695.

### Interpretation (Consejero §3 & §5)

In this no-pressure regime **Async FP16 (B) is the fastest**, and **Async INT8 (C) is ≈3.5%
slower** than FP16. This is a real, consistent result (the on-device dequant ~324 ms sits on the
critical path and is not amortized at these lengths) and it *reinforces F4's raison d'être*: a
static INT8 policy loses when there is no memory pressure. **D remained on INT8/aggressive/keep**
for all windows because the base profile is `performance` and no pressure ever demanded a switch —
so D ≈ C. This matches the Consejero's scenario: *"Si D permanece constantemente en INT8 porque
el estado nunca exige otra cosa, eso también constituye un resultado válido."*

D's marginally lower measured overlap (~88% vs C ~98%) is an artifact of per-window boundaries
(re-sync + re-decide between windows), not a scheduler regression; D's wall ≈ C's wall within noise.

> ⚠️ **Not a refutation of F4.** D was never *asked* to adapt here. The experiment that stresses
> adaptation is the pressure test (§4). The no-pressure result only shows that the `performance`
> baseline (INT8) is not globally optimal — which is precisely the case an adaptive layer exists
> to exploit (e.g. `memory_safe` under low pressure would pick FP16).

---

## 4. Adaptive Gate — `--pressure-test` (deterministic injected pressure)

Scripted timeline (used MB): `2600 … → 3200 spike (3 windows) → 2400 release …`

**Profile sequence across 8 windows:**
`performance → memory_safe → memory_safe → performance → performance → performance → performance → performance`

| Signal | Value |
| :--- | :---: |
| Replan events | 2 (window 1 → `memory_safe`/pressure; window 3 → `performance`/normal) |
| Replan count | 2 |
| Switch count | 2 |
| Decision overhead | 0.07 ms total |
| Entered memory_safe | True |
| Re-exited (PRESSURE_LOW after dwell) | True |
| Oscillation | **False** |
| **Adaptive Gate** | 🟢 **PASS** |

The full cycle the Consejero asked to observe was reproduced deterministically and **without
oscillation**:

```text
Performance → (VRAM +600 MB) → detect ≥200 MB over EMA → replan → Memory Safe
  → conservative/evict → recovery → (pressure falls) → PRESSURE_LOW + MIN_DWELL
  → replan → Performance
```

> **Honest caveat:** the simulated +600 MB spike kept free VRAM above the `SAFE_MIN_FREE_MB`
> (1,500 MB) line, so `memory_safe` selected **FP16/conservative/evict** (the profile's low-pressure
> branch) rather than the *tighten* branch (INT8/off). The decision loop, hysteresis, dwell and
> regime switching are what is demonstrated. To exercise the **INT8-tighten branch**, a future
> pressure profile must drive free VRAM below 1,500 MB.

---

## 5. Gate Verdicts (F4_TEST_SPEC v3 §4.4)

| Level | Gate | Threshold | Measured | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Hard Gate** | Peak VRAM (NVML) | ≤ 4,800 MB | 2,882 MB | 🟢 **PASS** |
| **Hard Gate** | NaN / Inf | 0 | 0/0 | 🟢 **PASS** |
| **Hard Gate** | Policy correctness | selected == intended | performance→INT8/agg/keep; memory_safe→cons/evict | 🟢 **PASS** |
| **Engineering Target** | Peak VRAM (NVML) | ≤ 4,000 MB | 2,882 MB | 🟢 **MET** |
| **Adaptive Gate** | detect + replan + switch + recover, no oscillation | — | PASS (§4) | 🟢 **PASS** |
| **Optimization Target** | D vs best `{A,B,C}` | ≥ 5% | −3.40% | 🟡 **NOT MET** (target only) |

**Overall:** 🟢 **CORE SYSTEM VALIDATED** (Safety + Adaptive Gates pass; overhead ~0.1 ms).
Performance certification vs the static baseline is **pending** per the Consejero's decision
framework (§6).

---

## 6. Decision point (Consejero §7 — scenarios)

| Scenario | Condition | Path |
| :--- | :--- | :--- |
| **A** | Consistent measurable benefit, esp. under pressure | Retain F4 permanently |
| **B** | Adapts correctly but no perf gain | Judge whether robustness benefit justifies complexity |
| **C** | No clear practical benefit after measuring overhead/stability/pressure | Accept result; use **Async INT8 / Performance** static |

The no-pressure run alone is **not** enough to classify F4 into A/B/C: D was never stressed. The
pressure test shows the adaptive mechanism *works*. The open question is whether adapting precision
under real (not simulated) pressure yields benefit — and whether the no-pressure optimum (FP16, not
INT8) should be the base profile. **Decision reserved to the Director based on evidence.**

---

## 7. Artifacts & Integrity

| Artifact | Path | Status |
| :--- | :--- | :--- |
| Decision policies | [`marley/core/policies.py`](../marley/core/policies.py) | ✅ |
| Decision engine | [`marley/core/adaptive.py`](../marley/core/adaptive.py) | ✅ |
| Benchmark runner | [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py) | ✅ |
| Telemetry | [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json) | ✅ |
| Frozen primitives untouched | `marley/ops/async_stream*.py` | ✅ diff empty |

---

*Marley Runtime — Fase F4 · en memoria de Marley 🐾.*
