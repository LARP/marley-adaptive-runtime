# TEST F4 — Adaptive Memory Decision Engine (Preliminary Validation)

**Phase:** F4 — Adaptive Memory Decision Engine *(CORE SYSTEM)*
**Date:** 2026-09-08
**Status:** 🟡 **PRELIMINARY** — end-to-end smoke validation PASS; not yet a certified benchmark.
**Authorized contract:** [`docs/F4_TEST_SPEC_01.md`](../docs/F4_TEST_SPEC_01.md) (v3, approved by Director).
**Runner:** [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py) · **Telemetry:** [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json).

> This report is a **functional smoke-test marker**. It validates that the F4 decision loop
> (`marley/core/adaptive.py` + `marley/core/policies.py`) executes end-to-end against the
> frozen F3/F3+INT8 primitives in a same-session A/B/C/D. The single-rep, 3-step numbers are
> **NOT statistically defensible** and MUST NOT be read as certified F4 results. A full
> `--reps ≥ 3` run and the `--pressure-test` (Adaptive Gate) are required before any claim.

---

## 1. Objective & Scope

F4 adds a **decision loop** (observer + profile selector + AOT planner + hysteresis checkpoints)
that selects `precision × prefetch × residency` per window and delegates execution to the frozen
`BudgetedAsyncStreamer` (FP16) / `INT8BudgetedStreamer` (INT8). It does **not** re-implement
transfer/compute, and it must leave `marley/ops/async_stream*.py` unmodified (frozen).

Artifacts produced for F4:
- [`marley/core/policies.py`](../marley/core/policies.py) — pure `PolicyDecision` / `RuntimeState` / `PolicySelector`.
- [`marley/core/adaptive.py`](../marley/core/adaptive.py) — `AdaptiveEngine` (EMA observer + hysteresis).
- [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py) — same-session A/B/C/D + `--pressure-test`.

---

## 2. Command Run

```
.venv\Scripts\python.exe f4_adaptive_benchmark.py --abc --steps 3 --reps 1 --window 5 --no-warmup
```

- Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU · total VRAM 6,144 MB.
- 30 DiT blocks · profile `performance` · decision window N=5 · 3 denoise steps (single window for D).
- Warmup: disabled (smoke run).

> **Note:** model was already HF-cached (Wan2.1-T2V-1.3B transformer). First invocation hit a
> host pinned-memory OOM due to duplicated streamers; the runner was refactored so A/B/C/D all
> reuse the engine's FP16 and INT8 streamers (one pre-computed pinned representation each),
> which removed the duplication. Re-run completed cleanly.

---

## 3. Same-Session Scorecard (A/B/C/D)

| Condition | Strategy | Wall (ms) | Overlap | Peak VRAM (MB) | NaN/Inf |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A** | Sync FP16 | 10,870.3 | 0.0% | 2,200 | 0/0 |
| **B** | Async FP16 | **9,599.7** | 100.0% | 2,200 | 0/0 |
| **C** | Async INT8 | 9,930.3 | 97.7% | 2,778 | 0/0 |
| **D** | Adaptive (engine) | 9,991.7 | 97.2% | ≤ 2,778 | 0/0 |

**Best same-session static:** B (Async FP16), 9,599.7 ms.
**D vs best static:** **−4.08%** (i.e. D slower by ~4% on this single 3-step sample).
**Decision overhead (D):** 0.01 ms total · **switch_count:** 0 · **replan_count:** 0 (static `performance` AOT in ABC mode).

---

## 4. Gate Verdicts (F4_TEST_SPEC v3 §4.4)

| Level | Gate | Threshold | Result | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Hard Gate** | Peak VRAM (NVML) | ≤ 4,800 MB | 2,778 MB | 🟢 **PASS** |
| **Hard Gate** | NaN / Inf | 0 | 0/0 | 🟢 **PASS** |
| **Hard Gate** | Policy correctness | selected == intended | performance → int8/aggressive/keep | 🟢 **PASS** |
| **Engineering Target** | Peak VRAM (NVML) | ≤ 4,000 MB | 2,778 MB | 🟢 **MET** |
| **Optimization Target** | D vs `best(A,B,C)` | ≥ 5% | −4.08% | 🟡 **NOT MET** (target only, not a gate) |

**Overall preliminary verdict: PASS (functional smoke).** Safety Gate holds. The Optimization
Target is *not* met on this 1-rep/3-step sample — expected and **not a validity failure** per the
approved contract (≥5% is an optimization objective; the fallback to a fixed policy is only
considered after a full evaluation by the Director, never automatically).

> **Interpretation caution:** with `window=5` ≥ `steps=3`, D executed a single static
> `performance` (INT8 async) window, so this sample shows D ≈ C with added window-level control
> overhead. It neither demonstrates nor refutes adaptive benefit; that requires more steps, reps,
> and the injected-pressure scenario.

---

## 5. Unit / offline validation (no GPU)

- `PolicySelector`: `performance/high` → int8/aggressive/keep · `memory_safe/low` → fp16/conservative/evict ·
  `memory_safe/high` → int8/off/evict; invalid `residency` values rejected (3-state enforced).
- Hysteresis monitor (simulated 2600 → 3200 spike → 2400 release): sequence `PPSSSPPP` —
  entered `memory_safe`, honored `MIN_DWELL_WINDOWS`, exited on `PRESSURE_LOW`, **no oscillation**;
  `replan_count = 2` (enter + exit).

---

## 6. Artifacts

| Artifact | Path | Status |
| :--- | :--- | :--- |
| Decision policies | [`marley/core/policies.py`](../marley/core/policies.py) | ✅ created |
| Decision engine | [`marley/core/adaptive.py`](../marley/core/adaptive.py) | ✅ created |
| Benchmark runner | [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py) | ✅ created |
| Telemetry | [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json) | ✅ written |
| Frozen primitives untouched | `marley/ops/async_stream*.py` | ✅ unchanged |

---

## 7. Next Steps (required before certifying F4)

1. `python f4_adaptive_benchmark.py --abc --steps 30 --reps 3` → certified same-session A/B/C/D
   (Safety + Optimization over alternating-order reps, warmup discarded).
2. `python f4_adaptive_benchmark.py --pressure-test` → Adaptive Gate (deterministic injected pressure,
   hysteresis stability, headroom recovery ≥ +1.5 GB).
3. Decide engine retention vs fixed-policy fallback based on measured overhead vs benefit (Director).
4. Update ROADMAP F4 status and registry row once certified.

---

*Marley Runtime — Fase F4 · en memoria de Marley 🐾.*
