# Experimental Plan — Phase F4 (Adaptive Memory Decision Engine · CORE)
## Design Plan · Dynamic FP16/INT8, prefetch, and residency selection under pressure

**Phase:** F4 — Adaptive Memory Decision Engine *(CORE SYSTEM)*
**Date:** 2026-09-08
**Status:** 🟡 **PLANNED — design only; no code yet**
**Baseline inputs (frozen):**
- F3 FP16 (certified baseline): +7.2% Async-vs-Sync · 2,584 MB · overlap 100%.
- F3 + INT8 (frozen baseline): +13.2% Async-vs-Sync · 46.5 MB/block (−49.9%) · 2,076 MB · cos ≥ 0.9999.
  Acta de cierre: [`docs/F3_INT8_CLOSURE_01.md`](F3_INT8_CLOSURE_01.md).
**Consultant guidance:** closure letter recommending F4 (dynamic strategy selection) and the
new observable bottleneck (dequantization on the critical path). Binding directive §5/§6:
[`docs/F3_F4_TECHNICAL_OPINION_01.md`](F3_F4_TECHNICAL_OPINION_01.md).
**Existing primitives to reuse (DO NOT rewrite):**
- [`marley/ops/async_stream.py`](../marley/ops/async_stream.py) — `BudgetedAsyncStreamer` (Sync/Async, double-buffer, `StreamMetrics`).
- [`marley/ops/async_stream_int8.py`](../marley/ops/async_stream_int8.py) — `INT8BudgetedStreamer` (INT8 transfer + on-device dequant, FP16 compute).

---

## 1. Objective

F4 is not about proving a single technique. It must make Marley **select the right
strategy dynamically** per denoise step (or checkpoint) given observed memory pressure,
PCIe latency, compute time, prefetch state, and dequantization cost.

Prior phases supplied the *levers* and *measurement*; F4 supplies the *decision loop*.

## 2. Available decision levers (built in prior phases)

| Lever | Domain | Values | Provided by |
| :--- | :--- | :--- | :--- |
| Weight precision | Transfer volume | FP16 / INT8 | F1.7, F3+INT8 |
| Prefetch aggressiveness | Overlap | aggressive / conservative / off | F3 async vs sync |
| Residency | VRAM | keep-on-GPU / prefetch / evict | F1, F3 double-buffer |

## 3. Observed state (decision inputs)

Must be read cheaply, at block or per-step granularity, reusing existing telemetry
(`StreamMetrics` + NVML sampler):

1. Free / used physical VRAM headroom (NVML).
2. Recent cumulative PCIe H2D transfer latency.
3. Layer compute elapsed time (per-block, from event timelines).
4. Exponential moving average (EMA) of memory pressure (background WDDM).
5. **Prefetch state / stall** — new: measured stall reveals when prefetch stops hiding cost.
6. **Dequantization cost** — new: the critical-path dequant observed in F3+INT8 (~324 ms
   total stall); becomes an explicit decision signal for F4.

## 4. Decision signals → action mapping (rule sketch)

Decision signal and the action it should trigger:

```text
Pressure de VRAM (alto)   -> INT8 + evict + prefetch conservador
  H2D volume (domina)     -> FP16? no: menor volumen si ya INT8 -> ajustar prefetch
  Stall de prefetch (crece) -> prefetch mas agresivo / antelacion mayor
  Coste dequant (crece)   -> anticipar/solapar dequant (no ad hoc en F3+INT8; disenar en F4)
  Compute-bound (stall 0) -> no hace falta prefetch agresivo; ahorrar VRAM
```

The engine may pick a **profile** (below) and, within it, tune per-step values.

## 5. Operational profiles (binding directive §5/§6)

- **Performance Profile:** aggressive prefetch + INT8 DiT projections + maximal allowable
  VRAM utilization → minimum generation latency.
- **Memory Safe Profile:** conservative prefetch + early eviction + large safety headroom
  (+1.5 GB) to withstand sudden background WDDM pressure.

Decision granularity is **block/layer** (avoid per-tensor scheduling overhead per directive).

## 6. Execution model

- Ahead-Of-Time (AOT) static plan generated following Step-1 warmup.
- Periodic re-evaluation checkpoints every `N` diffusion steps, **or** on abrupt memory
  pressure deviation (event-triggered), per ROADMAP F4.
- On a checkpoint, choose precision + prefetch + residency for the next window.

## 7. Evaluation against static baselines (kill-gate design)

The engine must be validated against the **best static policy** available at each condition,
not against an arbitrary number. Candidate static policies to compare:

| Static policy | Ref result (frozen) |
| :--- | :--- |
| Sync FP16 | F3 baseline |
| Async FP16 | +7.2% vs Sync · 2,584 MB |
| Async INT8 | +13.2% vs Sync · 2,076 MB |

**Kill gate (ROADMAP F4):** if the adaptive engine does not outperform the **best static
policy** by ≥ 5% in memory headroom **or** execution speed, simplify to a fixed policy.
The comparison must be **same-session** (see §8) to remove thermal/clock confound.

## 8. Mandatory same-session A/B/C methodology (consultant recommendation §10)

Before any absolute claim, run the three conditions **in one controlled session**:

```text
A = Sync FP16
B = Async FP16
C = Async INT8
```

This separates scheduler benefit (B vs A), INT8 benefit (C vs B), and session thermal/clock
variance. F4 gains are then expressed only relative to the same-session best static policy
(e.g. Async INT8 = condition C), never as cross-session absolute deltas.

## 9. Metric scorecard (reuse F3 7-metric protocol)

Wall-clock (external, official) · cumulative H2D · effective overlap (empirical) · peak
NVML (≤ 4,800; target ≤ 4,000) · real stall · forced syncs · throughput (s/frame). Plus:
- Decision correctness (policy matched intended profile).
- Switch latency / overhead (cost of re-planning).

## 10. Scope control (freeze discipline)

F4 introduces **only the decision loop**. It reuses the frozen F3 and F3+INT8 execution
primitives unchanged. In-scope: observer + profiler switch + policy selector + re-planning
checkpoints. Out-of-scope during F4: NF4, scheduler rewrites, ad-hoc dequant tweaks.

## 11. Kill gates

| Gate | Threshold |
| :--- | :---: |
| VRAM headroom / speed vs best same-session static | ≥ 5% |
| Peak VRAM (NVML) | ≤ 4,800 MB |
| NaN / Inf | none |
| Policy correctness | selected profile == intended |

Fallback: **deterministic static block policy** (return to Async INT8 / Performance).

## 12. Sequencing

```text
F3 FP16 (frozen) ─► F3 + INT8 (frozen baseline) ─► [F4 decision engine] ─► F6 (480p/33f)
                                        └──── same-session A/B/C validation
```

Primary scalability target after F4 stabilizes: **480p / 33 frames**; 720p remains a
secondary stretch milestone (F6).

## 13. Artifacts (to be produced during F4 implementation)

- `marley/core/adaptive.py` — decision engine (profile selector, observer, planner).
- `marley/core/policies.py` — FP16/INT8, prefetch, residency strategy objects.
- `f4_adaptive_benchmark.py` — runner incl. same-session A/B/C.
- `logs/f4_adaptive_benchmark.json`, `results/TEST_F4_adaptive_benchmark.md`, registry row.

## 14. Open questions to resolve at implementation kickoff

- Exact `N` (re-evaluation period) and the pressure-deviation trigger threshold.
- Whether per-step INT8↔FP16 switching requires weight re-quantize on the fly or a
  pre-computed dual representation on host (avoids re-quantization latency).
- Whether dequant should be prefetched/overlapped within F4 (design decision, not ad hoc).
