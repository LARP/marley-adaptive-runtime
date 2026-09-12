# F9-1A — Confirmatory Workspace Reduction — Report

**Date:** 2026-09-12T14:38:28.944562  
**Protocol:** [`docs/F9_1_PREREGISTERED_WORKSPACE_REDUCTION_PROTOCOL_01.md`](F9_1_PREREGISTERED_WORKSPACE_REDUCTION_PROTOCOL_01.md)  
**Frozen sequence hash:** `e0aac00319dc2444414f63c1577fb80662ee558345a66dbb6166c6117547227e`  
**Blocks (valid):** 10  

---

## 1. Paired results (Delta = Intervention - Control)

| Metric | Median | 95% CI low | 95% CI high |
| :-- | :--: | :--: | :--: |
| Peak NVML (MB) | +294.00 | +294.00 | +294.00 |
| Cat3 workspace (MB) | +203.90 | +203.88 | +203.91 |
| Cat4 allocator (MB) | +9.62 | +9.62 | +9.66 |

## 2. Criteria

- **A (causal):** median(ΔPeak)<0 and upper CI<0 → **FAIL**
- **B (operational):** 0/10 intervention runs < 4800 MB → **FAIL**
- **C (mechanistic):** median(ΔCat3)<0 and upper CI<0 → **FAIL**
- **D (anti-shifting):** shifted blocks 1 (max 1) → **PASS**
- **Cadence non-regression (<=15%):** True

**Causal success (A ∧ D ∧ cadence):** **NO**
