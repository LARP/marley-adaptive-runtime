# F9-1B — Confirmatory Allocator Policy Isolation — Report

**Date:** 2026-09-13T11:09:54.432682  
**Protocol:** [`docs/F9_1B_PREREGISTERED_ALLOCATOR_PROTOCOL_01.md`](F9_1B_PREREGISTERED_ALLOCATOR_PROTOCOL_01.md)  
**Frozen sequence hash:** `323fda6381c065db0c7dc72d2d665b84ae7828cb58beca52d49e6c7574580411`  
**Blocks (valid):** 10  

---

## 1. Paired results (Delta = Intervention - Control)

| Metric | Median | 95% CI low | 95% CI high |
| :-- | :--: | :--: | :--: |
| Peak NVML (MB) | +0.00 | -88.62 | +0.00 |
| Cat4 allocator (MB) | +0.00 | -0.00 | +51.98 |
| Cat3 workspace (MB) | -0.01 | -110.70 | +0.21 |

## 2. Criteria

- **A (causal):** median(ΔPeak)<0 and upper CI<0 → **FAIL**
- **B (operational):** 4/10 intervention runs < 4800 MB → **FAIL**
- **C (mechanistic):** median(ΔCat4)<0 and upper CI<0 → **FAIL**
- **D (anti-shifting):** shifted blocks 1 (max 1) → **PASS**
- **Cadence non-regression (<=15%):** True

**Causal success (A ∧ D ∧ cadence):** **NO**

---

## 3. Closure (2026-09-13)

- **Verdict:** 🔴 **NULL RESULT — NO CAUSAL EFFECT**. Criteria A, B, C FAIL; D PASS; cadence non-regression PASS.
- **Intervention validity caveat:** all intervention arms logged `UserWarning: expandable_segments not supported on this platform` (PyTorch 2.6.0+cu124, Windows 11 WDDM). The manipulated allocator policy was a functional no-op on this stack, consistent with Δ≈0 in B6–B9.
- **Operational anomalies:** S0 idle rose from ~1,000 MB (B1–B4) to ~1,327 MB (B5–B9) and 1,382 MB (B10-I); B10 control peaked at 5,217.9 MB (Cat3 2,870 MB), the highest of the campaign.
- **Conclusion:** `expandable_segments:True` does not reduce physical NVML peak or Cat4 footprint on RTX 3050 6GB Laptop / WDDM. F9-1B is **CLOSED as NULL / NOT APPLICABLE on this platform**. Next: evaluate F9-1C (cuDNN algorithm pinning / pooling) in isolation.
