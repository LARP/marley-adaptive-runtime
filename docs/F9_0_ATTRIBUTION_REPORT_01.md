# F9-0 — Preregistered Memory Peak Attribution Diagnostic — Report

**Date:** 2026-09-11T07:28:07.681727  
**Protocol:** [`docs/F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md`](F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md)  
**Workload:** 1280x720 @ 33 frames, 30 DiT steps, mode `adaptive`, seed 42  
**Frozen reference:** F7-D5 = 5096.1 MB (deficit +296.1 MB)  
**Attribution iteration:** 1  
**Campaign verdict:** **CLOSED - ATTRIBUTED**  

---

## 1. Cross-run summary

- Independent runs: 5 (exclusive: 5)
- Peak_NVML_raw_dit_mb: median 4648.8 | mean 4686.73 | min 4648.8 | max 4838.43 | std 75.85 | 95% CI ±66.49
- S0_idle_mb: median 890.28 | mean 893.83 | min 890.28 | max 900.51 | std 4.43 | 95% CI ±3.88
- Delta_induced_mb: median 3758.52 | mean 3792.9 | min 3751.02 | max 3937.92 | std 72.57 | 95% CI ±63.61
- overhead_mb: median 92.52 | mean 90.7 | min 83.4 | max 92.52 | std 3.65 | 95% CI ±3.2
- fraction_of_Delta_induced: median 1.0 | mean 1.0 | min 1.0 | max 1.0 | std 0.0 | 95% CI ±0.0

## 2. Per-run attribution (7 mutually exclusive categories)

| Run | Peak NVML | S0 | Delta_induced | Cat1 | Cat2 | Cat3 | Cat4 | Cat6 | Cat7 | Attributed | Fraction | Verdict |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-- |
| 1 | 4838.43 | 900.51 | 3937.92 | 404.97 | 196.36 | 79.25 | 3173.94 | 83.4 | 0.0 | 3937.92 | 1.0 | CLOSED - ATTRIBUTED |
| 2 | 4648.8 | 897.78 | 3751.02 | 404.97 | 187.37 | 2614.54 | 451.62 | 92.52 | 0.0 | 3751.02 | 1.0 | CLOSED - ATTRIBUTED |
| 3 | 4648.8 | 890.28 | 3758.52 | 404.97 | 187.37 | 2622.05 | 451.61 | 92.52 | 0.0 | 3758.52 | 1.0 | CLOSED - ATTRIBUTED |
| 4 | 4648.8 | 890.28 | 3758.52 | 404.97 | 187.37 | 2622.04 | 451.62 | 92.52 | 0.0 | 3758.52 | 1.0 | CLOSED - ATTRIBUTED |
| 5 | 4648.8 | 890.28 | 3758.52 | 404.97 | 187.37 | 2622.04 | 451.62 | 92.52 | 0.0 | 3758.52 | 1.0 | CLOSED - ATTRIBUTED |

> Category 5 (fragmentation) is a subset of category 4 and is NOT additive. Category 7 is the residual and is never split.

## 3. Lower bounds (per run)

| Run | LB_analitico (MB) | Min_allocated_obs (MB) | LB_operativo (MB) | Overhead (MB) |
| :--: | :--: | :--: | :--: | :--: |
| 1 | 993.54 | 416.51 | 1076.94 | 83.4 |
| 2 | 993.54 | 416.51 | 1086.06 | 92.52 |
| 3 | 993.54 | 416.51 | 1086.06 | 92.52 |
| 4 | 993.54 | 416.51 | 1086.06 | 92.52 |
| 5 | 993.54 | 416.51 | 1086.06 | 92.52 |

- `LB_analitico` = pesos_persistentes + max_i(pesos_bloque_i) + max(activaciones_simultaneas) (ESTIMATE)
- `Min_allocated_obs` = minimum observed `torch.cuda.memory_allocated` at synced points
- `LB_operativo` = `LB_analitico` + `Overhead_CUDA/Driver/Runtime`

## 4. Observer-effect cross-check (§2.5, option a)

| Run | Peak NVML (DiT) | torch max_allocated | torch max_reserved | NVML Hz | Polling mean (ms) | Consistent |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1 | 4838.43 | 2180.17 | 3776.0 | 36.82 | 0.2238 | True |
| 2 | 4648.8 | 2172.05 | 3606.0 | 37.13 | 0.0808 | True |
| 3 | 4648.8 | 2172.05 | 3606.0 | 37.25 | 0.0663 | True |
| 4 | 4648.8 | 2172.05 | 3606.0 | 37.13 | 0.0735 | True |
| 5 | 4648.8 | 2172.05 | 3606.0 | 36.93 | 0.1545 | True |

> NVML polling is read-only and allocates no GPU memory; per-sample cost is negligible vs the 20 ms interval. This cross-check is methodological evidence only and does not modify PASS/FAIL criteria or the preregistration.

## 5. Dominant component and F9-1 recommendation

- Run 1: dominant = `4_memoria_allocator` → Gestión del pool del allocator (scheduling de empty_cache, trimming de reserved).
- Run 2: dominant = `3_workspace_kernels` → Reutilización de workspace de kernels (pool de workspaces, cuDNN/cuBLAS).
- Run 3: dominant = `3_workspace_kernels` → Reutilización de workspace de kernels (pool de workspaces, cuDNN/cuBLAS).
- Run 4: dominant = `3_workspace_kernels` → Reutilización de workspace de kernels (pool de workspaces, cuDNN/cuBLAS).
- Run 5: dominant = `3_workspace_kernels` → Reutilización de workspace de kernels (pool de workspaces, cuDNN/cuBLAS).

## 6. Governance

F9-0 is observational and reversible: it does not modify the runtime, does not change the 4,800 MB gate, and does not reclassify F7-D5 / F7 Stage B / F8. F7 and F8 remain CLOSED. The next mechanism (F9-1) is selected solely from this diagnostic.
