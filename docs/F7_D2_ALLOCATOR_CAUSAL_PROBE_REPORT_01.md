# F7-D2 Allocator Causal Release Probe -- Pilot Report

**Date:** 2026-09-09T02:57:58.538438  
**Workload:** 1280x720 @ 33 frames, 1 DiT step, mode `adaptive`  

**Evidence:** `MEASURED` instrumented; `OBSERVED` sampled; `DERIVED` computed; `HYPOTHESIS` not demonstrated.

---

## 1. Boundary Snapshots (OBSERVED)

| Snapshot | NVML (MB) | Alloc (MB) | Reserved (MB) | Delta (MB) | Seg(cur/alloc/freed) | RAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| retained | 1240.5 | 102.3 | 546.0 | 443.7 | 7/54/47 | 381.95 |
| released_rep1 | 1240.87 | 102.3 | 546.0 | 443.7 | 7/54/47 | 382.02 |
| released_rep2 | 1241.06 | 102.3 | 546.0 | 443.7 | 7/54/47 | 382.04 |

## 2. Derived Deltas

- Delta Reserved (retained vs released_rep1): **0.0 MB**
- Delta NVML (retained vs released_rep1): **-0.37 MB**
- Segments freed between snapshots: **0**
- NVML reproducibility spread (released reps): **[1240.87, 1241.06]**
- Interpretation: **Case D candidate (pool not releasable at this boundary)**

## 3. Memory Peaks (OBSERVED)

- Peak physical NVML: 6078.5 MB (F7-0 reference 6058.5 MB)
- Peak PyTorch Allocated: 2147.9 MB (F7-0 2187.5 MB)
- Peak PyTorch Reserved: 5778.0 MB (F7-0 5894.0 MB)

## 4. Interpretation & Next Decision

This is a diagnostic causal micro-test at the DiT->VAE boundary. It does NOT by itself authorize any permanent runtime change. The interpretation above informs the Director whether H1 (allocator retention as a physical driver) gains causal support or must be revised toward a WDDM/driver-residency explanation.
