# F7-D3 Cond->Uncond Seam Release -- Causal Report

**Date:** 2026-09-09T03:33:40.215494  
**Workload:** 1280x720 @ 33 frames, 1 DiT step, mode `adaptive`  
**Variable:** controlled empty_cache at the cond->uncond seam (only).

---

## 1. Seam snapshots (OBSERVED)

| Snapshot | NVML (MB) | Alloc (MB) | Reserved (MB) | FreePool (MB) | Seg(cur/alloc/freed) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| pre_release | 4230.64 | 436.71 | 3426.0 | 2989.29 | 41/41/0 |
| post_release | 1546.64 | 436.71 | 742.0 | 305.29 | 23/41/18 |

empty_cache duration: 148.1 ms

## 2. Uncond per-block Reserved after release (DERIVED)

| Pass | Blk | PreRes (MB) | PostRes (MB) |
| :--- | :---: | :---: | :---: |
| uncond | 00 | 940 | 3196 |
| uncond | 01 | 3196 | 3196 |
| uncond | 02 | 3196 | 3196 |
| uncond | 03 | 3196 | 3196 |
| uncond | 04 | 3196 | 3196 |
| uncond | 05 | 3196 | 3196 |
| uncond | 06 | 3196 | 3196 |
| uncond | 07 | 3196 | 3196 |
| uncond | 08 | 3196 | 3196 |
| uncond | 09 | 3196 | 3196 |
| uncond | 10 | 3196 | 3196 |
| uncond | 11 | 3196 | 3196 |
| uncond | 12 | 3196 | 3196 |
| uncond | 13 | 3196 | 3196 |
| uncond | 14 | 3196 | 3196 |
| uncond | 15 | 3196 | 3196 |
| uncond | 16 | 3196 | 3196 |
| uncond | 17 | 3196 | 3196 |
| uncond | 18 | 3196 | 3196 |
| uncond | 19 | 3196 | 3196 |
| uncond | 20 | 3196 | 3196 |
| uncond | 21 | 3196 | 3196 |
| uncond | 22 | 3196 | 3196 |
| uncond | 23 | 3196 | 3196 |
| uncond | 24 | 3196 | 3196 |
| uncond | 25 | 3196 | 3196 |
| uncond | 26 | 3196 | 3196 |
| uncond | 27 | 3196 | 3196 |
| uncond | 28 | 3196 | 3196 |
| uncond | 29 | 3196 | 3196 |

## 3. Result

- Peak NVML: 4403.5 MB (D2c no-release: 6006.2)
- Peak Reserved: 3576.0 MB (D2c: 5682.0)
- Reserved after uncond block0: 3196.0 MB (regrowth vs cond end: 150.0 MB)
- Classification: **FAVORABLE (peak <= 4800)**

## 4. Governance

Causal experiment in an isolated runner. Does NOT authorize runtime modification.
