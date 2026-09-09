# F7-D2c Per-Block Allocator Growth Diagnostic -- Report

**Date:** 2026-09-09T03:20:55.222640  
**Workload:** 1280x720 @ 33 frames, 1 DiT step, mode `adaptive`  
**Variable changed:** NONE (observational).

---

## 1. Reserved series per block (DERIVED)

| Pass | Reserved start (MB) | Reserved end (MB) | Growth (MB) |
| :--- | :---: | :---: | :---: |
| cond (blocks 0..29) | 790.0 | 3046.0 | +2256.0 |
| uncond (blocks 0..29) | 3046.0 | 5682.0 | +2636.0 |

Detailed per-block (pre/post reserved, post free-pool):

| Pass | Blk | PreRes (MB) | PostRes (MB) | FreePool (MB) |
| :--- | :---: | :---: | :---: | :---: |
| cond | 00 | 790 | 3046 | 2308 |
| cond | 01 | 3046 | 3046 | 2308 |
| cond | 02 | 3046 | 3046 | 2308 |
| cond | 03 | 3046 | 3046 | 2308 |
| cond | 04 | 3046 | 3046 | 2308 |
| cond | 05 | 3046 | 3046 | 2308 |
| cond | 06 | 3046 | 3046 | 2308 |
| cond | 07 | 3046 | 3046 | 2308 |
| cond | 08 | 3046 | 3046 | 2308 |
| cond | 09 | 3046 | 3046 | 2308 |
| cond | 10 | 3046 | 3046 | 2308 |
| cond | 11 | 3046 | 3046 | 2308 |
| cond | 12 | 3046 | 3046 | 2308 |
| cond | 13 | 3046 | 3046 | 2308 |
| cond | 14 | 3046 | 3046 | 2308 |
| cond | 15 | 3046 | 3046 | 2308 |
| cond | 16 | 3046 | 3046 | 2308 |
| cond | 17 | 3046 | 3046 | 2308 |
| cond | 18 | 3046 | 3046 | 2308 |
| cond | 19 | 3046 | 3046 | 2308 |
| cond | 20 | 3046 | 3046 | 2308 |
| cond | 21 | 3046 | 3046 | 2308 |
| cond | 22 | 3046 | 3046 | 2308 |
| cond | 23 | 3046 | 3046 | 2308 |
| cond | 24 | 3046 | 3046 | 2308 |
| cond | 25 | 3046 | 3046 | 2308 |
| cond | 26 | 3046 | 3046 | 2308 |
| cond | 27 | 3046 | 3046 | 2308 |
| cond | 28 | 3046 | 3046 | 2308 |
| cond | 29 | 3046 | 3046 | 2308 |
| uncond | 00 | 3426 | 5682 | 4932 |
| uncond | 01 | 5682 | 5682 | 4932 |
| uncond | 02 | 5682 | 5682 | 4932 |
| uncond | 03 | 5682 | 5682 | 4932 |
| uncond | 04 | 5682 | 5682 | 4932 |
| uncond | 05 | 5682 | 5682 | 4932 |
| uncond | 06 | 5682 | 5682 | 4932 |
| uncond | 07 | 5682 | 5682 | 4932 |
| uncond | 08 | 5682 | 5682 | 4932 |
| uncond | 09 | 5682 | 5682 | 4932 |
| uncond | 10 | 5682 | 5682 | 4932 |
| uncond | 11 | 5682 | 5682 | 4932 |
| uncond | 12 | 5682 | 5682 | 4932 |
| uncond | 13 | 5682 | 5682 | 4932 |
| uncond | 14 | 5682 | 5682 | 4932 |
| uncond | 15 | 5682 | 5682 | 4932 |
| uncond | 16 | 5682 | 5682 | 4932 |
| uncond | 17 | 5682 | 5682 | 4932 |
| uncond | 18 | 5682 | 5682 | 4932 |
| uncond | 19 | 5682 | 5682 | 4932 |
| uncond | 20 | 5682 | 5682 | 4932 |
| uncond | 21 | 5682 | 5682 | 4932 |
| uncond | 22 | 5682 | 5682 | 4932 |
| uncond | 23 | 5682 | 5682 | 4932 |
| uncond | 24 | 5682 | 5682 | 4932 |
| uncond | 25 | 5682 | 5682 | 4932 |
| uncond | 26 | 5682 | 5682 | 4932 |
| uncond | 27 | 5682 | 5682 | 4932 |
| uncond | 28 | 5682 | 5682 | 4932 |
| uncond | 29 | 5682 | 5682 | 4932 |

## 2. Reutilization / growth signal

- Growth during cond (30 blocks): **+2256.0 MB**
- Additional growth during uncond over cond plateau: **+2636.0 MB**
- Segments ever allocated / freed: **53 / 0**

Interpretive guidance (Escenarios A/B/C): if uncond reuses cond segments, its additional growth should be near 0 and Reserved should plateau; a large positive uncond growth indicates the second pass acquires new reserved segments rather than reusing the freed cond ones.

## 3. Memory peaks (OBSERVED)

- Peak physical NVML: 6006.2 MB
- Peak Allocated: 2048.1 MB
- Peak Reserved: 5682.0 MB

## 4. Governance

Observational diagnostic. Does NOT authorize F7-D3 or any runtime modification.
