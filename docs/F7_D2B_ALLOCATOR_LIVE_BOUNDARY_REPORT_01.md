# F7-D2b Allocator Live-Boundary & In-Step Profile -- Pilot Report

**Date:** 2026-09-09T03:06:37.690796  
**Workload:** 1280x720 @ 33 frames, 1 DiT step, mode `adaptive`  

---

## 1. Per-Phase Maxima (OBSERVED, during single step)

| Phase | Max Alloc (MB) | Max Reserved (MB) | Max FreePool (MB) | Max NVML (MB) | Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| init | 45.96 | 50.0 | 4.04 | 752.02 | 1364 |
| prompt_encode | 52.43 | 70.0 | 19.34 | 798.98 | 3184 |
| latent_prep | 239.26 | 262.0 | 22.74 | 1012.11 | 802 |
| rope_patch_embed | 631.2 | 674.0 | 140.8 | 1447.23 | 98 |
| cond_blocks_30 | 2130.94 | 3142.0 | 2498.94 | 3915.23 | 1708 |
| cond_unpatchify | 833.07 | 3522.0 | 2965.72 | 4264.48 | 3 |
| uncond_prep | 557.62 | 3522.0 | 2965.72 | 4264.48 | 2 |
| uncond_blocks_30 | 2147.89 | 5778.0 | 5117.99 | 6051.5 | 3703 |
| combine_scheduler_step | 584.87 | 5778.0 | 5201.04 | 6048.25 | 9 |
| seam_live_held | 592.78 | 5778.0 | 5185.22 | 6043.62 | 1 |
| seam_release | 415.62 | 880.0 | 580.37 | 1571.62 | 2 |
| done | 107.63 | 568.0 | 477.14 | 1259.62 | 10 |

## 2. Seam (OBSERVED)

- **live_held:** NVML 6043.62 MB | Alloc 592.78 MB | Reserved 5778.0 MB | FreePool 5185.22 MB
- **released:** NVML 1259.62 MB | Alloc 107.63 MB | Reserved 568.0 MB | FreePool 460.37 MB
- Delta Reserved (live_held -> released): **+5210.0 MB**
- Delta NVML (live_held -> released): **+4784.0 MB**

## 3. Memory Peaks (OBSERVED)

- Peak physical NVML: 6051.5 MB
- Peak Allocated: 2147.9 MB
- Peak Reserved: 5778.0 MB

## 4. Interpretation (DERIVED)

Phase maxima show whether Reserved tracks Allocated closely (unavoidable simultaneous live demand) or carries a persistent large free-pool cushion (potential stream/order reclaim). The live_held vs released seam quantifies the VAE-transition state. This is diagnostic; it does not authorize runtime changes.
