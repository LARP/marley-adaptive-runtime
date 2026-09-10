# F7-D5 (Full) 30-Step End-to-End Seam-Release Validation -- Report

**Date:** 2026-09-09T21:17:38.625251  
**Workload:** 1280x720 @ 33 frames, 30 steps, mode `adaptive`  
**Intervention:** empty_cache at every step start AND cond->uncond seam (single-variable discipline from F7-D3/D4 favorable config).

---

## 1. Result

- Peak NVML: 5096.1 MB (D4 10-step: 4708.9)
- Peak Reserved: 3776.0 MB (D4: 3776.0)
- Reserved accumulation (last-first step-end): 246.0 MB
- Wall-clock total: 2667.7 s (44.46 min) | Kill gate 45 min
- Denoise: 1864.1 s; cadence 62.13 s/step; seam releases 30; step-start releases 30
- VAE decode: 68.06 s; NaN/Inf False; mp4 True
- Classification: **PARTIAL/NEGATIVE**

## 2. Per-step end-of-step Reserved (MB)

- step 1: 3520.0
- step 2: 3676.0
- step 3: 3772.0
- step 4: 3776.0
- step 5: 3772.0
- step 6: 3772.0
- step 7: 3772.0
- step 8: 3768.0
- step 9: 3768.0
- step 10: 3768.0
- step 11: 3768.0
- step 12: 3768.0
- step 13: 3768.0
- step 14: 3768.0
- step 15: 3768.0
- step 16: 3768.0
- step 17: 3768.0
- step 18: 3768.0
- step 19: 3766.0
- step 20: 3766.0
- step 21: 3766.0
- step 22: 3766.0
- step 23: 3766.0
- step 24: 3766.0
- step 25: 3766.0
- step 26: 3766.0
- step 27: 3766.0
- step 28: 3766.0
- step 29: 3766.0
- step 30: 3766.0

## 3. Governance

Full 30-step E2E causal validation in an isolated runner. If FAVORABLE, authorizes formal integration of the seam release into `marley/pipeline/end_to_end.py` gated to resolutions > 480p, preserving the 480p baseline immutable (F6).
