# F7-D4 (Reduced) Multi-Step Seam-Release Validation -- Report

**Date:** 2026-09-09T04:15:30.439811  
**Workload:** 1280x720 @ 33 frames, 10 steps, mode `adaptive`  
**Intervention:** empty_cache at every cond->uncond seam (single variable, from F7-D3).

---

## 1. Result

- Peak NVML: 4708.9 MB (D3 single-step: 4403.5)
- Peak Reserved: 3776.0 MB
- Reserved accumulation (last-first step-end): 248.0 MB
- Denoise: 598.91 s; cadence 59.89 s/step; seam releases 10
- VAE decode: 69.57 s; NaN/Inf False; mp4 True
- Classification: **FAVORABLE (multi-step peak <=4800, VAE ok)**

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

## 3. Governance

Reduced multi-step causal validation in an isolated runner. Does NOT authorize runtime integration; full 30-step / E2E remain pending.
