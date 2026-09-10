# F7-D6 — NVML Metric Attribution Probe -- Report

**Date:** 2026-09-10T00:57:03.569470  
**Workload:** 1280x720 @ 33 frames, 3 DiT steps, mode `adaptive` (VAE skipped)  
**Status:** measurement-metrology probe (per External Consultant authorizations no.1-4). Does NOT reclassify F7-D5 and does NOT change the gate.  
---

## 1. Instrument Validation Result

- **VERDICT: INSTRUMENT VALIDATED**
- Reason: S0 stable; controls consistent (<=10%); workload measurable; cudaMemGetInfo coherent. S3/S4 characterized as system residency phenomenon.

### 1.1 Milestones

- S0 idle_baseline (60 s): mean 962.2 MB (min 909.62 / max 1056.1 / median 955.94 / std 40.9 / spread 146.47)
- S1 operational_baseline (post-init): 1043.57 MB
- S2 workload_peak (device used): 4624.00390625 MB (D5: 5096.1)
- S3 post_workload mean: 1481.64 MB (spread 29.0)
- S4 final mean: 1511.74 MB (spread 86.84)
- workload increment (S2-S1): 3580.4 MB
- **workload-induced device residency delta (S2-S0): 3661.8 MB** (NOT proven ownership)

### 1.2 Conditions

- s0_stable: True
- control_a_observable: True
- control_consistent: True
- workload_measurable: True
- cudaMemGetInfo_coherent: True
- s3_returns_to_baseline: False
- s3_phenomenon: persistent_device_residency
- s4_stable: True
- control A delta: 512.0 MB | control B delta: 500.0 MB | rel diff: 2.34 %
- signal floor: 150.0 MB
- Peak Reserved (torch): 3606.0 MB (D5: 3776)

## 2. Interpretation

The device-wide `nvmlDeviceGetMemoryInfo().used` metric does not start at zero: an idle baseline (S0) is observed even with no Marley workload. `usedGpuMemory=None` per PID under WDDM means the counter is UNAVAILABLE, not 0 MB. The quantity `S2 - S0` is a **workload-induced device residency delta** (device-wide growth), NOT proven process ownership; it is cross-checked against `torch.reserved` and validated by known CUDA control workloads (A pre-workload, B post-workload).

## 3. Falsification scenarios (remain open)

1. S0 varies by hundreds of MB with system state -> subtraction not robust.
2. Desktop/WDDM grows its own residency during the workload -> S2-S0 overestimates Marley.
3. Marley induces driver allocations not in torch.reserved -> NVML-reserved may be real Marley memory.
4. Device-wide increment grows while reserved stays flat -> investigate driver/WDDM/contexts/shared memory.
5. Full 30-step run increases the delta vs the short probe -> 3-step probe insufficient.

## 4. Governance

Observational metrology probe in an isolated runner (VAE skipped, reduced steps). Does NOT reclassify F7-D5 and authorizes NO runtime modification and NO gate change. Two-layer metric under study (A: device-wide raw, always reported; B: incremental workload residency, label pending validation). Any redefinition of the ground-truth metric of ROADMAP section 2 is a separate Director/Consejero decision.
