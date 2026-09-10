# F7-D6 — NVML Metric Attribution Probe -- Report

**Date:** 2026-09-09T22:08:43.888200  
**Workload:** 1280x720 @ 33 frames, 3 DiT steps, mode `adaptive` (VAE skipped)  
**Status:** measurement-metrology probe (per External Consultant authorizations no.1-4). Does NOT reclassify F7-D5 and does NOT change the gate.  
---

## 1. Instrument Validation Result

- **VERDICT: INSTRUMENT NOT VALIDATED**
- Reason: Control A/B not reproducible (relative difference > 10%)

### 1.1 Milestones

- S0 idle_baseline (60 s): mean 983.0 MB (min 982.96 / max 983.21 / median 982.96 / std 0.05 / spread 0.25)
- S1 operational_baseline (post-init): 1095.42 MB
- S2 workload_peak (device used): 4883.26953125 MB (D5: 5096.1)
- S3 post_workload mean: 1887.22 MB (spread 349.75)
- S4 final mean: 1902.93 MB (spread 235.59)
- workload increment (S2-S1): 3787.8 MB
- **workload-induced device residency delta (S2-S0): 3900.3 MB** (NOT proven ownership)

### 1.2 Conditions

- s0_stable: True
- control_a_observable: True
- control_consistent: False
- workload_measurable: True
- s3_returns_to_baseline: False
- cudaMemGetInfo_coherent: True
- s4_stable: False
- control A delta: 592.53 MB | control B delta: 500.13 MB | rel diff: 15.59 %
- signal floor: 150.0 MB
- Peak Reserved (torch): 3776.0 MB (D5: 3776)

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
