# F7-D6 — NVML Metric Attribution Probe (Measurement Validation)

**Project:** Marley Runtime — Low-VRAM Diffusion Orchestration
**Phase:** F7 (1280×720) — F7-D6 measurement-metrology diagnostic
**Date:** 2026-09-09
**Status:** ⏸ **PROPOSAL — REVISED per External Consultant (analysis, NOT approval). Awaiting Director authorization to execute.**
**Trigger:** F7-D5 returned NEGATIVE (peak NVML device `used` 5,096.1 MB > 4,800 MB gate) despite `torch.reserved` stabilizing at ~3,766 MB across all 30 steps with no allocator divergence.
**References:** F7-D1 forensic profiling; F7-D3/D4/D5 seam-release series; ROADMAP §2 three-layer telemetry model; ROADMAP F7-D5 kill-gate; **External Consultant response (2026-09-09)** — this revision incorporates its directives.

**Evidence:** `MEASURED` / `OBSERVED` / `DERIVED` / `HYPOTHESIS` throughout.

---

## 0. Consultant directives incorporated (2026-09-09)

The External Consultant reviewed the initial F7-D6 proposal and ruled:

- **Do NOT** reclassify F7-D5 yet; **do NOT** modify the 4,800 MB gate; **do NOT** integrate 720p; **do NOT** close the 720p line.
- **DO** treat F7-D6 strictly as a **metrology / measurement-validation** phase, and only if the Director authorizes it.
- `usedGpuMemory = None` under WDDM **does NOT mean the process uses 0 MB**; it means NVML does not expose the per-process counter in this environment. Therefore a "PID NVML ≤ gate" metric is **not** viable.
- Rename the derived quantity `S2 − S0` from "process attributable peak" to **"workload-induced device residency delta"** (it measures device-wide growth, not proven ownership).
- Validate **stability of the idle baseline** over 30–60 s (min/max/mean/median/std) before using it as a reference.
- Add a **known CUDA control workload** (~500 MB alloc/compute/free) to validate that the incremental instrument responds proportionally, before trusting it on Wan.
- Record `cudaMemGetInfo` alongside NVML and torch counters; verify `S3 ≈ S0`.
- Governance: investigate a **two-layer metric** (A: device-wide raw, always reported; B: incremental workload residency, label pending validation).

## 1. Purpose (single question)

F7-D5 established (measured): over 30 FULL 720p steps with seam + step-start release,
- `torch.reserved` is **stable** at ~3,766 MB, accumulation +246 MB over 30 steps (no divergence);
- Peak NVML **device-wide `used`** = 5,096.1 MB → classified NEGATIVE vs the 4,800 MB gate.

A preliminary read-only measurement (no Marley process) reports **NVML device `used` ≈ 1,065.8–1,071.8 MB**, with **0 MB attributable** to enumerated processes (`usedGpuMemory = None` under WDDM).

F7-D6 tests **one measurement hypothesis**:

> Is a material fraction of the F7-D5 "peak" a **device-wide baseline** (OS/WDDM/desktop/driver) counted by `nvmlDeviceGetMemoryInfo().used` but **not demonstrated to be attributable to the Marley process**, such that the *workload-induced device residency delta* is materially below the raw peak?

It does **not** optimize memory, does **not** re-run the full 30-step schedule, and does **not** by itself reclassify F7-D5. It validates whether the measurement instrument is fit to inform a later governance decision.

## 2. Why per-process attribution is unavailable under WDDM

`nvmlDeviceGetComputeRunningProcesses()`/`GraphicsRunningProcesses()` report `usedGpuMemory = None` under Windows/WDDM (VidMm-controlled residency). This is a **counter-availability limitation, not a 0-MB reading**. F7-D6 therefore reconstructs **device-wide milestones** and an incremental delta; it does **not** claim byte-level process ownership.

## 3. Milestones (revised protocol)

| Milestone | Meaning |
| :--- | :--- |
| `S0 = idle_baseline` | Device `used` sampled 30–60 s with NO Marley process. Report min/max/mean/median/std. |
| `S1 = ctx_baseline` | Device `used` after CUDA + PyTorch + pipeline init, then `synchronize` + `empty_cache`. |
| `S2 = workload_peak` | Peak device `used` during the DiT micro-workload (3–5 steps, same release discipline as F7-D5). |
| `S3 = post_run_floor` | Device `used` after `synchronize` + `empty_cache` + release; verify `S3 ≈ S0`. |

**Derived (revised naming):**
- `so_share = S0` → observed idle device occupancy (NOT asserted "irreducible" until stability is shown).
- `ctx_overhead = S1 − S0`.
- `workload_induced_device_residency_delta = S2 − S0` → **device-wide increment** (not proven ownership).
- Cross-check vs `torch.reserved` and `cudaMemGetInfo`.

## 4. Control workload (new, per Consultant)

Before the Wan workload, run a **known CUDA control**: allocate ~500 MB, perform a small compute, free it, `synchronize` + `empty_cache`, and repeat the `S0 → S1 → S2 → S3` measurement. This validates that the incremental metric (`ΔNVML = S2 − S0`) responds proportionally to a controlled load **before** applying the instrument to Wan.

## 5. Variable and workload (strict isolation)

- **Single independent variable:** NONE changed in the runtime. The probe only *measures* the same seam-release configuration validated in F7-D5 (`seam_and_step_start`), on a reduced **3–5 step** window.
- Identical to F7-D5: 1280×720, 33 frames, `adaptive`, seed 42, same prompt/negative/scheduler, same subclass seam. VAE decode **skipped**.
- No runtime file modified. Only `f7_d6_attribution_probe.py` and this spec are added.

## 6. Decision criteria (pre-defined; falsification-oriented)

| Result | Observed condition | Interpretation / action |
| :---: | :--- | :--- |
| 🟢 Instrument validated | `S0` stable (low spread) AND control workload `ΔNVML` proportional AND `S3 ≈ S0` AND Wan `S2 − S0` ≤ ~4,000 MB consistent with `torch.reserved` | Strong basis to propose a two-layer metric to the Director (raw + incremental); does NOT auto-reclassify F7-D5 |
| 🟡 Partial | Baseline significant but unstable, or `S2 − S0` > 4,800 MB, or inconsistent with `torch.reserved` | Metric is part contributor; additional driver residency exists; Director decides |
| 🔴 Negative | `S0` small/unstable AND incremental delta > 4,800 MB | F7-D5 NEGATIVE stands; pursue real memory optimization |

**Falsification scenarios (must remain open, per Consultant §13):**
1. `S0` varies by hundreds of MB with system state → subtracting it is not robust.
2. During the workload the desktop/WDDM increases its own residency → `S2 − S0` overestimates Marley's memory.
3. Marley induces driver allocations not visible in `torch.reserved` → `NVML − reserved` may represent real Marley-induced memory.
4. Device-wide increment grows with workload while `reserved` stays flat → investigate driver/WDDM/contexts/shared memory.
5. A full 30-step run increases the delta vs the short probe → a 3–5 step probe is insufficient to certify the metric.

## 7. Governance (revised, per Consultant §12/§16)

- **Two-layer metric under study (not adopted):**
  - **Metric A — device-wide raw** (`Peak NVML used`): always reported for transparency.
  - **Metric B — incremental workload residency** (`Peak NVML during workload − stable session baseline`): complementary indicator; **exact label pending F7-D6 validation**.
- Documented caveats: **raw device-wide occupancy ≠ process ownership**; **baseline-adjusted delta ≠ exact ownership unless independently demonstrated**.
- **F7-D5 verdict remains NEGATIVE** under the frozen rule until a methodological phase demonstrates the metric was not appropriate for attributing consumption to the workload. No retroactive rewrite.
- F7-D6 is **observational and reversible**; it authorizes **no** runtime modification and **no** gate change. Any reclassification / metric redefinition is a separate Director/Consejero decision.
- Safe-abort NVML guard ~5,050 MB (flag only). Frozen references unchanged: F7-0, F7-1, F7-D1, F7-D2/D2b/D2c, F7-D3, F7-D4, F7-D5.
