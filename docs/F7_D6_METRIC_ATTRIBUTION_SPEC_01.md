# F7-D6 — NVML Metric Attribution Probe: Separating Process-Resident vs Irreducible OS/WDDM Baseline

**Project:** Marley Runtime — Low-VRAM Diffusion Orchestration
**Phase:** F7 (1280×720) — F7-D6 measurement-attribution diagnostic
**Date:** 2026-09-09
**Status:** SPEC — awaiting Director execution authorization
**Trigger:** F7-D5 returned NEGATIVE (peak NVML device `used` 5,096.1 MB > 4,800 MB gate) despite `torch.reserved` stabilizing at ~3,766 MB across all 30 steps with no allocator divergence.
**References:** F7-D1 forensic profiling; F7-D3/D4/D5 seam-release series; ROADMAP §2 three-layer telemetry model; ROADMAP F7-D5 kill-gate.

**Evidence:** `MEASURED` / `OBSERVED` / `DERIVED` / `HYPOTHESIS` throughout.

---

## 1. Purpose (single question)

F7-D5 established (measured): over 30 FULL 720p steps with seam + step-start release,
- `torch.reserved` (attributable to the PyTorch caching allocator) is **stable** at ~3,766 MB, accumulation +246 MB over 30 steps (no divergence);
- Peak NVML **device-wide `used`** = 5,096.1 MB → classified NEGATIVE vs the 4,800 MB gate.

A preliminary read-only measurement (this session) shows the GPU **idle with no Marley process** reports **NVML device `used` ≈ 1,071.8 MB**, with **0 MB attributable** to any enumerated process (all PIDs return `usedGpuMemory=None` under WDDM).

F7-D6 tests **one measurement hypothesis**:

> Is a material fraction of the F7-D5 "peak" an **irreducible OS/WDDM/desktop baseline** that is counted by `nvmlDeviceGetMemoryInfo().used` but is **NOT attributable to the Marley process**, such that the *process-attributable* peak is actually below the 4,800 MB gate?

It does **not** optimize memory and does **not** re-run the full 30-step schedule. It is an isolated, low-step attribution probe whose output informs whether the ground-truth metric of §2 should be redefined.

## 2. Why per-process attribution is non-trivial under WDDM

Under the Windows Display Driver Model, `nvmlDeviceGetComputeRunningProcesses()`/`GraphicsRunningProcesses()` report `usedGpuMemory = None` for processes whose memory cannot be cleanly attributed at the driver level. Therefore **PID-level byte attribution is unavailable**. The runner instead reconstructs attribution from **in-session NVML milestones**, which are robust:

| Milestone | NVML device `used` meaning |
| :--- | :--- |
| `S0 = so_baseline` | Pre-context (before pipeline init). = OS + desktop + other processes + driver. |
| `S1 = ctx_baseline` | Post-pipeline-init, post-`empty_cache`. = S0 + CUDA context + resident non-block modules. |
| `S2 = denoise_peak` | Peak during DiT denoise. = S1 + live denoise residency. |

**Attributions derived:**
- `SO_share = S0` → memory that **no** process change can ever release (irreducible baseline).
- `ctx_overhead = S1 − S0` → CUDA-context + resident model overhead (process-attributable, fixed once loaded).
- `denoise_resident = S2 − S1` → the incremental residency actually driven by the denoise loop (the only part the seam release controls).
- Cross-check: `torch.reserved` peak (already measured in D5 = 3,766 MB) is a lower-bound process-attributable and should be consistent with `S2 − S0`.

## 3. Variable and workload (strict isolation)

- **Single independent variable:** NONE changed in the runtime. The probe only *measures* the same seam-release configuration already validated in F7-D5 (`seam_and_step_start`), run for a **reduced 3-step** window to bound runtime (~8–10 min incl. model load).
- Everything identical to F7-D5: 1280×720, 33 frames, `adaptive`, seed 42, same prompt/negative/scheduler, same subclass seam, same release discipline. VAE decode is **skipped** to focus the probe strictly on DiT residency attribution.
- No runtime file is modified. Only `f7_d6_attribution_probe.py` and this spec are added.

## 4. Method

1. `pynvml` init; record `S0` (device `used`) and enumerate compute/graphics processes + summed `usedGpuMemory` (expected ~0 attributable).
2. Instantiate the subclassed pipeline (DiT fp16, blocks on CPU, adaptive engine) — the same class pattern as F7-D5 but without VAE/export.
3. `gc.collect()` + `synchronize` + `empty_cache`; record `S1` (context baseline).
4. Run **3** DiT steps with the seam + step-start release discipline, sampling at ~20 ms: NVML device `used`, `torch.allocated`, `torch.reserved`, temp/clock/util.
5. Record `S2` (denoise peak device `used`); compute derived attributions (SO_share, ctx_overhead, denoise_resident) and cross-check `denoise_resident` vs `torch.reserved` peak.
6. Emit telemetry JSON + Markdown report.

## 5. Decision criteria (pre-defined)

| Result | Observed condition | Interpretation / action |
| :---: | :--- | :--- |
| 🟢 Attribution confirmed | `S0` (irreducible) ≈ 1.0–1.2 GB AND `S2 − S0` (process-attributable peak) ≤ ~4,000 MB (< 4,800) AND consistent with `torch.reserved` | F7-D5's NEGATIVE was a **measurement artifact of the device-wide metric**; recommend redefining the gate to process-attributable residency → F7-D5 likely FAVORABLE |
| 🟡 Partial | `S0` significant but `S2 − S0` still > 4,800 MB, or inconsistency between delta and `torch.reserved` | metric is part contributor; additional driver-level residency exists; Director decides |
| 🔴 Negative | `S0` small AND process-attributable peak genuinely > 4,800 MB | F7-D5 NEGATIVE stands; 720p confined to experimental under any metric |

## 6. Safety & governance

- Safe-abort NVML guard ~5,050 MB (flag only).
- 3 steps, no VAE → bounded runtime. This probe is **observational and reversible**; it authorizes **no** runtime modification and does **not** by itself reclassify F7-D5. A metric redefinition, if warranted, is a separate Director/Consejero governance decision. F7-0/D1/1/D2/D2b/D2c/D3/D4/D5 remain frozen references.
