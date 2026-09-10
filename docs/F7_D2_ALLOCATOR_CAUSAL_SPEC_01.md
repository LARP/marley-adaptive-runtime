# F7-D2 — Allocator Causal Release Probe: Technical Specification

**Project:** Marley Runtime — Low-VRAM Diffusion Orchestration
**Phase:** F7 (1280×720) — Diagnostic sub-phase F7-D2
**Date:** 2026-09-09
**Status:** SPEC — awaiting Director execution authorization
**References:** F7-D1 Forensic Report (`docs/F7_D1_FORENSIC_PROFILING_REPORT_01.md`); Consultant Letter 2026-09-09 (§7 causal experiment); Director Response (`docs/private/F7_DIRECTOR_RESPONSE_TO_CONSULTANT_01.md` §4 Paso 3); F7-1 Pilot (Result C).

**Evidence classification:** `MEASURED` = instrumented in-run; `OBSERVED` = sampled telemetry; `DERIVED` = computed; `HYPOTHESIS` = expectation, never presented as result.

---

## 1. Purpose

F7-D1 established a strong correlational finding (H1): a large fraction of physical residency is **retained inside the PyTorch caching allocator** pool rather than being actively used by live tensors:

| Quantity | Observed (F7-D1 / F7-1) |
| :--- | :---: |
| PyTorch Allocated (live) | ~2,155–2,188 MB |
| PyTorch Reserved (pool) | ~5,742–5,894 MB |
| Allocator free pool delta | ~3,600–3,700 MB |
| Peak physical NVML | ~5,968–6,088 MB |

The consultant correctly cautions that this is **correlation, not demonstrated causality**: we have not yet shown that returning that reserved pool to CUDA actually lowers physical NVML.

F7-D2 is a **minimal, monovariable, reversible** causal probe that answers one question:

> **Does a controlled release of the caching allocator pool (empty_cache after releasing dead references + gc) produce a significant, reproducible reduction of PyTorch Reserved and physical NVML?**

This maps directly to the consultant's Case A / Case B discrimination at the boundary.

## 2. Governance and scope (single variable)

- **Only files modified:** the standalone runner `f7_d2_allocator_causal_probe.py` and this spec. New artifact files under `logs/`, `docs/`.
- **Runtime untouched:** `MarleyEndToEndPipeline`, VAE, streamers, `AdaptiveEngine`, diffusers, `end_to_end.py` are NOT modified.
- **Seam technique (reused from F7-D1):** a subclass of `MarleyEndToEndPipeline` replicates the stage logic in its own method so a controlled release can be inserted at the DiT→VAE boundary without editing the base class.
- **Independent variable (only this changes):** the execution of `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` (+ `synchronize`) at the post-denoise boundary.
  - Scope deliberately chosen: **no weight residency changes** (DiT non-block submodules remain on GPU), **no VAE decode**, **no full 5-step denoise**. This isolates the pure cached-pool-return effect and minimizes WDDM thrash/system exposure.
  - Dropping the single-step temporaries is a *precondition* for `empty_cache()` (it can only free unused blocks), not an additional optimization lever.
- **Evidence honesty:** every printed/reported number is tagged MEASURED / OBSERVED / DERIVED / HYPOTHESIS. No "causa raíz / demostrado / garantizado" claims without supporting data.

## 3. Why "one DiT step" reproduces the relevant state

F7-0, F7-D1 and F7-1 all show the physical/reserved ceiling is reached **during step 1** (activation sizes set the ceiling; F7-1 reached Reserved 5,742 MB / NVML ~5,968 MB inside the first partial step). Therefore executing a **single full DiT denoising step (cond + uncond, 30 block-streaming passes, adaptive mode)** at 1280×720/33f builds a realistic post-denoise retained-pool state, without the cumulative host-RAM pressure and the ~22-minute VAE thrash of a full run.

## 4. Method

1. Initialize NVML high-frequency sampler + reset peak stats.
2. Instantiate the subclassed pipeline (DiT fp16 on CPU blocks / GPU non-block, VAE bf16 tiled, same as F7-D1).
3. Encode prompt, prepare latents (identical to F7-D1).
4. Build `AdaptiveEngine` streamer.
5. Execute **exactly one** denoising iteration in an inner function scope so all step temporaries are dropped on return; then restore blocks / release streamer (mirrors the runtime `finally` path) and `gc.collect()`.
6. **Measurement event M1 — RETAINED (OBSERVED/MEASURED):** snapshot {timestamp, NVML used, torch allocated, torch reserved, allocator delta, # segments, RAM, temp, clock}.
7. **Intervention (single variable):** `torch.cuda.synchronize()` → `torch.cuda.empty_cache()` → `torch.cuda.synchronize()`.
8. **Measurement event M2 — RELEASED:** snapshot the same metrics.
9. Repeat the release cycle `--release-reps` times to test reproducibility (same operation, not a new variable).
10. Emit telemetry JSON and a Markdown report with labels.

## 5. Success / interpretation criteria (DERIVED decision table)

Let `ΔReserved = Reserved_RETAINED − Reserved_RELEASED` and `ΔNVML = NVML_RETAINED − NVML_RELEASED` (both ≥ 0):

| Case | Condition (measured) | Interpretation | Priority signal |
| :---: | :--- | :--- | :--- |
| **A** | `ΔNVML` large & reproducible | Returning the allocator pool lowers physical residency → H1 gains strong causal support | allocator retention is a primary physical driver |
| **B** | `ΔReserved` large but `ΔNVML ≈ 0` | Reserved returns to CUDA but physical stays high → physical not explained by the PyTorch cache alone (WDDM/driver residency) | allocator is NOT the sole physical driver |
| **C** | both `ΔReserved` and `ΔNVML` only partial | allocator is one of several contributors | investigate second component |
| **D** | `ΔReserved ≈ 0` | cached pool not releasable at this boundary (live references/segmentation) | revisit release scope |

A quantitative NVML drop threshold is reported numerically (no arbitrary pass/fail manufactured pre-run); reproducibility is assessed across the repeated release cycles.

## 6. Safety

- **Safe abort** guard near the demonstrated physical tolerance (~6.0–6.1 GB) to protect the system from OOM instability, via the high-frequency NVML sampler.
- Single step, no VAE decode, no full denoise ⇒ bounded runtime (model load ~65 s + one step ~60 s + probing ≈ 3–4 min).
- Reversible: no code, tile, quantization, or residency change persists.

## 7. Artifacts

- Telemetry: `logs/f7_d2_allocator_causal_telemetry.json`
- Report: `docs/F7_D2_ALLOCATOR_CAUSAL_PROBE_REPORT_01.md`

## 8. Awaiting authorization

Per project rule `execution_authorization.md`, this probe will NOT be executed until the Director explicitly authorizes the exact runner command.
