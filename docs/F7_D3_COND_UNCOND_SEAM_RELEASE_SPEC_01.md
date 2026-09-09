# F7-D3 — Controlled cond→uncond Seam Release: Technical Specification

**Project:** Marley Runtime — Low-VRAM Diffusion Orchestration
**Phase:** F7 (1280×720) — F7-D3 causal experiment
**Date:** 2026-09-09
**Status:** SPEC — awaiting Director execution authorization
**References:** F7-D2c per-block diagnostic; Consultant Report 2026-09-09 (recommendation for F7-D3)

**Evidence:** `MEASURED` / `OBSERVED` / `DERIVED` / `HYPOTHESIS` throughout.

---

## 1. Purpose (single question)

F7-D2c established (observed):
- cond pass: Reserved 790 → 3,046 MB (first-block jump +2,256 MB), then a flat plateau;
- FreePool after cond ≈ 2,308 MB;
- uncond pass: Reserved 3,046 → 5,682 MB (another +2,256 MB first-block jump);
- 53 segments allocated, 0 freed (no `empty_cache()` was ever called);
- Peak NVML 6,006 MB (gate deficit ≈ 1,206 MB).

F7-D3 tests **one causal hypothesis**:

> Does a controlled release (`empty_cache` after dropping dead refs + sync) at the natural cond→uncond seam prevent or reduce the second Reserved/NVML growth of the uncond pass, and does the resulting peak stay below the 4,800 MB hard gate?

It does **not** optimize the runtime. It is an isolated, single-variable causal probe executed in a subclass seam (as F7-D2/D2b/D2c), with the release inserted in the **runner** between the two streamer calls.

## 2. Variable and workload (strict isolation)

- **Single independent variable:** a `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` (+ `synchronize`) executed exactly once at the cond→uncond seam.
- **Everything else identical to F7-D2c:** 1280×720, 33 frames, 1 DiT step, `adaptive`, seed 42, same prompt/negative/scheduler, same runtime, same subclass seam technique, same per-block forward hooks.
- **Not changed simultaneously:** INT8, AdaptiveEngine, attention, VAE, offload strategy, frames, resolution, scheduler, allocator config (`max_split_size_mb`), or any runtime file.
- Runtime files remain **unmodified**; only `f7_d3_cond_uncond_seam_release.py` and this spec are added.

## 3. Method

1. Instantiate the subclassed pipeline (DiT fp16, blocks on CPU, adaptive engine).
2. Encode prompt, prepare latents.
3. Register per-block forward hooks (reused from F7-D2c) to observe the uncond Reserved series block by block.
4. Run the **cond** 30-block pass.
5. **Seam intervention (single variable):**
   - record `pre_release` snapshot {NVML, allocated, reserved, free_pool, segments};
   - `gc.collect()`; `torch.cuda.synchronize()`; `torch.cuda.empty_cache()`; `torch.cuda.synchronize()`;
   - record `post_release` snapshot (same fields).
6. Run the **uncond** 30-block pass; observe whether `uncond block 0` still jumps Reserved to ~5,682 MB or stays near the post-release level.
7. Record peak NVML / allocated / reserved, per-block Reserved series, segments allocated/freed, timing, NaN/Inf, and telemetry (temp/clock/util/RSS).
8. Emit telemetry JSON + Markdown report.

## 4. Decision criteria (pre-defined; not manufactured post-hoc)

| Result | Observed condition | Interpretation / action |
| :---: | :--- | :--- |
| 🟢 Favorable | Peak NVML ≤ 4,800 MB, no catastrophic cadence regression, no NaN/Inf, and Reserved does **not** re-grow immediately to ~5.7 GB | strong evidence for the allocator-release path; candidate runtime integration |
| 🟡 Partial | Peak drops significantly but > 4,800 MB (e.g. ~6,006 → ~5,100) | allocator is a contributor but insufficient alone; Director decides whether to pursue further (no automatic chaining) |
| 🔴 Negative | NVML ≈ unchanged, or Reserved returns to ~5,682 MB immediately, or heavy reallocation penalty / CUDA error / NaN / instability | close the allocator seam-release line |

Primary quantitative guard: **peak NVML** (25 ms sampling). Secondary: Reserved immediately after `uncond block 0`.

## 5. Safety & governance

- Safe-abort NVML guard ~6,050 MB (OOM protection), non-intrusive (flag only).
- Single step, no VAE decode, no full denoise → bounded ~3–4 min.
- This experiment is **causal and reversible**; it does **not** authorize any runtime modification. If favorable, a subsequent isolated integration would require its own authorization. F7-0/D1/1/D2/D2b/D2c remain frozen references.
