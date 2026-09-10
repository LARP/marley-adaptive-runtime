# F7-D6 — Implementation Changes & Methodological Justification

**Project:** Marley Runtime — Low-VRAM Diffusion Orchestration
**Phase:** F7 (1280×720) — F7-D6 metrology probe
**Date:** 2026-09-09
**Artifact:** [`f7_d6_attribution_probe.py`](../f7_d6_attribution_probe.py)
**Authorization:** External Consultant authorizations no.1–4 (2026-09-09), recorded in
[`docs/private/F7_D6_CONSULTANT_AUTHORIZATION_01.md`](private/F7_D6_CONSULTANT_AUTHORIZATION_01.md) …
[`docs/private/F7_D6_CONSULTANT_AUTHORIZATION_04.md`](private/F7_D6_CONSULTANT_AUTHORIZATION_04.md).
**Status:** IMPLEMENTED — `py_compile` OK, `--dry-run` OK. **NOT executed.**

**Evidence:** `MEASURED` / `OBSERVED` / `DERIVED` / `HYPOTHESIS` throughout.

---

## 1. Purpose of this document

Record, before execution, **every change applied to the F7-D6 runner** relative to the
initial proposal, mapped to the Consultant's directive, and **justify why this information
is the appropriate basis for running the F7-D6 tests**. No benchmark has been executed.

---

## 2. Context and trigger (why F7-D6 exists)

- F7-D5 (`f371ebf`) ran the full 720p/33f/30-step schedule and was classified **NEGATIVE**
  because the **device-wide NVML `used`** peaked at **5,096.1 MB > 4,800 MB**.
- However, `torch.reserved` (allocator, process-attributable) was **stable at ~3,766 MB**
  across all 30 steps with no divergence, and `torch.allocated` peaked at 2,180.2 MB.
- A passive read with no Marley process showed the device already reports **~1,066–1,072 MB**
  used in idle, with `usedGpuMemory = None` for every enumerated PID (WDDM).

The asymmetry motivates a **measurement-metrology** question (not an optimization):
*does the device-wide NVML metric conflate an OS/WDDM baseline with workload-induced
residency?* F7-D6 answers only that question.

---

## 3. Change log (directive → change)

| # | Consultant directive | Change applied in `f7_d6_attribution_probe.py` |
| :-- | :--- | :--- |
| C1 | Control workload **before and after** the workload (`C0 → … → C1`) | Added `Control A` (pre-workload, before CUDA/pipeline init) and `Control B` (post-workload, after S3). Function `run_control_workload()` reused for both. |
| C2 | `S0 = 60 s` | `--s0-seconds` default changed **30.0 → 60.0**. |
| C3 | `S3 = 30 s` observation with full statistics | Added `--s3-seconds` (default 30) and `sample_stability()` at S3 (min/max/mean/median/std/spread). |
| C4 | `S4 = 30 s` final observation (confirmatory) | Added `--s4-seconds` (default 30) and a final `sample_stability()` after Control B. |
| C5 | `S1` = **operational baseline**, not "idle" | S1 relabeled `operational_baseline` and measured **after** Control A and after pipeline init. |
| C6 | Control A/B tolerance **±10% relative** | `classify_authorized()` computes `control_rel_diff_pct = |ΔB−ΔA|/ΔA`; `control_consistent = rel_diff ≤ 10%`. |
| C7 | Guard when `ΔNVML_A` below floor → INCONCLUSIVE (no tolerance) | `signal_floor = max(3*S0_std, 150 MB)`; if `ΔA < signal_floor` → `control_guard_tripped=True` and the 10% tolerance is **not** evaluated → verdict INCONCLUSIVE. |
| C8 | Workload measurable threshold `max(3*S0_std, 150 MB)` | `workload_increment (S2−S1)` compared against `signal_floor`; below it → INCONCLUSIVE. |
| C9 | `cudaMemGetInfo` coherence **qualitative/temporal** | `run_probe()` now captures `cudaMemGetInfo` at the workload peak; `classify_authorized()` checks CUDA-used rises S1→S2 and falls S2→S3, matching NVML direction. |
| C10 | 3-level classification | `classify_authorized()` returns **INSTRUMENT VALIDATED / INCONCLUSIVE / INSTRUMENT NOT VALIDATED** with per-condition flags. |
| C11 | `S2−S0` naming | Kept as **`workload_induced_device_residency_delta`** everywhere (telemetry, report, console). Never labeled ownership. |
| C12 | `usedGpuMemory = None` semantics | Console + telemetry note: "counter UNAVAILABLE under WDDM, NOT 0 MB". |
| C13 | Record NVML + torch + cudaMemGetInfo + timestamps/phases | Telemetry now records S1/S2/S3/S4 `cudaMemGetInfo`, torch allocated/reserved, per-step data, and both controls. |
| C14 | No gate change / no reclassification / no pipeline change | Governance note embedded in telemetry and report; only the runner changed; `pipeline.py` untouched. |

**Final authorized protocol implemented:**
`S0 (60 s) → Control A → init CUDA/PyTorch/pipeline → S1 → Wan 3 steps → S3 (30 s) → Control B → S4 (30 s)`.

---

## 4. Consolidated operational criteria (as implemented)

| Element | Criterion |
| :--- | :--- |
| S0 | 60 s; stable if spread ≤150 MB |
| Control A | ~500 MB; observable if ΔA ≥ max(3×S0_std, 150 MB) |
| Control B | ~500 MB; consistent if relative diff vs A ≤10% |
| S1 | operational baseline (system + CUDA/PyTorch + loaded pipeline) |
| Wan | 3 DiT steps, VAE skipped |
| S3 | 30 s; stable if spread ≤150 MB; returns if \|S3_mean − S0_mean\| ≤200 MB |
| S4 | 30 s; confirmatory/informative (not an independent gate) |
| cudaMemGetInfo | qualitative/temporal coherence with NVML |
| Verdict | INSTRUMENT VALIDATED / INCONCLUSIVE / INSTRUMENT NOT VALIDATED |

These are **operational criteria for F7-D6**, explicitly **not** a redefinition of the
4,800 MB production gate.

---

## 5. Justification — why this information is the right basis for the tests

1. **The gate metric itself is in question.** F7-D5's only formal FAIL was a device-wide
   NVML reading that does not start at zero (idle ≈1.07 GB). Before spending more GPU on
   720p, it is rational to establish **whether the instrument can distinguish a fixed
   device baseline from workload-induced residency**. This is exactly what F7-D6 measures.

2. **Process ownership is not directly observable under WDDM.** `usedGpuMemory = None`
   per PID means NVML does not expose a reliable per-process counter. Therefore the only
   defensible approach is **in-session milestones** (`S0/S1/S2/S3/S4`) plus a **known
   control workload** — which is what was implemented.

3. **A known control load validates the instrument before trusting it on Wan.** The
   ~500 MB CUDA control (alloc/compute/free) checks that the incremental metric responds
   reproducibly to a controlled perturbation. Running it **before and after** (Control A/B)
   guards against WDDM residency/budget drift within the session (Microsoft documents that
   residency budgets change dynamically).

4. **Statistical baselines prevent noise-driven conclusions.** S0/S3/S4 are sampled
   (min/max/mean/median/std/spread) so that a single instantaneous reading cannot be
   mistaken for a stable baseline, and the "workload measurable" test uses
   `max(3×S0_std, 150 MB)` to require a signal clearly above noise.

5. **Separation of instrument validation from workload validation.** F7-D6 validates the
   *measurement*; only if it is 🟢 INSTRUMENT VALIDATED would a full 30-step re-run be
   justified. This keeps a clean boundary and avoids re-litigating F7-D5 prematurely.

6. **Governance and traceability are preserved.** The runner, telemetry and report embed
   the prohibition on declaring PASS, discounting 1.07 GB, changing the gate, or modifying
   `pipeline.py`. F7-D5 remains frozen as NEGATIVE; any reclassification would require a
   separate formal decision.

7. **Bounded cost and reversibility.** The probe is 3 DiT steps + ~2 min of stability
   windows (~10–12 min GPU), observational and reversible, with a safe-abort guard at
   5,050 MB. It is the minimal experiment that can answer the metrology question.

---

## 6. What F7-D6 will and will NOT do

**Will:** measure and classify the *state of the instrument* under a controlled protocol;
produce raw values (NVML, torch, cudaMemGetInfo, controls) plus derived deltas.

**Will NOT:** reclassify F7-D5; change the 4,800 MB gate; discount 1.07 GB automatically;
integrate 720p; modify `pipeline.py`; present `S2−S0` as process ownership.

---

## 7. Execution checklist (pending Director go-ahead)

1. ✅ Runner implements the authorized protocol (this document).
2. ✅ `py_compile` OK.
3. ✅ `--dry-run` OK.
4. ⏳ Director go-ahead to execute.
5. ⏳ Execute `f7_d6_attribution_probe.py` (~10–12 min).
6. ⏳ Emit telemetry + report; submit to Consultant review.

---

*Marley Runtime is dedicated in loving memory to Marley 🐾.*
