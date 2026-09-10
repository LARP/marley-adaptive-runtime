# F3 Script Calibration Document
## Technical Justification for Architecture and Implementation Changes

**Date:** September 8, 2026  
**Status:** 🟢 APPROVED — Code ready for writing. Execution requires explicit human authorization.  
**Origin:** Two technical advisory letters from External Consultant (Generative AI / SD / ComfyUI)  
**Applies to:** `marley/ops/__init__.py` · `marley/ops/async_stream.py` · `f3_async_scheduler_benchmark.py`

---

## 0. Advisory Letters and Final Approval

Two formal technical advisory letters were received and reviewed on 2026-09-08:

### Letter 1 — Pre-Implementation Technical Review
Identified two critical vulnerabilities in the original F3 proposal:
1. Race condition in buffer ownership (one-directional events only).
2. Incompatibility between `enable_sequential_cpu_offload()` and Marley's `copy_stream` (AlignDevicesHook bypasses our streams).

Also introduced methodological improvements: F3-A/F3-B structure, Marley Sync vs. Marley Async comparison, wall-clock as definitive metric, and the 4,800 MB hard limit / 4,000 MB design target distinction.

### Letter 2 — Final Approval Letter
After reviewing the revised plan incorporating Letter 1's corrections, the consultant issued:

> **"🟢 APROBADO PARA IMPLEMENTACIÓN. El plan revisado ha corregido los principales riesgos técnicos identificados anteriormente. Recomiendo proceder con F3."**

The Letter 2 ratified all changes without adding new ones. It added one methodological clarification:

> **"F3-A demuestra que el scheduler funciona; F3-B demuestra que el scheduler aporta valor real a Wan2.1."**

Both letters are stored in `docs/private/F3_F4_TECHNICAL_OPINION_01.md` for full traceability.

### Human Supervisor Approval
Explicit authorization received on 2026-09-08: code writing authorized, execution pending separate authorization.

---

## 1. Overview of Changes

After two rounds of technical advisory review (letters dated September 8, 2026), the Phase F3 implementation plan and associated scripts have undergone significant architectural calibration. This document records **every change made**, **why it was made**, and the empirical or methodological justification for each decision.

---

## 2. Change Log

### Change 1 — Buffer Ownership Race Condition Fix

**File:** `marley/ops/async_stream.py`  
**Classification:** Correctness fix (not optional)

**Problem (original code):**  
The original `execute_async_loop()` swapped the double-buffer slots immediately after `compute_stream.wait_event(copy_done[i+1])`, allowing `copy_stream` to start writing into the old compute slot **before `compute_stream` had finished reading from it**. This is a classic CUDA stream ownership violation that can produce:
- Silent tensor corruption (wrong weights loaded mid-computation).
- Non-deterministic CUDA errors that are difficult to reproduce.
- Occasional model output degradation without any error message.

**Corrected Protocol (bidirectional events):**
```
Iteration i — block[i] in slot_compute:

  1. Launch block[i].forward() on compute_stream
  2. Record compute_done_events[i] on compute_stream
  3. copy_stream.wait_event(compute_done_events[i])   ← NEW: ensures slot_compute is fully released
  4. If i+2 < N: prefetch block[i+2] → old slot_compute (now slot_prefetch after swap)
  5. compute_stream.wait_event(copy_done_events[i+1]) ← ensure block[i+1] is ready
  6. Swap slots
  7. Bind block[i+1] to new slot_compute
```

**Justification:**  
Step 3 is the critical addition. Without it, `copy_stream` could begin overwriting `slot_compute` for `block[i+2]` while `compute_stream` is still mid-forward-pass on `block[i]`. The CUDA API does not guarantee any protection across streams — only explicit events ensure correct ordering.

**Verification:** The benchmark will run 5+ repeated iterations per condition to detect non-deterministic corruption.

---

### Change 2 — Remove `enable_sequential_cpu_offload()` from F3 Execution

**File:** `f3_async_scheduler_benchmark.py`  
**Classification:** Architectural decision (required for measurement validity)

**Problem:**  
When `enable_sequential_cpu_offload()` is active, Hugging Face Accelerate installs an `AlignDevicesHook` on each module. This hook intercepts every `module.forward()` call and executes `module.to("cuda")` / `module.to("cpu")` **synchronously on PyTorch's default CUDA stream**, completely bypassing our `copy_stream`. The result is that our asynchronous scheduler has no actual effect on H2D transfers — Accelerate controls them.

**Solution:**  
F3 does **not** call `enable_sequential_cpu_offload()`. Instead, the Marley executor manages block residency directly:
- All 30 DiT blocks remain on CPU at pipeline load time.
- `BudgetedAsyncStreamer` explicitly transfers each block to the pre-allocated GPU double-buffer slots using `copy_stream`.
- The compute loop directly calls `block.forward()` after binding parameters to the active GPU slot.

**Consequence:** This makes Marley the **true owner of the DiT block lifecycle** — not Accelerate.

**What we do NOT re-implement:** The full Wan2.1 pipeline (text encoding, VAE decode, noise scheduler, conditioning). Only the DiT forward pass loop is replaced. Text encoder, VAE, and conditioning remain managed by diffusers as-is.

---

### Change 3 — Two-Level Validation Structure (F3-A and F3-B)

**File:** `f3_async_scheduler_benchmark.py` (new structure)  
**Classification:** Methodological improvement

**Problem:**  
Using only synthetic activations to declare F3 "PASS" would be a misleading claim. Synthetic forward passes prove the scheduler mechanism works; they do not prove that real Wan2.1 denoising is faster.

**Solution:**  
F3 is structured as two distinct validation levels:

| Level | Name | What It Proves |
|:---|:---|:---|
| **F3-A** | Scheduler Validation | Mechanism is correct: no race conditions, no CUDA errors, no NaN/Inf, events fire correctly |
| **F3-B** | Real Performance Certification | Real Wan2.1 denoising wall-clock is reduced by Marley Async vs. Marley Sync |

**F3-A uses** synthetic tensors with shapes captured from a real forward hook on Wan2.1 (exact 480p/17f dimensions).  
**F3-B uses** real DiT blocks with the full denoising simulation (noise scheduler, real latents, real steps).

**F3 is only declared PASS when F3-B confirms a wall-clock improvement.**

---

### Change 4 — Marley Sync vs. Marley Async (Not Accelerate vs. Marley)

**File:** `f3_async_scheduler_benchmark.py`  
**Classification:** Experimental design improvement

**Problem:**  
The original A/B comparison was:
- A = Accelerate `enable_sequential_cpu_offload()`
- B = Marley async

This comparison has a confounding variable: architecture difference. Any speedup or slowdown could be attributed to the Marley execution model itself, not to the async/sync distinction.

**Solution:**  
Both conditions use the same Marley execution architecture:
- **Condition A (Marley Sync):** Marley executor with synchronous block loading (`execute_sync_loop()`).
- **Condition B (Marley Async):** Marley executor with asynchronous double-buffered prefetch (`execute_async_loop()`).

This cleanly isolates the variable under test: **does async prefetching improve latency when everything else is identical?**

---

### Change 5 — Wall-Clock as Definitive Metric (CUDA Overlap as Instrumental)

**File:** `f3_async_scheduler_benchmark.py`  
**Classification:** Metric interpretation

**Clarification:**  
CUDA Event timing measures GPU-side overlap. It is possible to observe 90% overlap but 0% wall-clock speedup if the scheduler itself introduces CPU-side overhead, synchronization latency, or host-device stall.

**Decision:** The definitive metric for F3 is the end-to-end wall-clock delta between Condition A and Condition B. CUDA overlap percentage is an **explanatory** metric.

**Interpretation Table (from consultant):**

| Wall-Clock Delta | Interpretation |
|:---|:---|
| `< 0%` (slowdown) | ❌ FAIL — scheduler introduces overhead |
| `0% – 5%` | ⚠️ Marginal benefit |
| `5% – 15%` | 🟢 SUCCESS |
| `15% – 30%` | 🟢 Very good result |
| `> 30%` | 🟢 Excellent |

---

### Change 6 — Memory Gate Separation (Hard Limit vs. Design Target)

**Classification:** Kill Gate refinement

| Level | Value | Meaning |
|:---|:---|:---|
| **Hard Limit** | 4,800 MB NVML | Exceeding this → immediate ABORT |
| **Design Target** | 4,000 MB NVML | Below this → excellent result |
| **Acceptable** | 4,000–4,800 MB | Pass, with headroom note |

This prevents rejecting a real performance improvement solely because VRAM landed at 4,200 MB instead of 4,000 MB.

---

### Change 7 — DiT FP16 Frozen for F3 Execution

**Classification:** Sequencing constraint

F3 executes strictly on DiT FP16. INT8 and NF4 are applied only in a subsequent benchmarking round after F3 is declared PASS. This ensures the wall-clock improvement attributable to async scheduling can be separated from the wall-clock improvement attributable to reduced payload (INT8 blocks = 45.9 MB vs. FP16 blocks = 88.6 MB).

The evolution sequence remains:
```
F3 (FP16)  →  F3 + INT8  →  F4 Adaptive Engine
```

---

### Change 8 — Real Overlap / Stall Instrumentation (Enmienda 1)

**File:** `marley/ops/async_stream.py`  
**Classification:** Measurement-integrity fix (required before overlap gate activation)

**Problem (found by post-certification audit, consultant letter 2):**  
`total_prefetch_latency` was initialized to `0.0` and never accumulated anywhere. The
formula `1.0 − (total_prefetch_latency / total_h2d_time)` therefore produced
**100% overlap by construction**, not by measurement. Compute events were also only read
after the full loop, reflecting the last step only.

**Corrected instrumentation:**
- Per-step, per-block CUDA events accumulate three empirical signals across ALL steps:
  - **Stall** — `stall_ev_start` recorded on `compute_stream` right before
    `wait_event(copy_done_events[i+1])`; elapsed vs. `copy_done_events[i+1]` gives the
    real GPU idle waiting on a late prefetch (negative → clamp 0 = no stall).
  - **Copy** — `ev_copy_start[j]` vs. `copy_done_events[j]`: GPU-measured H2D copy
    duration of each prefetched block (j ≥ 1).
  - **Compute** — `ev_compute_start[i]` vs. `ev_compute_end[i]`: GPU-measured block
    forward duration.
- Elapsed times are computed immediately after each step's `torch.cuda.synchronize()`,
  before the next step re-records the reused event arrays.
- New `StreamMetrics` fields: `per_block_copy_s`, `per_block_compute_s`,
  `per_block_stall_s` (auditable per-block timeline). `prefetch_latency_s` now holds the
  real accumulated stall (it was always `0` before).
- `effective_overlap_pct = (1 − real_stall / measured_async_copy) × 100`, clamped to
  [0, 100], where `measured_async_copy` is the GPU-measured prefetch work (blocks ≥ 1).
- `total_h2d_transfer_time_s` is now measured (GPU copies + synchronous block-0 loads);
  `sync_baseline_h2d_s` is retained only as a fallback.

Full methodology and protocol: [`docs/F3_MEASUREMENT_VERIFICATION_03.md`](F3_MEASUREMENT_VERIFICATION_03.md).

---

### Change 9 — F3-B Overlap Kill Gate + External Wall-Clock (Enmiendas 1 & 3)

**File:** `f3_async_scheduler_benchmark.py`  
**Classification:** Methodological correction

**Gate gap (Hallazgo Nº1, consultant letter 1):** F3-B called `check_kill_gates()` without
`overlap_pct`, so a hypothetical 0%-overlap run could still PASS. The F3-B gate now
receives the **minimum overlap over all repetitions** (`overlap_pct=min_overlap`,
threshold `< 5% → FAIL`) — active only after Change 8 made the metric empirical.

**Wall-clock gap (Hallazgo Nº2, consultant letter 1):** the speedup was computed from
`total_denoise_time_s`. Now the official metric is the **external wall-clock** captured by
`perf_counter()` bracketing each condition; the internal time is retained as a diagnostic
and reported alongside so internal/external consistency can be verified.

Independent copies of the initial hidden state are supplied per condition (`h_a`, `h_b`)
to eliminate any A/B state contamination.

---

### Change 10 — Repeated A/B Verification with Alternating Order (Enmienda 2)

**File:** `f3_async_scheduler_benchmark.py`  
**Classification:** Reproducibility protocol

F3-B now runs `--f3b-reps` repetitions (default 5) with **execution order alternated**
(`A/B`, `B/A`, …) and a **discarded warmup pass** (`--no-f3b-warmup` to disable) before the
measured repetitions. This decouples thermal / driver / caching drift from either
condition and answers the certification question "is the +10,4% stable and reproducible?"

Reports per condition and for the speedup: **mean / min / max / std dev**. Kill gates
evaluate the worst case per gate (min overlap, min speedup, max VRAM, any NaN). The JSON
output now contains an `aggregate` block plus a `per_rep` array with per-block GPU
timelines.

---

## 3. Summary of All Changes

| # | Change | Scope | Type |
|:---|:---|:---|:---|
| 1 | Bidirectional buffer ownership events | `async_stream.py` | Bug fix (mandatory) |
| 2 | Remove `enable_sequential_cpu_offload()` | `f3_benchmark.py` | Architecture (required) |
| 3 | Two-level F3-A / F3-B validation | `f3_benchmark.py` | Methodology |
| 4 | Marley Sync vs. Marley Async comparison | `f3_benchmark.py` | Experimental design |
| 5 | Wall-clock as definitive metric | `f3_benchmark.py` | Metric interpretation |
| 6 | 4,800 MB hard limit / 4,000 MB design target | `f3_benchmark.py` | Kill gate refinement |
| 7 | DiT FP16 frozen during F3 | All | Sequencing constraint |
| 8 | Real overlap/stall instrumentation (Enmienda 1) | `async_stream.py` | Measurement-integrity fix |
| 9 | F3-B overlap gate + external wall-clock (Enmiendas 1, 3) | `f3_benchmark.py` | Methodological correction |
| 10 | Repeated A/B alternated verification + stats (Enmienda 2) | `f3_benchmark.py` | Reproducibility protocol |

---

## 4. Code Architecture Summary

### `marley/ops/__init__.py`
Exports `BudgetedAsyncStreamer` and `StreamMetrics` from `async_stream.py`.

### `marley/ops/async_stream.py`

| Component | Purpose |
|:---|:---|
| `StreamMetrics` | Dataclass holding all 7 mandatory F3 metrics + per-block GPU timeline (`per_block_copy_s` / `per_block_compute_s` / `per_block_stall_s`) |
| `BudgetedAsyncStreamer.__init__()` | Loads block params into pinned CPU memory, creates 2 GPU slots |
| `_prepare_host_weights()` | Pins all 30×N params to CPU pinned memory for DMA-safe transfers |
| `_init_gpu_slots()` | Allocates 2 GPU buffer sets (slot A and slot B) matching block param shapes |
| `_load_block_sync()` | Synchronous H2D copy to a given slot (Condition A baseline) |
| `_prefetch_block_async()` | Non-blocking H2D copy on `copy_stream` + records `copy_done_events[i]` |
| `_bind_block_to_slot()` | Patches `param.data` → GPU slot tensor for all block parameters |
| `_restore_block_to_cpu()` | Restores `param.data` → original pinned CPU tensor |
| `probe_shapes()` | CPU-only shape probe via transformer.rope module (no GPU needed) |
| `execute_sync_loop()` | Condition A: sync transfer per block, wall-clock and transfer timing |
| `execute_async_loop()` | Condition B: bidirectional events, double-buffering, all 7 metrics |
| `release()` | Frees GPU slots and clears cached state |

### `f3_async_scheduler_benchmark.py`

| Stage | Description |
|:---|:---|
| **Startup** | Load transformer blocks (CPU). Probe activation shapes. Init NVML sampler. |
| **F3-A** | 5 repetitions of Sync+Async. NaN/Inf check per rep. Race condition detection. |
| **F3-B** | Real Wan2.1 DiT Execution Performance Certification: `--f3b-reps` repetitions (Sync + Async, order alternated, warmup discarded). Official speedup = external wall-clock. Reports mean/min/max/std + per-block timeline. |
| **Output** | Console scorecard + `logs/f3_async_scheduler_benchmark.json` |

---

## 5. Authorization Protocol

> [!IMPORTANT]
> **STRICT RULE: No scripts, commands, or benchmark executions will be launched without explicit written authorization from the human supervisor.** (Rule registered in `.agents/rules/execution_authorization.md`)

### Sequence Pending Authorization:

| Step | Action | Authorization / Status |
|:---|:---|:---:|
| 1 | Write `marley/ops/__init__.py` | ✅ Completed |
| 2 | Write `marley/ops/async_stream.py` | ✅ Completed (bidirectional events + no_grad) |
| 3 | Write `f3_async_scheduler_benchmark.py` | ✅ Completed |
| 4 | **Run F3-A validation** | ✅ Completed — 🟡 PROVISIONAL (overlap re-measured after Enmienda 1) |
| 5 | **Run F3-B benchmark** | ✅ Completed — 🟡 PROVISIONAL (+10.4% speedup preliminary; formal PASS pending repeated verification after Changes 8–10) |
| 6 | Write `results/TEST_F3_async_scheduler.md` | ✅ Completed (first-run reference, superseded) |
| 7 | **Run F3 verification battery** (Changes 8–10) | ✅ Completed — 🟢 F3 CERTIFIED PASS (mean +7.2%, 5 A/B reps) · [`results/TEST_F3_verification.md`](../results/TEST_F3_verification.md) |
| 8 | **`git commit` + `git push`** | ⏳ Requires explicit user command |
