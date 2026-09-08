# Technical Benchmark Report — Phase F3: Budgeted Asynchronous Scheduler

**Date:** 2026-09-08  
**Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,144 MB total)  
**Host:** Windows 11 (WDDM concurrency active)  
**Model:** Wan2.1-T2V-1.3B-Diffusers (30 DiT blocks)  
**Execution Mode:** Double-buffered Async Scheduler (`marley.ops.async_stream.BudgetedAsyncStreamer`)  
**Telemetry Artifact:** [`logs/f3_async_scheduler_benchmark.json`](../logs/f3_async_scheduler_benchmark.json)  
**Status:** 🟡 **SUPERSEDED — First-run reference (see certified [`TEST_F3_verification.md`](../results/TEST_F3_verification.md))**

> **Superseded 2026-09-08:** this first-run report is retained for traceability. Its +10.4% was an
> optimistic single sample and its "100% overlap" was an instrumentation artifact. The
> measurement-corrected, certified F3 result is **mean +7.2%** external wall-clock (stable
> over 5 A/B reps), overlap 100% measured, 2,584 MB VRAM → **F3 PASS, advance to F3+INT8**. See
> [`results/TEST_F3_verification.md`](../results/TEST_F3_verification.md).

---

## 1. Executive Summary

Phase F3 replaces synchronous per-block weight swapping with an **asynchronous double-buffered scheduler** executing on dual CUDA streams (`copy_stream` and `compute_stream`).

By enforcing a **bidirectional event ownership protocol** (`compute_done` $\leftrightarrow$ `copy_done`), the scheduler completely eliminates the race conditions present in single-event architectures, hides 100% of the PCIe H2D block transfer time behind preceding block computation, and achieves a **+10.4% wall-clock speedup** with **zero numerical deviation (0 NaNs / 0 Infs)**.

> **Measurement-integrity note:** the "100% overlap / full hiding" phrasing reflects the *first-run
> instrumentation*, later found to be an artifact (see §5). It is reported here for traceability but
> must NOT be cited as empirical evidence until the corrected measurement confirms it.

Peak physical memory allocation was maintained at **1,694 MB NVML**, comfortably below the 4,000 MB design budget and well within the 4,800 MB hard kill gate limit.

---

## 2. Benchmark Scorecard & Kill Gates Evaluation

All kill gates established in the approved implementation plan were evaluated and passed:

| Evaluation Criterion | Threshold / Kill Gate | Measured Result | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak VRAM (NVML)** | $\le 4,800\text{ MB}$ (Hard limit) | **1,694 MB** (Async) / 1,706 MB (Sync) | 🟢 **PASS** (+3,106 MB headroom) |
| **Effective Overlap** | $\ge 5.0\%$ (F3-A minimum) | **100.0%** ⚠️ *artifact — re-measuring* | 🟡 **RE-MEASURE** (real stall instrumentation pending) |
| **Wall-Clock Speedup** | $> 0.0\%$ (Async vs. Sync) | **+10.4%** ($17,186.9\text{ ms} \rightarrow 15,407.2\text{ ms}$) | 🟢 **PASS** (Definitive speedup) |
| **Numerical Integrity** | Zero NaN / Inf in outputs | **0 NaNs / 0 Infs** | 🟢 **PASS** (Zero corruption) |
| **Consistency / Race Check** | Low std dev across reps | **$\pm 165.5\text{ ms}$ std dev** | 🟢 **PASS** (Stable ownership) |
| **Stream Sync Overhead** | Minimize pipeline stalls | **Reduced from 150 to 5 forced syncs** | 🟢 **PASS** (-96.7% sync stalls) |

---

## 3. Detailed A/B Comparison (Sync vs. Async)

Measurements conducted on 5 simulated denoising steps across all 30 DiT transformer blocks (150 block executions per condition) using exact Wan2.1 activation shapes:

```
  hidden_states         : (1, 7800, 1536) torch.float16
  encoder_hidden_states : (1, 226, 1536)  torch.float16
  temb                  : (1, 6, 1536)    torch.float16
  rotary_emb            : 2 × (1, 7800, 1, 128) torch.float16
```

| Metric | Condition A (Marley Sync) | Condition B (Marley Async) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Total Wall-Clock Time** | $17,186.9\text{ ms}$ | **$15,407.2\text{ ms}$** | **-1,779.7 ms (+10.4% faster)** |
| **H2D Transfer Total** | $1,512.7\text{ ms}$ | $1,512.7\text{ ms}$ | Concurrent with compute |
| **Effective Overlap** | 0.0% | **100.0%** | Complete PCIe latency hiding |
| **Peak VRAM (NVML)** | 1,706 MB | **1,694 MB** | -12 MB lower jitter |
| **Prefetch Stall Latency** | 0.0 ms (N/A) | **0.00 ms** | Block $i+1$ ready before block $i$ ends |
| **Forced Stream Syncs** | 150 syncs | **5 syncs** | 1 sync per denoising step |
| **Time per Output Frame** | $1,011.0\text{ ms}$ | **$906.3\text{ ms}$** | **-104.7 ms/frame** |
| **Numerical Integrity** | Clean | Clean | 0 NaNs / 0 Infs |

---

## 4. Key Architectural Insights & Implementation Fixes

1. **Isolation of Autograd via `torch.no_grad()`**:
   - In preliminary runs, omitting `torch.no_grad()` resulted in PyTorch building autograd graphs for each block, causing virtual memory allocations to inflate. Wrapping both loops in `torch.no_grad()` contained the active VRAM footprint to just **~235 MB**, completely eliminating UMA paging.
2. **Direct Block Streaming vs. Accelerate Offload**:
   - By not invoking Hugging Face Accelerate's `enable_sequential_cpu_offload()`, Marley avoided `AlignDevicesHook` hijacking transfers to PyTorch's default stream. Marley's `BudgetedAsyncStreamer` is the sole authority over H2D copies, achieving true overlap.
3. **Double-Buffer VRAM Footprint**:
   - Pre-allocating two slots (`slot_0`, `slot_1`) on GPU requires only $2 \times 88.6\text{ MB} \approx 177.2\text{ MB}$ of VRAM, leaving over 4 GB of headroom for activations and VAE processing.

---

## 5. Phase Verdict and Next Steps

> ⚠️ **PROVISIONAL STATUS (2026-09-08):** A post-certification audit found that the reported
> **Effective Overlap = 100.0%** was an **instrumentation artifact** — `total_prefetch_latency`
> was never accumulated in `marley/ops/async_stream.py`, so the overlap formula returned 100% by
> construction. The **+10.4% wall-clock speedup** (external timing) remains valid as a preliminary
> signal but is **not yet certified**. Measurement-integrity corrections (Enmiendas 1–3: real
> overlap/stall instrumentation, F3-B overlap kill gate, external wall-clock as official metric,
> repeated A/B with alternating order + warmup) are implemented — see
> [`docs/F3_MEASUREMENT_VERIFICATION_03.md`](../docs/F3_MEASUREMENT_VERIFICATION_03.md) and
> [`docs/F3_CALIBRATION_CHANGES_02.md`](../docs/F3_CALIBRATION_CHANGES_02.md).

- **F3-A (Scheduler Correctness Validation):** 🟡 **PASS (correctness) — overlap re-measurement pending**
- **F3-B (Real Performance Certification):** 🟡 **PROVISIONAL** (+10.4% preliminary; pending repeated verification)
- **Phase F3 Overall Verdict:** 🟡 **IN VERIFICATION — not advanced to F3+INT8 until reproducible PASS**
