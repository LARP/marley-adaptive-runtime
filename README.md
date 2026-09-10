# marley-runtime

<div align="center">

<img src="docs/logo.png" alt="Marley Runtime logo" width="160"/>

**Adaptive Memory Management Runtime for Video Diffusion Models on Ultra-Low VRAM GPUs**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.6 CUDA 12.4](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA%2012.4-ee4c2c.svg)](https://pytorch.org/)
[![Target: RTX 3050 Laptop](https://img.shields.io/badge/Hardware-RTX%203050%206GB%20Laptop-76b900.svg)](https://www.nvidia.com/)
[![Roadmap: v5 Approved](https://img.shields.io/badge/Roadmap-v5%20Approved-8b5cf6.svg)](ROADMAP.md)
[![Phase F0: PASS](https://img.shields.io/badge/Phase%20F0-PASS-22c55e.svg)](ROADMAP.md)
[![Phase F0.5: PASS](https://img.shields.io/badge/Phase%20F0.5-PASS%20(2.9GB)-22c55e.svg)](results/TEST_J_vae_tiling_bf16.md)
[![Phase F0.6: PASS](https://img.shields.io/badge/Phase%20F0.6-PASS%20(WDDM%20overlap)-22c55e.svg)](results/TEST_F0.6_wddm_overlap.md)
[![Phase F1: PASS](https://img.shields.io/badge/Phase%20F1-PASS%20(3.2GB)-22c55e.svg)](results/TEST_F1_lifetime_profiler.md)
[![Phase F1.7: PASS](https://img.shields.io/badge/Phase%20F1.7-PASS%20(-50%25%20T5)-22c55e.svg)](results/TEST_F1.7_selective_quantization.md)
[![Phase F3: CERTIFIED](https://img.shields.io/badge/Phase%20F3-CERTIFIED%20(+7.2%25)-22c55e.svg)](results/TEST_F3_verification.md)
[![Phase F3+INT8: PASS](https://img.shields.io/badge/Phase%20F3%2BINT8-PASS%20(+13.2%25)-22c55e.svg)](results/TEST_F3_INT8_benchmark.md)
[![Phase F4: CORE VALIDATED](https://img.shields.io/badge/Phase%20F4-CORE%20VALIDATED-8b5cf6.svg)](results/TEST_F4_adaptive_benchmark.md)
[![Phase F5: RETIRED](https://img.shields.io/badge/Phase%20F5-RETIRED%20(2.1GB%20%40%2033f)-22c55e.svg)](results/TEST_F5_vae_probe_33f.md)
[![Phase F6: CERTIFIED PASS & FROZEN](https://img.shields.io/badge/Phase%20F6-CERTIFIED%20%26%20FROZEN-22c55e.svg)](docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md)
[![Phase F7-0: GATE FAIL](https://img.shields.io/badge/Phase%20F7--0-GATE%20FAIL%20(6.05GB)-ef4444.svg)](docs/F7_0_FEASIBILITY_PROBE_REPORT_01.md)
[![Phase F7-D4: FAVORABLE](https://img.shields.io/badge/Phase%20F7--D4-FAVORABLE%20(4.71GB)-22c55e.svg)](docs/F7_D4_MULTISTEP_VALIDATION_REPORT_01.md)
[![Phase F7-D5: NEGATIVE](https://img.shields.io/badge/Phase%20F7--D5-NEGATIVE%20(5.10GB)-ef4444.svg)](docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md)
[![Phase F7-D6: VALIDATED](https://img.shields.io/badge/Phase%20F7--D6-INSTRUMENT%20VALIDATED-22c55e.svg)](docs/F7_D6_METRIC_ATTRIBUTION_REPORT_01.md)

*In loving memory of Marley 🐾*

</div>

---

## 📌 Executive Summary

**`marley-runtime`** is a specialized, experimental inference runtime designed to execute advanced video diffusion models (specifically **Wan2.1-T2V-1.3B**) under severe consumer hardware constraints: **~4.8 GB of effective physical VRAM** on an NVIDIA GeForce RTX 3050 6GB Laptop GPU running under Windows WDDM.

Rather than relying purely on blunt sequential CPU offloading, `marley-runtime` investigates and measures **adaptive, block-level memory management policies** that dynamically balance PCIe transfer latency, activation recomputation, tensor lifetime management, and selective quantization.

> [!NOTE]
> The current architectural design and phased milestone strategy is governed by **[Roadmap v5](ROADMAP.md)**, reviewed and approved through a multi-agent consensus using **free-tier models** (ChatGPT, DeepSeek Pro, and Gemini Pro). Active execution is performed by **Antigravity Pro** (Gemini 2.5 Flash and Claude Sonnet 4.6), with formal strategic adoption of technical guidance from an external **Generative AI & ComfyUI Runtime Expert** (knowledge acquired through conversation with an AI agent).

---

## 🎯 Target Hardware & Budget Specification

| Specification | Value |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, Compute Cap. 8.6) |
| **Total VRAM** | 6144 MB (6.0 GB) |
| **Effective Headroom** | **~4.8 GB** (accounting for WDDM OS residency) |
| **Host CPU & RAM** | Intel Core i7-13650HX (14c/20t) · 24 GB DDR5 |
| **OS & Driver** | Windows 11 · Driver 581.86 · CUDA 13.0 (runtime: 12.4) |
| **Python Stack** | Python 3.10.11 · PyTorch 2.6.0+cu124 · diffusers 0.40.0 |

### Ground-Truth Telemetry Model (ROADMAP §2)

Memory is tracked across **three distinct layers** to prevent WDDM virtualization artifacts:

1. `torch.cuda.memory_allocated()` — active PyTorch tensor footprint
2. `torch.cuda.memory_reserved()` — PyTorch caching allocator pool
3. **NVML physical residency** — hardware ground-truth (sampled continuously at 50ms) · **the strict 4.8 GB gate applies here**

---

## 📊 Current Phase Status

| Phase | Name | Status | Key Result |
| :---: | :--- | :---: | :--- |
| **F0.1** | Hardware Verification | 🟢 **PASS** | Driver 581.86 · CUDA 13.0 · RTX 3050 6GB confirmed |
| **F0** | Reproducible Baseline | 🟢 **PASS (Func.) / ❌ Gate** | Wan2.1 run at 480p/16f; peak 5,451 MB (+651 MB gate violation) · [`TEST_F0_baseline_ref.md`](results/TEST_F0_baseline_ref.md) |
| **F0.5** | Progressive Exploration | 🟢 **PASS** | **Peak NVML 2,902 MB (+1,898 MB headroom)** · VAE BF16 + Tiling (Test J) · [`TEST_J_vae_tiling_bf16.md`](results/TEST_J_vae_tiling_bf16.md) |
| **F0.6** | WDDM Concurrency | 🟢 **PASS** | **Overlap 79.9–96.3%** · PCIe async copy hides compute → F3 ACTIVE · [`TEST_F0.6_wddm_overlap.md`](results/TEST_F0.6_wddm_overlap.md) |
| **F1** | Lifetime Profiler | 🟢 **PASS** | **Peak NVML 3,223.7 MB** · 32 components traced live (text/DiT×30/VAE) · [`TEST_F1_lifetime_profiler.md`](results/TEST_F1_lifetime_profiler.md) |
| **F1.5** | Bottleneck Analysis | 🟢 **PASS** | **5 Categories Decomposed** · Allocator frag 44.9 MB (<1%) → **F2 Bypassed**; Prefetch safe (88.6 MB vs 1.57 GB) → **F3 Greenlit** · [`TEST_F1.5_bottleneck_classification.md`](results/TEST_F1.5_bottleneck_classification.md) |
| **F1.7** | Selective Quantization | 🟢 **PASS** | **T5 -50.0% (7.38 GB RAM saved)** · DiT Proj -48.2% (-76.9 GB PCIe payload) · Cosine 0.9996 · [`TEST_F1.7_selective_quantization.md`](results/TEST_F1.7_selective_quantization.md) |
| **F2** | Static Slab Allocator | ❌ **BYPASSED** | Allocator fragmentation is < 1.0% (44.9 MB vs 720 MB threshold); custom allocator discarded |
| **F3** | Async Stream Scheduler | 🟢 **PASS** | Real Wan2.1 DiT, measurement-corrected: **mean +7.2% wall-clock** (stable 6.0–8.4% over 5 A/B reps), overlap 100% measured, 2,584 MB VRAM. Certified → F3+INT8. · [`Verification`](results/TEST_F3_verification.md) · [`TEST_F3`](results/TEST_F3_async_scheduler.md) · [`Methodology`](docs/F3_MEASUREMENT_VERIFICATION_03.md) |
| **F3+INT8** | Transfer-Volume Isolation | 🟢 **PASS** | INT8 linear proj + FP16 compute, scheduler frozen: Async-vs-Sync **+13.2%**, payload −49.9%, overlap 98.1%, peak **2,076 MB**, cos ≥ 0.9999 · [`Report`](results/TEST_F3_INT8_benchmark.md) · F3 FP16 baseline frozen |
| **F4** | Adaptive Decision Engine | 🟢 **CORE VALIDATED** | Same-session A/B/C/D (30×3): Safety PASS (2,882 MB, 0 NaN), Adaptive Gate PASS (no oscillation), overhead 0.16 ms. D vs best static B **−3.40%** (no-pressure, perf cert pending) · [`Report`](results/TEST_F4_adaptive_benchmark.md) · [`Contract`](docs/F4_TEST_SPEC_01.md) |
| **F5** | Temporal VAE Stitcher | 🟢 **RETIRED** | Probe F5-A certified: 33f VAE decodes in **26.99s @ 2,109 MB** (Hard Gate ≤ 4,800 PASS); stitcher unnecessary, resolved by native tiling · [`TEST_F5`](results/TEST_F5_vae_probe_33f.md) |
| **F6** | Multidimensional Benchmarks | 🟢 **CERTIFIED & FROZEN** | 480p/33f/30s Multirun 4×3 certified ($n=3, df=2$): **12/12 runs PASS $\le 4,800$ MB**, cadencia ~14.25 s/p, 0 NaNs. CERRADA y CONGELADA (`c6e4bfb`) · [`Report`](docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md) |
| **F7-0** | 720p Feasibility Probe | ❌ **GATE FAIL / 🟢 PASS** | 1280×720 @ 33f (5 steps): Peak NVML **6,058.5 MB** (+1,258.5 MB over gate), 0 NaNs, MP4 OK. Frozen (`83f138a`) · [`Report`](docs/F7_0_FEASIBILITY_PROBE_REPORT_01.md) |
| **F7-1** | Attention Chunking Pilot | ❌ **RESULT C (BYPASSED)** | Peak NVML **5,968.4 MB** ($C=2048$). No reduce pico dominante; atención descartada como cuello de botella · [`Report`](docs/F7_1_ATTENTION_CHUNKING_PILOT_REPORT_01.md) |
| **F7-D1**| Forensic Memory Profiling | 🟢 **COMPLETE** | Diagnóstico causal: Tensores vivos son solo **~2.15 GB**; el exceso es **~5.18 GB de `FreePool` inactivo** retenido por el allocator · [`Spec`](docs/F7_D1_FORENSIC_SPEC_01.md) |
| **F7-D2**| Allocator Causal Probes | 🟢 **COMPLETE** | Trazado a 8ms demostró no-reutilización de segmentos entre pases cond y uncond · knowledge acquired through conversation with an AI agent |
| **F7-D3**| Seam Release Causal Probe | 🟢 **FAVORABLE** | `empty_cache()` focalizado en costura cond→uncond baja pico NVML a **4,403.5 MB** (1 paso) · [`Report`](docs/F7_D3_COND_UNCOND_SEAM_RELEASE_REPORT_01.md) |
| **F7-D4**| Reduced Multi-Step Validation | 🟢 **FAVORABLE** | 10 pasos con costura cond→uncond: **Peak NVML 4,708.9 MB ($\le 4,800$ MB PASS)**, cadencia 59.89 s/p, 0 NaNs · [`Report`](docs/F7_D4_MULTISTEP_VALIDATION_REPORT_01.md) |
| **F7-D5**| Full 30-Step E2E Validation | 🔴 **PARTIAL/NEGATIVE** | 30 pasos continuos a 720p/33f con costura cond→uncond: **Peak NVML 5,096.1 MB ($\le 4,800$ MB FAIL)**, Reserved estabilizado 3,766 MB, 0 NaNs. Integración **NO autorizada** · [`Report`](docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md) |
| **F7-D6**| NVML Metric Attribution Probe | 🟢 **INSTRUMENT VALIDATED** | Probe corregido ejecutado: Control A/B diff **2.34% ($\le 10\%$ PASS)** (512 vs 500 MB), S0 estable (spread 146.5 MB), S2 4,624.0 MB. S3 post-workload caracterizado como meseta plana estática (~1,481.6 MB, spread 29 MB; +438 MB sobre S1). F7-D5 permanece NEGATIVE; gate 4.8 GB intacta · [`Report`](docs/F7_D6_METRIC_ATTRIBUTION_REPORT_01.md) · knowledge acquired through conversation with an AI agent |

---

## 🔬 Phase F0 — Baseline Results (2026-09-08)

The first real model run with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` completed successfully.

### VRAM Telemetry Snapshots

| Checkpoint | NVML Physical | PyTorch Allocated | Gate |
| :--- | :---: | :---: | :---: |
| Idle (pre-load) | 1261 MB | 0 MB | ✅ |
| Post-VAE load | 1287 MB | 0 MB | ✅ |
| Post-pipeline load | 1265 MB | 0 MB | ✅ |
| Post sequential-offload setup | 1264 MB | 1 MB | ✅ |
| **Post-inference peak (VAE decode)** | **5451 MB** | 15.9 MB | ❌ +651 MB |

### Performance

| Metric | Value |
| :--- | :--- |
| Resolution | 832×480 (480p) |
| Frames | 16 |
| Precision | FP16 |
| Offload | `enable_sequential_cpu_offload()` |
| Denoising (30 steps) | 297s · ~9.93 s/step |
| **VAE decode** | **322s** ← bottleneck |
| **Total inference time** | **619.8s (~10.3 min)** |
| Peak NVML | 5451 MB (gate exceeded by 651 MB) |
| OOM crash | ❌ None (WDDM paged the excess) |

> [!IMPORTANT]
> The 4.8 GB NVML gate was exceeded **only during VAE decode** (+4,187 MB delta).
> The DiT denoising phase itself fits comfortably within the budget.
> This pre-activates the **Phase F5 (Temporal VAE Stitcher)** trigger.

---

## 🚀 Phase F0.5 — VAE Memory Optimization & Benchmark Results (2026-09-08)

Following Phase F0, an experimental ladder of low-cost isolation tests was executed to resolve the VAE memory bottleneck without introducing unnecessary algorithmic complexity:

### Experimental Ladder & Scorecard

| Test ID | Strategy / Configuration | Peak NVML (50ms) | PyTorch Alloc | VAE Decode | Total Time | Gate (4.8 GB) | Report |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0 (Ref)** | `sequential_cpu_offload` + FP32 VAE | 5,451 MB | ~7.9 GB virt | 322s | 619.8s (10.3m) | ❌ +651 MB | [TEST_F0](results/TEST_F0_baseline_ref.md) |
| **Test G** | `model_cpu_offload` + FP32 VAE | 5,861 MB | 13,036.9 MB | 484s | 696.3s (11.6m) | ❌ +1,061 MB | [Test G Report](results/TEST_G_model_cpu_offload.md) |
| **Test H** | `sequential_cpu_offload` + BF16 VAE | 6,028 MB | 4,366.2 MB | 65s 🟢 | 339.0s (5.65m) | ⚠️ Post: 4,839 MB | [Test H Report](results/TEST_H_bfloat16_vae.md) |
| **Test J** | **`sequential_cpu_offload` + BF16 + VAE Tiling** | **2,902.0 MB** 🟢 | **2,021.1 MB** 🟢 | **58s** 🟢 | **328.2s (5.47m)** 🟢 | 🟢 **PASS (-1,898 MB)** | [Test J Report](results/TEST_J_vae_tiling_bf16.md) |

### Key Breakthroughs Achieved in Test J:
1. **Physical VRAM Plummeted to 2,902 MB:** The full pipeline now runs utilizing only **47.2% of the physical 6 GB VRAM**, leaving **~1.9 GB of free headroom** below the strict 4.8 GB gate.
2. **VAE Latency Decimated by 82%:** VAE decode time dropped from **322 seconds to 58 seconds**, eliminating the primary pipeline bottleneck.
3. **Total Generation Time Cut in Half:** Wall-clock runtime for 17 frames at 480p dropped from 10.3 minutes to **5.47 minutes**.
4. **Zero Visual Artifacts:** Native spatial tiling (256×256 px) with causal temporal caching preserved video smoothness and quality with zero NaNs.

All test reports and raw telemetry are centralized in the [`results/`](results/) directory.

---

## 🔀 Phase F0.6 — WDDM Concurrency Benchmark Results (2026-09-08)

Phase F0.6 measured whether the Windows WDDM driver scheduler permits genuine overlap of PCIe `cudaMemcpyAsync` (host→device) transfers with Tensor-Core `matmul` kernels on separate non-default CUDA streams.

| H2D Transfer | Copy Alone | Matmul Alone | Concurrent | **Overlap** | Gate (≥10%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 512 MB | 43.7 ms | 38.7 ms | 43.4 ms | **89.3%** | 🟢 **PASS** |
| 1024 MB | 87.2 ms | 69.2 ms | 86.8 ms | **79.9%** | 🟢 **PASS** |
| 2048 MB | 174.4 ms | 172.0 ms | 178.4 ms | **96.3%** | 🟢 **PASS** |

> **Key finding:** Concurrent makespan tracked the *copy-alone* time (not the serial sum), proving the WDDM copy engine runs **in parallel** with SM compute. With overlap of **79.9–96.3%** — an order of magnitude above the 10% gate — **Phase F3 (async prefetch scheduler) is justified and stays active**. Report: [`results/TEST_F0.6_wddm_overlap.md`](results/TEST_F0.6_wddm_overlap.md).

---

## 🔬 Phase F1 — Tensor Lifetime Profiler Results (2026-09-08)

A `TensorLifetimeProfiler` (`marley/profiler/lifetime.py`) hooked every leaf of the real Wan2.1 pipeline under sequential CPU offload (DiT=FP16, VAE=BF16+tiling) and attributed NVML physical residency to whichever component was actively driving execution, at 50 ms sampling.

### Per-Component Lifetime & Residency (480p · 17f · 30 steps)

| Component | Kind | Static weights | **Peak live NVML** | Activation Δ |
| :--- | :---: | :---: | :---: | :---: |
| text_encoder[0] | text_encoder | 14,758.5 MB | **3,223.7 MB** | 12.1 MB |
| 30 × diT_block[i] | diT_block | 88.6 MB each | 1,645.7–1,743.5 MB | 22.9–23.2 MB |
| vae (tiled decode) | vae | 242.0 MB | **2,112.1 MB** | 96.0 MB |

> **Key findings:** (1) Dynamic tracing is **viable** — the pure static analytical fallback is preserved but not required. (2) DiT block residency is **uniform** (1.65–1.74 GB) → low fragmentation prior (Phase F2 unlikely to trigger). (3) VAE peak is **activation-dominated** (2,112 MB on only 242 MB of weights). (4) **Peak NVML 3,223.7 MB** stays well under the 4.8 GB gate, keeping headroom for F3/F4. Report: [`results/TEST_F1_lifetime_profiler.md`](results/TEST_F1_lifetime_profiler.md).

---

## 🎯 Phase F1.5 — Real Bottleneck Identification & Roadmap Dispatch (2026-09-08)

Phase F1.5 ingested canonical Phase F1 telemetry to quantitatively classify memory into the 5 core categories and evaluate conditional roadmap triggers.

### 5-Category Memory Decomposition

| # | Category | Measured Footprint | Key Architectural Behavior |
| :---: | :--- | :---: | :--- |
| **1** | **Static Model Weights** | **17,658.5 MB total** (88.6 MB active) | T5 Text Encoder is 83.6% of weights. Under sequential offload, only ~88.6 MB resides on GPU. |
| **2** | **DiT Attention Activations** | **~550–600 MB** (1,743.5 MB peak) | Highly uniform across 30 blocks (delta < 98 MB). Per-block activation delta is ~22.8 MB. |
| **3** | **VAE Decoding Activations** | **~770 MB pure** (2,112.1 MB peak) | Activation-dominated (8.7× weights), but fully contained via 256×256 spatial micro-tiling. |
| **4** | **Allocator Fragmentation** | **44.9 MB (< 1.0% of Gate)** | Measured delta between PyTorch reserved and allocated. Trigger threshold (>15%) refuted. |
| **5** | **Prefetch Buffers (F3)** | **88.6 MB buffer needed** | Consumes only **5.6%** of +1,576.3 MB free headroom, leaving +1,487.7 MB safety buffer. |

### Binding Dispatch Verdicts
1. **Phase F2 (Static Slab Allocator) → ❌ BYPASSED / DISCARDED:** Fragmentation is < 1% (44.9 MB vs 720 MB threshold). PyTorch's allocator is optimal.
2. **Phase F3 (Budgeted Async Scheduler) → 🟢 GREENLIT / APPROVED:** Overlap of 80–96% + 88.6 MB prefetch safe under the 4.8 GB gate. Will overlap PCIe transfers to reduce the 455s inference latency.
3. **Phase F1.7 (Selective Quantization) → 🟡 PRIORITIZED FOR T5:** Quantizing the 14.7 GB T5 text encoder to reduce system RAM demand.

Full technical report: [`results/TEST_F1.5_bottleneck_classification.md`](results/TEST_F1.5_bottleneck_classification.md) · Telemetry: [`logs/f1_5_bottleneck_decomposition.json`](logs/f1_5_bottleneck_decomposition.json).

---

## ⚡ Phase F1.7 — Selective & Adaptive Quantization Results (2026-09-08)

Phase F1.7 evaluated selective 8-bit / 4-bit precision scaling targeted at the dominant weight bottlenecks identified in F1.5 without introducing perceptual or numerical degradation.

### Quantization Benchmark & Numerical Fidelity

| Component | Precision Mode | Memory Footprint | Savings | Cosine Similarity | RMSE | Zero NaNs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Encoder (UMT5-XXL)** | FP16 Baseline | 14,758.5 MB | 0.0% | 1.000000 | 0.000 | ✅ |
| **Text Encoder (UMT5-XXL)** | **INT8 (`load_in_8bit`)** | **7,379.2 MB** | **-50.0% (-7.38 GB)** | **1.000017** | 0.011 | ✅ |
| **Text Encoder (UMT5-XXL)** | NF4 (`load_in_4bit`) | 3,689.6 MB | -75.0% (-11.07 GB) | 0.980313 | 0.048 | ✅ |
| **DiT Unit Block (Linear)** | FP16 Baseline | 88.60 MB | 0.0% | 1.000000 | — | ✅ |
| **DiT Unit Block (Linear)** | **8-Bit Projections** | **45.89 MB** | **-48.2% (-42.7 MB)** | **0.999610** | 0.003 | ✅ |

> **Key Takeaway:** 8-bit text encoding saves **7.38 GB of host memory** with perfect semantic fidelity ($> 0.99$), eliminating paging on 16–24 GB laptops. 8-bit DiT projections save **-76.9 GB of cumulative PCIe transfers** across 30 denoising steps.  
> **Kill Gate F1.7 Status:** 🟢 **PASS**. Full report: [`results/TEST_F1.7_selective_quantization.md`](results/TEST_F1.7_selective_quantization.md) · Benchmark script: [`f1_7_selective_quantizer.py`](f1_7_selective_quantizer.py).

---

## ⚡ Phase F3 — Budgeted Asynchronous Scheduler (🟢 CERTIFIED PASS)

Phase F3 replaces synchronous per-block weight swapping with an **asynchronous double-buffered scheduler** (`marley.ops.async_stream.BudgetedAsyncStreamer`) on dual CUDA streams (`copy_stream`, `compute_stream`) with a bidirectional event-ownership protocol.

### Certified result (2026-09-08, measurement-corrected)

Following a post-certification audit, the overlap/stall instrumentation and benchmark were corrected (Enmiendas 1–3, [`docs/F3_MEASUREMENT_VERIFICATION_03.md`](docs/F3_MEASUREMENT_VERIFICATION_03.md)) and a repeated A/B verification battery was executed on the real Wan2.1-T2V-1.3B DiT (30 blocks · FP16 · 480p/17f). Full report: [`results/TEST_F3_verification.md`](results/TEST_F3_verification.md) · Telemetry: [`logs/f3_verification_benchmark.json`](logs/f3_verification_benchmark.json).

| Metric | Marley Sync | Marley Async |
| :--- | ---: | ---: |
| External wall-clock (mean, 5 reps) | 18,127.4 ms | **16,824.1 ms** |
| **Speedup (mean / min / max / σ)** | — | **+7.2%** / +6.0% / +8.4% / ±1.1% |
| Effective overlap (measured) | 0.0% | **100.0%** (stall 0.00 ms, per-block timeline) |
| Peak VRAM (NVML) | — | 2,584 MB |
| NaN / Inf | 0 | 0 |
| Forced stream syncs | 150 | 5 |

- **Order alternated A/B ↔ B/A with discarded warmup** → the +7.2% is stable and reproducible (no regression in any of the 5 reps).
- The initial single-run **+10.4%** was an optimistic sample; the corrected reproducible figure is **+7.2%**. The earlier "100% overlap" was an instrumentation artifact, now re-confirmed **empirically** (copies ~43 ms ≪ compute ~550 ms per block → zero GPU stall).
- **Verdict:** 🟢 **F3 PASS — Certified. Advance to F3 + INT8.**

First-run (preliminary) reference: [`results/TEST_F3_async_scheduler.md`](results/TEST_F3_async_scheduler.md).

---

## 🧠 Phase F3 + INT8 — Transfer-Volume Isolation (🟢 PASS, baseline frozen)

F3+INT8 isolated a single variable — the per-block H2D byte volume — by quantizing DiT linear
projections to INT8 (FP16 compute, on-device dequant) on the **unchanged** F3 async scheduler.

| Metric | Value |
| :--- | :---: |
| H2D payload / block | 46.5 MB (**−49.9%** vs 92.9 MB FP16) |
| Async-vs-Sync wall-clock | **+13.2%** (min +10.4 / max +15.4, σ ±2.5) |
| Effective overlap | 98.1% |
| Peak VRAM (NVML) | **2,076 MB** (−508 vs FP16) |
| Dequant fidelity (float64) | cos mean 0.999959 / min 0.999935 · 0 NaN/Inf |

Residual ~324 ms dequant stall on the critical path became the **F4 decision signal**.
Report: [`results/TEST_F3_INT8_benchmark.md`](results/TEST_F3_INT8_benchmark.md) · Closure:
[`docs/F3_INT8_CLOSURE_01.md`](docs/F3_INT8_CLOSURE_01.md) · F3 FP16 baseline remains frozen.

---

## ⚙️ Phase F4 — Adaptive Memory Decision Engine (🟢 CORE SYSTEM VALIDATED)

F4 adds a **decision layer** over the frozen F3/F3+INT8 primitives (no scheduler/dequant changes —
causal attribution preserved). It observes memory/transfer state and selects, per decision window,
`precision × prefetch × residency`, delegating execution to `BudgetedAsyncStreamer` (FP16) or
`INT8BudgetedStreamer` (INT8). Code: [`marley/core/policies.py`](marley/core/policies.py) +
[`marley/core/adaptive.py`](marley/core/adaptive.py) · Runner: [`f4_adaptive_benchmark.py`](f4_adaptive_benchmark.py)
· Authorized contract: [`docs/F4_TEST_SPEC_01.md`](docs/F4_TEST_SPEC_01.md).

### Same-session A/B/C/D — 30 steps × 3 reps (external wall-clock)

| Condition | Strategy | Mean (ms) | Overlap | NaN/Inf |
| :--- | :--- | :---: | :---: | :---: |
| **A** | Sync FP16 | 113,538.7 | 0.0% | 0/0 |
| **B** | Async FP16 | **105,441.5** | 100.0% | 0/0 |
| **C** | Async INT8 | 109,157.9 | 98.1% | 0/0 |
| **D** | Adaptive | 109,022.8 | ~88.4% | 0/0 |

- **Best static = B (Async FP16).** **D vs best = −3.40%** (no-pressure regime: D correctly stayed
  on its `performance` INT8 policy because no switch was demanded → D ≈ C). **Not a refutation of
  F4** — D was never asked to adapt.
- **Decision overhead:** 0.16 ms total · switch 0 · replan 0 · **Peak VRAM 2,882 MB**
  (Hard Gate ≤ 4,800 PASS · Engineering Target ≤ 4,000 MET).
- **Key experimental finding:** **Async FP16 ≈3.5% faster than Async INT8** at this length (dequant
  on the critical path) — INT8 is **not universally superior**; its value depends on memory state,
  which is precisely the adaptive rationale.

### Adaptive Gate — `--pressure-test` (deterministic injected pressure)

`performance → memory_safe → performance` cycle reproduced with **replan 2 · switch 2 ·
overhead 0.07 ms · oscillation False → 🟢 PASS** (hysteresis + minimum-dwell validated).

> **Honest caveats (kept per Consejero §5):** not yet demonstrated = D faster than the best static
> policy; consistent ≥5% advantage; benefit under **real** (vs injected) memory pressure; and the
> `INT8-tighten` branch below `SAFE_MIN_FREE_MB` (the simulated +600 MB spike kept free VRAM
> > 1,500 MB). Verdict: **CORE SYSTEM VALIDATED — PERFORMANCE CERTIFICATION PENDING.**

Full report: [`results/TEST_F4_adaptive_benchmark.md`](results/TEST_F4_adaptive_benchmark.md) ·
Telemetry: [`logs/f4_adaptive_benchmark.json`](logs/f4_adaptive_benchmark.json) · Change analysis:
[`docs/F4_IMPLEMENTATION_CHANGES_01.md`](docs/F4_IMPLEMENTATION_CHANGES_01.md).

---

## 🎞️ Phase F5 — Temporal VAE Stitcher (🟢 RETIRED · Probe F5-A PASS)

Per the **Decision-First** protocol agreed with the Director and External Consultant (knowledge acquired through conversation with an AI agent), Phase F5 executed the canonical 33-frame probe before building any temporal chunker.

### Canonical 33-Frame VAE Decode (F5-A)

- **Workload:** $832 \times 480$ (480p) · **33 frames exact** · Latent `(1, 16, 9, 60, 104)` · `torch.bfloat16`.
- **Configuration:** Native `enable_tiling()` with $256 \times 256$ spatial tiles and causal temporal caching.

| Metric | Target / Gate | Measured (F5-A) | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak VRAM (NVML 50ms)** | $\le 4,800\text{ MB}$ (Hard Gate) | **2,109.0 MB** | 🟢 **PASS (+2,691 MB headroom)** |
| **Engineering Target VRAM** | $\le 4,000\text{ MB}$ | **2,109.0 MB** | 🟢 **MET** |
| **VAE Decode Duration** | $\le 150.0\text{ s}$ | **26.99 s** | 🟢 **MET (5.5× faster than budget)** |
| **Integrity (NaNs / Infs)** | $0 / 0$ | **0 / 0** | 🟢 **PASS** |
| **Output Shape** | `[1, 3, 33, 480, 832]` | `[1, 3, 33, 480, 832]` | 🟢 **MATCH** |
| **Host Process RAM (RSS)** | Audited | **1,402.6 MB** | 🟢 Clean (no thrashing) |

**Conclusion & Action:**  
Because the existing tiled VAE path (`bfloat16` + $256 \times 256$ spatial tiling) decodes 33 frames in **26.99 seconds** using only **2,109 MB** of VRAM, building a custom temporal stitcher (`marley/ops/vae_stitch.py`) is completely unnecessary. F5-C is **CANCELLED / RETIRED** as resolved by existing tiling. Phase F5 is formally closed; project advances directly to **Phase F6 (480p / 33f End-to-End Pipeline)**.

> [!IMPORTANT]
> **VAE Non-Regression Rule (Binding Directive):**  
> Any future changes in Phase F6 pipeline integration must maintain the canonical F5-A baseline: physical peak VRAM $\le 2,109\text{ MB}$ (noise ceiling $\le 2,300\text{ MB}$) and decode latency $\approx 27\text{ s}$ (ceiling $\le 35\text{ s}$).  
> The VAE is no longer considered a project bottleneck; 100% of technical risk is now shifted to the **DiT denoising loop (30 steps)** at 33 frames.

Full report: [`results/TEST_F5_vae_probe_33f.md`](results/TEST_F5_vae_probe_33f.md) · Telemetry: [`logs/f5_vae_probe_33f.json`](logs/f5_vae_probe_33f.json) · Probe: [`f5_vae_probe_33f.py`](f5_vae_probe_33f.py) · Closure Resolution: [`docs/F5_CLOSURE_RESOLUTION_01.md`](docs/F5_CLOSURE_RESOLUTION_01.md).

---

## 🏆 Phase F6 — Multidimensional Benchmarks & Multirun 4×3 (🟢 CERTIFIED PASS & FROZEN)

Phase F6 evaluated the end-to-end pipeline across 5 dimensions at **480p ($832 \times 480$), 33 frames, 30 denoising steps**.

Following the single-run benchmarks (F6-0 through F6-E), the project executed the formal **Campaña Multirun Core 4×3 de Reproducibilidad y Variabilidad** ($n=3$, $df=2$, 12 corridas independientes contrabalanceadas en 3 rondas con 60s de enfriamiento térmico entre ejecuciones).

### Resumen Estadístico Multirun Core 4×3

| Streaming Policy | Wall-Clock (Media μ) | DiT Denoise (Media μ) | Cadencia (Media μ) | Peak NVML (Media μ) | CV% Latencia | CV% VRAM | Rango Peak NVML [Min–Max] | Gate Individual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sync FP16** | 531.65 s | 450.94 s | 15.03 s/p | 2,777.4 MB | 3.03% | 3.54% | [2,719.5 – 2,890.8] MB | 🟢 100% PASS |
| **Async FP16** | 515.57 s | 429.66 s | 14.32 s/p | 2,768.3 MB | 2.64% | 3.08% | [2,719.0 – 2,866.8] MB | 🟢 100% PASS |
| **Async INT8** | **512.21 s** | 433.24 s | 14.44 s/p | **2,741.4 MB** | **1.94%** | **0.17%** | [2,738.5 – 2,746.7] MB | 🟢 100% PASS |
| **Adaptive** | 527.07 s | **427.63 s** | **14.25 s/p** | 4,047.8 MB | 4.56% | 1.75% | [4,007.0 – 4,129.5] MB | 🟢 100% PASS |

- **Criterio Individual Estricto:** 12 de 12 corridas (100%) superaron el Hard Gate físico $\le 4,800.0\text{ MB}$ y el objetivo de tiempo $< 600\text{ s}$.
- **Integridad Numérica:** 0 NaNs / 0 Infs en el 100% de las repeticiones; todos los videos MP4 generados fueron decodificables.
- **Dictamen:** Phase F6 certificada formalmente, **CERRADA Y CONGELADA** bajo el commit `c6e4bfb`.
- **Reportes:** [`docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md`](docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md) · [`docs/F6_FINAL_CERTIFICATION_REPORT_01.md`](docs/F6_FINAL_CERTIFICATION_REPORT_01.md) · Telemetría: [`logs/multirun/f6_statistical_summary.json`](logs/multirun/f6_statistical_summary.json).

---

## 🔍 Phase F7 — 720p Strategy & Forensic Memory Profiling

### F7-0: 720p Memory Feasibility Probe (Baseline Inmutable)
- **Workload:** 1280×720 (720p, 16:9), 33 frames, 5 pasos exploratorios, modo `adaptive`.
- **Área Latente Espacial:** $90 \times 160 = 14,400$ ($2.31\times$ frente a 480p).
- **Resultados Medidos:**
  - **Peak Físico NVML:** **6,058.5 MB** frente al Hard Gate $\le 4,800.0\text{ MB}$ (**❌ FAIL**, exceso de $+1,258.5\text{ MB}$ / $+26.2\%$).
  - **PyTorch Allocated:** 2,187.5 MB | **PyTorch Reserved:** 5,894.0 MB ($\Delta = 3,706.5\text{ MB}$).
  - **Cadencia DiT:** 68.75 s/paso (frente a 14.25 s/paso en F6). VAE Tiled decode: 69.13 s.
  - **Integridad:** 0 NaNs / 0 Infs (**🟢 PASS**), video MP4 decodificable (**🟢 PASS**).
- **Conclusión:** Ejecutabilidad computacional de 720p demostrada; viabilidad física $\le 4,800\text{ MB}$ no cumplida. Baseline F7-0 **congelado e inmutable** ([`docs/F7_0_FEASIBILITY_PROBE_REPORT_01.md`](docs/F7_0_FEASIBILITY_PROBE_REPORT_01.md)).

### F7-1: Attention Sequence Chunking Pilot (❌ RESULT C — BYPASSED)
- **Hipótesis:** La atención densa a $14,400$ latentes es el detonante de la sobre-reserva.
- **Resultado:** Chunking $C=2048$ arrojó un pico NVML de **5,968.4 MB** (reducción marginal de ~90 MB).
- **Conclusión:** La atención no constituye el cuello de botella dominante. Subfase cerrada sin alterar el runtime ([`docs/F7_1_ATTENTION_CHUNKING_PILOT_REPORT_01.md`](docs/F7_1_ATTENTION_CHUNKING_PILOT_REPORT_01.md)).

### F7-D1: Forensic Memory Profiling (🟢 COMPLETE)
- **Desglose de Memoria:** Tensores vivos reales (`Allocated`) alcanzan solo **~2.15 GB**. El exceso físico a ~6.0 GB está dominado por **~5.18 GB de `FreePool` inactivo (`Reserved` = 5,778 MB)** retenido por el PyTorch Caching Allocator y anclado físicamente por WDDM.
- **Especificación & Reporte:** [`docs/F7_D1_FORENSIC_SPEC_01.md`](docs/F7_D1_FORENSIC_SPEC_01.md).

### F7-D2 / F7-D2b / F7-D2c: Allocator Causal & Live Boundary Probes (🟢 COMPLETE)
- **Hallazgo Causal:** Trazado a resolución de ~8 ms determinó que durante el segundo pase DiT (`uncond_blocks_30`), PyTorch no reutiliza los bloques de memoria liberados por `cond_blocks_30`, duplicando la reserva a ~5.8 GB.
- **Reportes:** knowledge acquired through conversation with an AI agent.

### F7-D3: Cond→Uncond Seam Release Causal Probe (🟢 FAVORABLE)
- **Intervención:** Liberación focalizada (`empty_cache()`) exclusivamente en la frontera sincrónica entre el pase condicional e incondicional (`cond → uncond`).
- **Resultado (1 paso):** Pico NVML se redujo de 6,006.2 MB a **4,403.5 MB** (🟢 **PASS** con +396.5 MB de holgura). Duración del flush: solo 148 ms.
- **Reporte:** [`docs/F7_D3_COND_UNCOND_SEAM_RELEASE_REPORT_01.md`](docs/F7_D3_COND_UNCOND_SEAM_RELEASE_REPORT_01.md).

### F7-D4: Reduced Multi-Step Seam-Release Validation (🟢 FAVORABLE)
- **Workload:** 10 pasos de difusión continuos a 720p/33f con liberación en costura `cond → uncond`.
- **Resultado:**
  - **Peak Físico NVML:** **4,708.9 MB** ($\le 4,800.0\text{ MB}$ Hard Gate **PASS**).
  - **Peak Reserved:** 3,776.0 MB (acumulación neta de solo 248 MB del paso 1 al 10; pool completamente acotado).
  - **Cadencia:** 59.89 s/paso (598.91 s en denoise). VAE decode: 69.57 s. 0 NaNs/Infs, MP4 generado correctamente.
- **Reporte:** [`docs/F7_D4_MULTISTEP_VALIDATION_REPORT_01.md`](docs/F7_D4_MULTISTEP_VALIDATION_REPORT_01.md).

### F7-D5: Full 30-Step E2E Validation & Integration (🔴 PARTIAL/NEGATIVE)
- **Workload:** 1280×720 @ 33 frames, **30 diffusion steps completos**, modo `adaptive` con liberación en costura `cond → uncond` + drenaje al inicio de cada paso (config FAVORABLE de F7-D4), seguido de VAE.
- **Resultado:**
  - **Peak Físico NVML:** **5,096.1 MB** ($\le 4,800.0\text{ MB}$ Hard Gate **FAIL**, margen $-296.1\text{ MB}$; safe-abort a >5,050 MB disparado, sin OOM).
  - **Peak Reserved:** 3,776.0 MB — allocator **totalmente estabilizado** en ~3,766 MB del paso 1 al 30 (acumulación neta +246 MB, sin divergencia).
  - **Cadencia:** 62.13 s/paso (1,864.1 s denoise); VAE decode 68.06 s; wall-clock total 2,667.7 s (44.46 min, $\le 45$ min gate). **0 NaNs/Infs**, MP4 generado correctamente.
- **Conclusión:** La liberación en costura **previene la divergencia del allocator** a 30 pasos, pero el delta físico NVML−Reserved (~1,130 MB, residencia WDDM/`FreePool` no devuelta al driver) mantiene el pico real **por encima de 4,800 MB**. Kill-gate F7-D5 activado → **720p permanece experimental; la integración en `marley/core/pipeline.py` NO se autoriza** por este resultado.
- **Reporte:** [`docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md`](docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md).

### F7-D6: NVML Metric Attribution Probe (🔴 INSTRUMENT NOT VALIDATED)
- **Motivación:** F7-D5 mostró un allocator **estable y acotado** (`reserved` ~3,766 MB en 30 pasos) mientras la métrica física del **dispositivo** (NVML `used`) alcanzó 5,096.1 MB (delta ~1,320 MB). Esto motivó investigar si parte del "pico" **no es atribuible al proceso**.
- **Advertencia del Consejero:** `usedGpuMemory = None` por PID bajo WDDM **NO significa 0 MB**; significa que NVML no expone el contador por proceso. Por tanto una gate "PID NVML" **no es viable**.
- **Protocolo autorizado (Consejero no.1–4):** `S0 idle 60 s → Control A → init → S1 → Wan 3 pasos → S3 30 s → Control B → S4 30 s`; `S2−S0` = *workload-induced device residency delta* (no ownership); clasificación de 3 niveles con tolerancia de control ±10%.
- **Resultado (2026-09-09, ejecutado):**
  - **S0 idle:** **983.0 MB**, spread **0.25 MB** (baseline muy estable).
  - **Control A / B:** 592.5 / 500.1 MB → **diferencia relativa 15.59% (>10%)** (Control A incluye el coste one-time de init del contexto CUDA; A corre antes del contexto).
  - **S1 operacional:** 1,095.4 MB · **S2 workload peak:** 4,883.3 MB · **delta (S2−S0): 3,900.3 MB**.
  - **S3 post-workload:** 1,887.2 MB (spread 349.8 MB) → **no retorna al baseline** (`|S3−S0|` = 904.2 MB). **S4 final:** 1,902.9 MB (spread 235.6 MB). `cudaMemGetInfo` coherente.
  - **Veredicto: 🔴 INSTRUMENT NOT VALIDATED** (Control A/B no reproducible >10%; residencia persistente S3/S4).
- **Conclusión:** El idle baseline es muy estable, pero el instrumento **no quedó validado**: (a) discrepancia A/B atribuible al ordenamiento del protocolo (Control A pre-contexto) y (b) **residencia device-wide persistente/ondulante tras el workload** (~900 MB sobre S0). Esto **no habilita** el re-run de 30 pasos ni ninguna reclasificación. **F7-D5 permanece NEGATIVE; gate 4,800 MB intacta; sin integración de 720p.**
- **Veredicto y Resolución Metodológica (2026-09-10):** El Consejero concedió GO definitivo para el probe corregido (`F7-D6 Corrected Attribution Probe`), aprobando la separación metrológica de S3, Control A pre-pipeline con warm-up de 10 MB, tratamiento auxiliar de NVML per-process y ventanas de 60s/120s/60s a 1 Hz.
- **Resultado de la Ejecución Corregida (2026-09-10, EJECUTADO):**
  - **S0 idle:** **962.2 MB**, spread **146.47 MB** ($\le 150\text{ MB PASS}$).
  - **Control A / B:** **512.0 / 500.0 MB** → **diferencia relativa 2.34% ($\le 10\%$ PASS)**. Demuestra de forma concluyente que la discrepancia de 15.6% previa era un artefacto del protocolo.
  - **S1 operacional:** 1,043.57 MB · **S2 workload peak:** 4,624.0 MB · **delta inducido (S2−S0): 3,661.8 MB**.
  - **S3 post-workload (120 s @ 1 Hz):** media **1,481.64 MB**, spread **29.0 MB** (meseta estática horizontal plana; +438 MB sobre S1). Falsa la hipótesis de liberación diferida lenta.
  - **S4 final (60 s @ 1 Hz):** media **1,511.74 MB**, spread **86.84 MB** ($\le 150\text{ MB PASS}$).
  - **Allocator PyTorch:** Peak Reserved **3,606.0 MB** (estable y acotado); Peak Allocated **2,171.2 MB**.
  - **Veredicto:** 🟢 **INSTRUMENT VALIDATED**.
  - *Gobernanza inalterada:* F7-D5 permanece `NEGATIVE` (5,096 MB); gate en 4,800 MB congelada; sin re-run de 30 pasos ni integración de 720p.
- **Documentos:** [`Report`](docs/F7_D6_METRIC_ATTRIBUTION_REPORT_01.md) · [`Spec`](docs/F7_D6_METRIC_ATTRIBUTION_SPEC_01.md) · knowledge acquired through conversation with an AI agent · [`Telemetry`](logs/f7_d6_attribution_probe_telemetry.json) · Runner [`f7_d6_attribution_probe.py`](f7_d6_attribution_probe.py).

---

## 🏗️ Architecture & Core Components

```
                    ┌────────────────────────────────────────────────┐
                    │               marley-runtime                   │
                    └───────────────────────┬────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│     marley.core       │       │    marley.profiler    │       │      marley.ops       │
│  - Adaptive Engine    │       │  - NVML Monitor       │       │  - Slab Allocator     │
│  - Layer Scheduler    │       │  - Lifetime Profiler  │       │  - Async Streamer     │
│  - Cost Model         │       │  - WDDM Benchmark     │       │  - VAE Stitcher       │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

- **Adaptive Memory Decision Engine (`marley.core`):** Block/layer granularity scheduling using analytical cost models augmented with live NVML telemetry.
- **NVML Profiler (`marley.profiler`):** Physical GPU residency sampling · WDDM PCIe/compute overlap benchmarking.
- **Static Slab Allocator (`marley.ops.slab`):** Zero-fragmentation ring-buffer pool (conditional: fragmentation > 15%).
- **Async Stream Scheduler (`marley.ops.async_stream`):** Double-buffered PCIe prefetch on dedicated CUDA streams (conditional: WDDM overlap ≥ 10%).
- **Temporal VAE Stitcher (`marley.ops.vae_stitch`):** Chunked temporal latent decode with frame blending (triggered: VAE confirmed as bottleneck in F0).

---

## ⚡ Quick Start

### Prerequisites

- Windows 10/11 · NVIDIA GPU driver 550+ (tested: 581.86)
- Python 3.10 or 3.11
- ~10 GB free disk (model cache ~5 GB + venv ~3 GB)
- Git

### Installation

```bash
git clone https://github.com/LARP/marley-runtime.git
cd marley-runtime

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### Run Inference & Benchmarks

```bash
# 🟢 Optimal Phase F0.5 configuration (Peak NVML: 2,902 MB · 58s VAE decode · 5.4 min total)
python f0_baseline_real.py --res 480p --frames 17 --dtype fp16 --offload cpu --vae-dtype bf16 --vae-tiling

# Sequential offload with FP32 VAE (Baseline F0 — 5,451 MB peak · 322s VAE decode)
python f0_baseline_real.py --res 480p --frames 17 --dtype fp16 --offload cpu --vae-dtype fp32

# Model-level offload (Test G — 6.3s/it DiT, but 13 GB virtual VAE paging in FP32)
python f0_baseline_real.py --res 480p --frames 17 --dtype fp16 --offload model_cpu
```

> [!NOTE]
> On first run the model (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, ~5 GB) is downloaded
> automatically from HuggingFace Hub and cached locally.
> Use `num_frames` values where `(frames - 1) % 4 == 0` (e.g. 5, 9, 13, 17, 21…).

Outputs are saved to:
- `logs/f0_*.mp4` — generated video
- `logs/f0_*_telemetry.json` — VRAM metrics (all 3 layers + 50ms continuous NVML tracker)
- `results/` — markdown technical reports for each benchmark run

---

## 📂 Repository Structure

```
marley-runtime/
├── results/                 # Benchmarks, test reports & validation logs
│   ├── README.md            # Central test registry & gate scorecard
│   ├── TEST_G_*.md          # Model-level offload test report
│   ├── TEST_H_*.md          # bfloat16 VAE precision test report
│   └── TEST_J_*.md          # VAE spatial-temporal tiling report (F0.5 PASS ✅)
├── logs/                    # Profiling run outputs (gitignored: *.mp4, *.json, *.log)
│   └── .gitkeep
├── benchmarks/              # Standardized benchmark suites
├── marley/                  # Core package
│   ├── core/                # Adaptive engine & layer scheduler
│   ├── models/              # Wan2.1 / DiT model adapters
│   ├── ops/                 # Slab allocator, async streams, VAE stitcher
│   └── profiler/            # Tensor lifetime & residency profiler (Phase F1)
├── tests/                   # Unit and integration tests
├── baseline_profiler.py     # Synthetic stress profiler (Phase F0 OOM boundary)
├── f0_baseline_real.py      # Real Wan2.1 pipeline baseline & profiler (F0/F0.5 ✅)
├── f1_lifetime_profiler.py  # Phase F1 tensor lifetime / residency profiler
├── requirements.txt         # Python dependencies (torch cu124, diffusers, etc.)
├── LICENSE                  # MIT License
├── README.md                # This file
└── ROADMAP.md               # Implementation Roadmap v5
```

---

## 🔬 Scientific & Engineering Principles

1. **Evidence Precedes Optimization** — No optimization without prior empirical profiling data.
2. **Kill Gates** — Every conditional phase has measurable, non-negotiable exit criteria.
3. **Falsifiable Hypotheses** — We test whether adaptive runtime scheduling outperforms static offloading within a strict 4.8 GB physical VRAM envelope.
4. **WDDM Adaptation** — PCIe/compute concurrency is measured empirically under the WDDM driver scheduler before any async scheduling is committed.
5. **Target Primacy** — 480p @ 33 frames is the primary objective; 720p is a secondary stretch goal.

---

## 🤝 Provenance & Attribution

The architecture, phased progression, and metric definitions of `marley-runtime` were shaped through an iterative multi-agent engineering protocol combining **ChatGPT**, **DeepSeek Pro**, and **Gemini Pro**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
