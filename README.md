# marley-runtime

<div align="center">

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

*In loving memory of Marley 🐾*

</div>

---

## 📌 Executive Summary

**`marley-runtime`** is a specialized, experimental inference runtime designed to execute advanced video diffusion models (specifically **Wan2.1-T2V-1.3B**) under severe consumer hardware constraints: **~4.8 GB of effective physical VRAM** on an NVIDIA GeForce RTX 3050 6GB Laptop GPU running under Windows WDDM.

Rather than relying purely on blunt sequential CPU offloading, `marley-runtime` investigates and measures **adaptive, block-level memory management policies** that dynamically balance PCIe transfer latency, activation recomputation, tensor lifetime management, and selective quantization.

> [!NOTE]
> The current architectural design and phased milestone strategy is governed by **[Roadmap v5](ROADMAP.md)**, reviewed and approved through a multi-agent consensus using **free-tier models** (ChatGPT, DeepSeek Pro, and Gemini Pro). Active execution is performed by **Antigravity Pro** (Gemini 2.5 Flash and Claude Sonnet 4.6).

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
| **F0** | Reproducible Baseline | 🟢 **PASS** | Wan2.1-T2V-1.3B run at 480p/16f/FP16 — video generated |
| **F0.5** | Progressive Exploration | 🟢 **PASS** | **Peak NVML 2,902 MB (+1,898 MB headroom)** · VAE BF16 + Tiling |
| **F0.6** | WDDM Concurrency | 🟢 **PASS** | **Overlap 79.9–96.3%** · PCIe async copy hides compute → F3 ACTIVE |
| **F1** | Lifetime Profiler | 🟢 **PASS** | **Peak NVML 3,224 MB** · 32 components traced (text/DiT×30/VAE) → F1.5 NEXT |
| **F1.5** | Bottleneck Analysis | ⚪ Scheduled | — |
| **F1.7** | Selective Quantization | ⚪ Conditional | — |
| **F2** | Static Slab Allocator | ⚪ Conditional | fragmentation > 15% |
| **F3** | Async Stream Scheduler | ⚪ Conditional | WDDM overlap ≥ 10% |
| **F4** | Adaptive Decision Engine | ⚪ CORE | — |
| **F5** | Temporal VAE Stitcher | 🟡 **TRIGGERED** | VAE confirmed as VRAM bottleneck in F0 |
| **F6** | Validation Benchmarks | ⚪ Final | — |

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
| **F0 (Ref)** | `sequential_cpu_offload` + FP32 VAE | 5,451 MB | ~7.9 GB virt | 322s | 619.8s (10.3m) | ❌ +651 MB | [F0 Baseline](logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json) |
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
