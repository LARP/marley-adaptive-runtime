# marley-runtime

<div align="center">

**Adaptive Memory Management Runtime for Video Diffusion Models on Ultra-Low VRAM GPUs**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.6 CUDA 12.4](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA%2012.4-ee4c2c.svg)](https://pytorch.org/)
[![Target: RTX 3050 Laptop](https://img.shields.io/badge/Hardware-RTX%203050%206GB%20Laptop-76b900.svg)](https://www.nvidia.com/)
[![Roadmap: v5 Approved](https://img.shields.io/badge/Roadmap-v5%20Approved-8b5cf6.svg)](ROADMAP.md)
[![Phase F0: PASS](https://img.shields.io/badge/Phase%20F0-PASS-22c55e.svg)](ROADMAP.md)

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
3. **NVML physical residency** — hardware ground-truth · **the strict 4.8 GB gate applies here**

---

## 📊 Current Phase Status

| Phase | Name | Status | Key Result |
| :---: | :--- | :---: | :--- |
| **F0.1** | Hardware Verification | 🟢 **PASS** | Driver 581.86 · CUDA 13.0 · RTX 3050 6GB confirmed |
| **F0** | Reproducible Baseline | 🟢 **PASS** | Wan2.1-T2V-1.3B run at 480p/16f/FP16 — video generated |
| **F0.5** | Progressive Exploration | 🔵 **NEXT** | INT8/INT4 quantization · target NVML ≤ 4.8 GB |
| **F0.6** | WDDM Concurrency | ⚪ Scheduled | — |
| **F1** | Lifetime Profiler | ⚪ Scheduled | — |
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

### Run Phase F0 Baseline (Real Model)

```bash
# 480p / 16 frames / FP16 / sequential CPU offload (recommended — lowest VRAM)
python f0_baseline_real.py --res 480p --frames 17 --dtype fp16 --offload cpu

# Without offload (all weights on GPU — will OOM on 6GB)
python f0_baseline_real.py --res 480p --frames 17 --offload none

# Model-level offload (faster than sequential, higher VRAM peak)
python f0_baseline_real.py --res 480p --frames 17 --offload model_cpu
```

> [!NOTE]
> On first run the model (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, ~5 GB) is downloaded
> automatically from HuggingFace Hub and cached locally.
> Use `num_frames` values where `(frames - 1) % 4 == 0` (e.g. 5, 9, 13, 17, 21…).

Outputs are saved to `logs/`:
- `f0_*.mp4` — generated video
- `f0_*_telemetry.json` — VRAM metrics (all 3 layers)
- `f0_real_*.log` — full run log with per-checkpoint snapshots

---

## 📂 Repository Structure

```
marley-runtime/
├── logs/                    # Profiling run outputs (gitignored: *.mp4, *.json, *.log)
│   └── .gitkeep
├── benchmarks/              # Standardized benchmark suites
├── marley/                  # Core package
│   ├── core/                # Adaptive engine & layer scheduler
│   ├── models/              # Wan2.1 / DiT model adapters
│   ├── ops/                 # Slab allocator, async streams, VAE stitcher
│   └── profiler/            # NVML telemetry & WDDM benchmarks
├── tests/                   # Unit and integration tests
├── baseline_profiler.py     # Synthetic stress profiler (Phase F0 OOM boundary)
├── f0_baseline_real.py      # Real Wan2.1 pipeline baseline (Phase F0 ✅)
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
