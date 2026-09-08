# marley-runtime

<div align="center">

**Adaptive Memory Management Runtime for Video Diffusion Models on Ultra-Low VRAM GPUs**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.4+ CUDA 12.4](https://img.shields.io/badge/PyTorch-2.4%2B%20%7C%20CUDA%2012.4-ee4c2c.svg)](https://pytorch.org/)
[![Target: RTX 3050 Laptop](https://img.shields.io/badge/Hardware-RTX%203050%206GB%20Laptop-76b900.svg)](https://www.nvidia.com/)
[![Roadmap: v5 Approved](https://img.shields.io/badge/Roadmap-v5%20Approved-8b5cf6.svg)](ROADMAP.md)

*In loving memory of Marley 🐾*

</div>

---

## 📌 Executive Summary

**`marley-runtime`** is a specialized, experimental inference runtime designed to execute advanced video diffusion models (specifically **Wan2.1-T2V-1.3B**) under severe consumer hardware constraints: **~4.8 GB of effective physical VRAM** on an NVIDIA GeForce RTX 3050 6GB Laptop GPU running under Windows (WDDM driver architecture).

Rather than relying purely on blunt sequential CPU offloading, `marley-runtime` investigates and measures **adaptive, block-level memory management policies** that dynamically balance PCIe transfer latency, activation recomputation, tensor lifetime management, and selective quantization.

> [!NOTE]
> The current architectural design and phased milestone strategy is governed by **[Roadmap v5](ROADMAP.md)**, formulated through a multi-agent review consensus (ChatGPT, DeepSeek Pro, and Gemini Pro).

---

## 🎯 Target Hardware & Budget Specification

| Specification | Hardware Value / Operating Parameter |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, Compute Capability 8.6) |
| **Total Dedicated VRAM** | 6144 MB (6.0 GB) |
| **Effective Headroom** | **~4.8 GB usable physical VRAM** (accounting for Windows Desktop Window Manager / WDDM OS residency) |
| **Host CPU & Memory** | Intel Core i7-13650HX (14 cores / 20 threads) · 24 GB DDR5 RAM |
| **OS & Driver Stack** | Windows 11 64-bit · Driver 581.86 · CUDA 12.4 |

### Ground-Truth Telemetry Model
To prevent false negatives caused by driver virtualization and memory paging under WDDM, memory tracking evaluates three layers:
1. `torch.cuda.memory_allocated()`: Active PyTorch tensor memory.
2. `torch.cuda.memory_reserved()`: PyTorch caching allocator reservation.
3. **Physical GPU Residency (NVML)**: **The primary constraint metric.** Physical memory assigned by the hardware driver to the process. The strict 4.8 GB threshold applies here.

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

- **Adaptive Memory Decision Engine (`marley.core.engine`):** Operates at block/layer granularity. Uses analytical cost models (transfer vs. recompute) augmented with live runtime telemetry (free VRAM, recent PCIe transfer latency, compute times) to schedule diffusion blocks dynamically.
- **WDDM Concurrency Evaluator & Profiler (`marley.profiler`):** Samples physical GPU residency via NVML and benchmarks true asynchronous PCIe/compute overlap under the Windows WDDM driver scheduler.
- **Static Slab Allocator (`marley.ops.slab`):** Ring-buffer slab memory pool that eliminates PyTorch allocator fragmentation during iterative diffusion steps.
- **Budgeted Asynchronous Scheduler (`marley.ops.async_stream`):** Prefetches upcoming DiT layers on dedicated CUDA streams within a bounded memory envelope when WDDM concurrency is verified.
- **Temporal VAE Stitcher (`marley.ops.vae_stitch`):** Decodes latent temporal chunks with seam-blending to prevent high-resolution latent decoding OOMs.

---

## 🗺️ Roadmap & Phase Overview

Full details, Kill Gates, and verification metrics are documented in **[`ROADMAP.md`](ROADMAP.md)**:

| Phase | Milestone Name | Focus & Scope | Status |
| :---: | :--- | :--- | :---: |
| **F0** | **Reproducible Baseline** | Wan2.1-T2V-1.3B FP16 @ 480p (16 frames) with Diffusers baseline | 🟢 **PASS** |
| **F0.5** | **Progressive Exploration** | Sequential parameter isolation (FP16 envelope → INT8/INT4 → Offload) | ⚪ In Progress |
| **F0.6** | **WDDM Concurrency** | Micro-benchmark `cudaMemcpyAsync` vs. compute kernel stream overlap | ⚪ Scheduled |
| **F1** | **Lifetime Profiler** | Layer-wise allocation tracing & NVML hardware telemetry | ⚪ Scheduled |
| **F1.5** | **Bottleneck Analysis** | Classify memory: weights, activations, VAE, fragmentation, prefetch | ⚪ Scheduled |
| **F1.7** | **Selective Quantization** | Target $\ge 20\%$ physical VRAM reduction without fidelity degradation | ⚪ Conditional |
| **F2** | **Static Slab Allocator** | Zero-fragmentation ring buffer (triggered if fragmentation $> 15\%$) | ⚪ Conditional |
| **F3** | **Async Stream Scheduler** | Double-buffered PCIe prefetching (triggered if F0.6 overlap $\ge 10\%$) | ⚪ Conditional |
| **F4** | **Adaptive Decision Engine** | Core block-level cost engine (transfer vs. compute vs. offload) | ⚪ CORE |
| **F5** | **Temporal VAE Stitcher** | Chunked temporal latent decoding with blending | ⚪ Conditional |
| **F6** | **Validation Benchmarks** | Multi-dimensional benchmark (480p @ 33f primary, 720p stretch) | ⚪ Final Gate |

---

## ⚡ Quick Start

### 1. Prerequisites
- Windows 10/11 with NVIDIA GPU drivers (550+ recommended).
- Python 3.10 or 3.11.
- Git.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/LARP/marley-runtime.git
cd marley-runtime

# Create an isolated virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA 12.4 support and project dependencies
pip install -r requirements.txt
```

### 3. Running the Baseline Profiler
```bash
# Execute Phase F0 baseline profiling
python baseline_profiler.py
```

---

## 📂 Repository Structure

```
marley-runtime/
├── .agents/                 # Internal project memory & interactive visualizers (local)
├── benchmarks/              # Standardized test suites & telemetry scripts
├── logs/                    # Profiling and hardware telemetry run logs
├── marley/                  # Core package
│   ├── core/                # Adaptive engine, decision models, and layer scheduler
│   ├── models/              # Model adapters for Wan2.1 and DiT variants
│   ├── ops/                 # Slab allocator, async streams, and VAE stitcher
│   └── profiler/            # NVML telemetry, lifetime hooks, and WDDM benchmarks
├── tests/                   # Automated unit and integration tests
├── baseline_profiler.py     # Reproducible Phase F0 verification script
├── registro_fases.md        # Comprehensive phase history and compute log
├── requirements.txt         # Pinned Python package dependencies
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── ROADMAP.md               # Implementation Roadmap v5
```

---

## 🔬 Scientific & Engineering Principles

1. **Evidence Precedes Optimization:** Every optimization begins with concrete hardware profiling.
2. **Kill Gates:** Every conditional phase has measurable, non-negotiable exit criteria.
3. **Falsifiable Hypotheses:** We explicitly test whether adaptive runtime scheduling can outperform static offloading policies within a tight 4.8 GB physical memory envelope.
4. **Target Primacy:** 480p is the primary objective; 720p is a secondary stretch goal.

---

## 🤝 Provenance & Attribution

The architecture, phased progression, and metric definitions of `marley-runtime` were shaped through an iterative multi-agent engineering protocol combining **ChatGPT**, **DeepSeek Pro**, and **Gemini Pro**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
