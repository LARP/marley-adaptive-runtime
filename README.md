# Marley Runtime

**Adaptive runtime for video diffusion models on ultra-low VRAM GPUs.**

> Experimental MVP · In memory of Marley 🐾

---

### Overview

Marley Runtime is an experimental adaptive runtime designed to run video diffusion models (such as Wan2.1) on GPUs with very limited VRAM (e.g. RTX 3050 6GB).

It focuses on reducing peak memory usage through a combination of:

- Static Slab Allocator (ring-buffer style preallocation)
- Asynchronous transfers with CUDA Streams + Prefetch
- Cost model (Transfer vs Recompute)
- Temporal VAE Stitching

The goal is **not** to claim universal compatibility or maximum performance, but to measure and validate concrete techniques that allow video generation under severe VRAM constraints.

---

### Current Status

🚧 **Experimental MVP** — Early development stage.

This project follows a strict phased approach. Currently focused on establishing a solid baseline and memory profiling before implementing the core runtime optimizations.

---

### Target Hardware

- NVIDIA GeForce RTX 3050 6GB (Laptop) and similar low-VRAM GPUs
- ~4.8 GB usable VRAM under Windows (WDDM)

---

### Planned Features

| Component                    | Description                                      | Status     |
|-----------------------------|--------------------------------------------------|------------|
| Static Slab Allocator       | Preallocated ring buffer to reduce fragmentation | Planned    |
| Async Prefetch + Streams    | Overlap compute and PCIe transfers               | Planned    |
| Cost Model                  | Dynamic decision: transfer vs recompute          | Planned    |
| Temporal VAE Stitcher       | Chunked decoding with temporal blending          | Planned    |
| Memory Profiler             | Detailed VRAM & transfer analysis                | In progress|

---

### Installation

```bash
# Coming soon
