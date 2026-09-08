# Phase & Consumption Registry — Marley Runtime

This document records compute time (wall-clock), agent working hours, consumed tokens, and gate statuses across each project phase.

---

## Phase History

| Phase | Date (ISO 8601) | Compute (min) | Agent Work (h) | Consumed Tokens | Gate Status | Deviations / Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F0** | 2026-09-08T04:49:26Z | 12 min | ~1.5 h | ~18.5k | 🟢 **PASS** | Environment verified, CUDA 12.4 operational, baseline OOM reproduced and documented (1853.94 GiB requested by un-tiled 3D attention). |
| **F1** | *Upcoming* | - | - | - | ⚪ PENDING | Block-level forward hooks and PCIe DMA latency profiling. |

---

### Detailed Phase Records

#### Phase F0 — Environment, Baseline & First OOM
- **Start Time:** 2026-09-08T01:45:00-03:00
- **Completed Time:** 2026-09-08T01:49:30-03:00
- **Hardware Verified:**
  - GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver 581.86, CUDA Capability 8.6).
  - VRAM: 6143.50 MB Total | 5154.00 MB Initial Free Net under WDDM.
  - CPU: Intel Core i7-13650HX (14 cores / 20 threads).
  - RAM: 24 GB DDR5.
- **Environment:** Isolated `.venv` with Python 3.10.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124.
- **Gate F0 Result:** **PASS**
  - **First CUDA OOM Trigger:** Resolution=720p (1280x720), Frames=81 (20 temporal latent steps, 288k 3D sequence tokens), Dtype=fp16.
  - **Memory Incident:** Block 01 un-tiled 3D attention ($Q \times K^T$) attempted allocating 1853.94 GiB with 1.71 GiB remaining.
  - **Incident Log:** Permanent record saved in [`logs/f0_baseline.log`](file:///d:/Gemini_Admin_Tool/marley_720p/logs/f0_baseline.log).
