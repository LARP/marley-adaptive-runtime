# F8 Screening Report — A/B Evaluation of Selective DiT Quantization

**Date:** 2026-09-10T03:27:52.612237  
**Protocol Reference:** [`docs/F8_PREREGISTERED_SCREENING_PROTOCOL_01.md`](F8_PREREGISTERED_SCREENING_PROTOCOL_01.md)  
**Official Verdict:** **🔴 RESULTADO NULO (CIERRE DEFINITIVO)**  

---

## 1. Executive Screening Scorecard

> **Governance Resolution:** CIERRE DEFINITIVO: Sin efecto apreciable (< 50 MB). La línea 720p se cierra definitivamente.

| Condition | S0 Baseline | S1 Operac. | Peak NVML (Steps 1-10) | Avg Cadence | Latent Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Condition A: Control FP16** | 916.4 MB | 1017.0 MB | **4483.0 MB** | 61.59 s/step | Baseline (1.000000) |
| **Condition B: Treatment INT8** | 910.8 MB | 1068.1 MB | **4488.0 MB** | 63.23 s/step | **1.00003** |
| **Delta (Reduction)** | — | — | **-5.0 MB** | +1.64 s/step | Tolerance $\ge 0.990$ |

## 2. Step-by-Step Telemetry Progression (Steps 1–10)

| Step | Control A NVML (MB) | Control A Reserved | Treatment B NVML (MB) | Treatment B Reserved | Delta NVML |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4358.9 | 3308.0 | 4329.9 | 3328.0 | +29.0 MB |
| 2 | 4308.0 | 3292.0 | 4322.9 | 3312.0 | -14.9 MB |
| 3 | 4445.5 | 3388.0 | 4449.5 | 3408.0 | -3.9 MB |
| 4 | 4446.1 | 3392.0 | 4458.8 | 3412.0 | -12.7 MB |
| 5 | 4467.5 | 3388.0 | 4448.1 | 3408.0 | +19.5 MB |
| 6 | 4425.3 | 3388.0 | 4440.2 | 3408.0 | -14.9 MB |
| 7 | 4477.5 | 3388.0 | 4405.6 | 3408.0 | +71.9 MB |
| 8 | 4483.1 | 3384.0 | 4432.6 | 3404.0 | +50.5 MB |
| 9 | 4436.4 | 3384.0 | 4441.2 | 3404.0 | -4.9 MB |
| 10 | 4430.2 | 3384.0 | 4439.1 | 3404.0 | -8.9 MB |

## 3. Epistemological Implications

The objective measurement demonstrates a physical VRAM peak delta of **-5.0 MB**. The reduction is insufficient to justify 720p feasibility under the 4.8 GB hard gate. The quantization lever does not overcome the physical hardware boundary on RTX 3050 Laptop under Windows 11 WDDM.
