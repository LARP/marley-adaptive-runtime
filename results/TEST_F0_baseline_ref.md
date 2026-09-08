# Test F0 — Reproducible Baseline Reference (`f0_baseline_real.py`)

**Phase:** F0 — Baseline Real Model Reference  
**Date:** 2026-09-08T03:04:37Z  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Total Physical VRAM: 6,144 MB · Hard Gate: 4,800 MB)  
**Model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`  
**Resolution / Frames:** 832×480 (480p) · 16 frames  
**Precision:** DiT: FP16 (`torch.float16`) · VAE: FP32 (`torch.float32`, default Diffusers un-tiled)  
**VAE Tiling / Chunking:** Disabled (standard un-tiled causal temporal loop)  
**Offload Strategy:** `enable_sequential_cpu_offload()` (`--offload cpu`)  
**Output Video:** [`logs/f0_480p_16f_fp16_cpu_20260908_030437.mp4`](../logs/f0_480p_16f_fp16_cpu_20260908_030437.mp4)  
**Telemetry JSON:** [`logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json`](../logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json)  

---

## 1. Executive Summary & Gate Verdict

| Criterion | Target / Gate | Baseline F0 Result | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak NVML Physical Residency** | ≤ 4,800 MB | **5,450.7 MB (~5,451 MB)** | ❌ **FAIL (+650.7 MB gate violation)** |
| **PyTorch Peak Allocated (`max_memory_allocated`)** | < 6,144 MB | **8,123.0 MB (~8.1 GB virtual)** | ❌ **Spillover / Paged to WDDM Shared Memory** |
| **DiT Denoising Time (30 steps)** | — | **297s (4m 57s · ~9.90 s/step)** | 🟢 Stable execution |
| **VAE Decode Time (16 frames)** | < 60s | **322s (5m 22s)** | ❌ **Severe Computational Bottleneck (52% of total time)** |
| **Total Wall-Clock Time** | ≤ 600s | **619.78s (10m 20s)** | ❌ **FAIL (+19.8s above 10-minute threshold)** |
| **Stability / OOM Crashes** | Zero crashes | 0 OOM crashes (completed with WDDM paging) | 🟢 PASS |

### **FINAL VERDICT: HARD GATE 4,800 MB VIOLATED (FAIL)**
While the sequential CPU offload architecture of `f0_baseline_real.py` successfully completed inference without raising an unhandled Python exception, **it failed both key operational criteria**:
1. **Physical Residency:** Peak NVML physical usage reached **5,450.7 MB** (88.7% of total GPU capacity), exceeding the 4,800 MB safety envelope by **650.7 MB**.
2. **Computational Latency:** The unoptimized `float32` VAE decoding loop took **322 seconds**, consuming more than half of the total generation pipeline and pushing wall-clock time over 10 minutes.

---

## 2. Comparative Matrix: F0 vs. Optimization Path (F0.5)

| Metric | Baseline F0 (Ref) | Test G (Model-Offload) | Test H (Plain BF16) | Test J (Winning Config) | Delta F0 → Test J |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Offload Scheme** | Sequential (`cpu`) | Model (`model_cpu`) | Sequential (`cpu`) | **Sequential (`cpu`)** | Same offload mechanism |
| **VAE Precision** | FP32 | FP32 | BF16 | **BF16** | Precision halved |
| **VAE Tiling** | None | None | None | **Enabled (`--vae-tiling`)** | Spatial 256×256 tiles |
| **Peak Physical NVML** | **5,450.7 MB** | 5,861 MB | 6,028.9 MB | **2,902.0 MB** | 🟢 **-2,548.7 MB (-46.8%)** |
| **Gate Margin (4,800 MB)** | **-650.7 MB** | -1,061 MB | -1,228 MB | **+1,898.0 MB** | 🟢 **+2,548.7 MB recovered** |
| **PyTorch Peak Allocated** | **8,123.0 MB** | 13,036.9 MB | 4,366.2 MB | **2,021.1 MB** | 🟢 **-6,101.9 MB (-75.1%)** |
| **DiT Denoise Time** | 297s | 212s | 274s | **270s** | -27s (-9.1%) |
| **VAE Decode Time** | **322s** | 484s | 65s | **58s** | 🟢 **-264s (-82.0%)** |
| **Total Wall-Clock** | **619.8s (10.3m)** | 696.3s (11.6m) | 339.0s (5.65m) | **328.2s (5.47m)** | 🟢 **-291.6s (-47.0%)** |
| **Gate F0.5 Verdict** | ❌ **FAIL** | ❌ **FAIL** | ⚠️ **ALLOC ALERT** | 🟢 **PASS** | 🟢 **Full Production Gate** |

---

## 3. Engineering Diagnosis & Root Cause Analysis

### 3.1 The Double Bottleneck of Baseline F0
The baseline F0 configuration was characterized by two primary architectural bottlenecks:
1. **Layer-by-Layer PCIe Serialization in DiT:**  
   `enable_sequential_cpu_offload()` hooks each individual sub-module and shuttles weights from host RAM to GPU VRAM and back for every single forward evaluation. Across 30 denoising steps with Classifier-Free Guidance (CFG=2, totaling 60 full passes over the 30 DiT blocks), this serialization imposed ~297 seconds of runtime.
2. **Untiled `float32` VAE Memory Expansion:**  
   Standard Diffusers instantiation loads `AutoencoderKLWan` in `float32`. When evaluating the 16-frame latent tensor through 33 causal 3D convolution layers simultaneously:
   - In-flight activation tensors expanded to **8,123 MB**.
   - Because physical VRAM is limited to 6,144 MB (and system reserved headroom leaves ~5,400 MB usable), Windows WDDM forced ~2.5 to 3.0 GB of activation buffers into system RAM across PCIe.
   - Paging thrashing caused the VAE forward pass to stall for **322 seconds**.

### 3.2 Significance to the Marley Roadmap
The empirical failure of F0 established the technical baseline and justified the subsequent sprints:
- **Sprint 1 (Test G):** Attempted `model_cpu_offload` to reduce PCIe overhead during DiT. It exacerbated the VAE memory crisis (spiking to 13,036 MB allocated).
- **Sprint 2 (Test H):** Proved that converting the VAE to `bfloat16` cuts decoding latency by 80% (322s → 65s), but left peak allocation near the threshold.
- **Sprint 3 (Test J):** Combined `bfloat16` with causal spatial-temporal tiling (`--vae-tiling`), completely resolving the memory crisis (**2,902 MB peak NVML**) and achieving final closure of Phase F0.5.

---

## 4. Telemetry Log Excerpt

From [`logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json`](../logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json):

```json
{
  "phase": "F0",
  "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "res": "480p",
  "resolution_px": "832x480",
  "frames": 16,
  "dtype": "fp16",
  "offload": "cpu",
  "inference_time_s": 619.78,
  "peak_torch_allocated_mb": 8123.0,
  "peak_nvml_used_mb": 5450.7,
  "vram_gate_4800mb_ok": false,
  "output_video": "D:\\Gemini_Admin_Tool\\marley_720p\\logs\\f0_480p_16f_fp16_cpu_20260908_030437.mp4",
  "status": "SUCCESS"
}
```

---

## 5. Exact Reproducibility Command

To reproduce this unoptimized reference baseline run from repository root:

```powershell
.venv\Scripts\python.exe f0_baseline_real.py `
    --res 480p `
    --frames 16 `
    --dtype fp16 `
    --vae-dtype fp32 `
    --offload cpu
```
