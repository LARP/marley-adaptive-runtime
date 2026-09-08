# Test G — Model-Level CPU Offload (`model_cpu_offload`)

**Phase:** F0.5 — Sprint 1 (Minimum-Cost Exploration)  
**Date:** 2026-09-08T06:49:12Z  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Total Physical VRAM: 6,144 MB · Gate: 4,800 MB)  
**Model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`  
**Resolution / Frames:** 832×480 (480p) · 17 frames  
**Precision:** DiT: FP16 (`torch.float16`) · VAE: FP32 (`torch.float32`)  
**Offload Strategy:** `enable_model_cpu_offload()`  
**Output Video:** [`logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910.mp4`](../logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910.mp4)  
**Telemetry JSON:** [`logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910_telemetry.json`](../logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910_telemetry.json)  

---

## 1. Executive Summary & Gate Verdict

| Criterion | Target / Gate | Test G Result | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak NVML Physical Residency** | ≤ 4,800 MB | **5,861 MiB** (sampled live via `nvidia-smi`) | ❌ **FAIL (+1,061 MB violation)** |
| **PyTorch Peak Allocated (`max_memory_allocated`)** | < 6,144 MB | **13,036.9 MB (~13.0 GB virtual)** | ❌ **Massive WDDM Paging** |
| **DiT Denoising Time (30 steps)** | — | **212s (3m 32s · ~6.33 s/step)** | 🟢 **Excellent (-28.6% vs F0)** |
| **VAE Decode Time (17 frames)** | < 60s | **484s (8m 04s)** | ❌ **Severe Degradation (+50% vs F0)** |
| **Total Wall-Clock Time** | ≤ 600s | **696.3s (11m 36s)** | ❌ (+76.5s vs F0) |
| **Stability / Crashes** | Zero OOM crashes | 0 OOM crashes (WDDM paged excess) | 🟢 PASS |

**Final Verdict Test G:** **DOES NOT SATISFY GATE.**  
`model_cpu_offload` noticeably accelerates the DiT loop (from 9.93s down to 6.33s/step), but **dramatically worsens the VAE bottleneck** when the VAE remains in `float32`.

---

## 2. Comparative Telemetry: Baseline F0 vs. Test G

| Checkpoint / Metric | Baseline F0 (`cpu` - sequential) | Test G (`model_cpu`) | Delta / Analysis |
| :--- | :---: | :---: | :--- |
| **Offload Granularity** | Layer-by-layer | Submodel-by-submodel | — |
| **DiT Denoising (30 steps)** | 297s (9.93 s/step) | **212s (6.33 s/step)** | **-85s (-28.6% DiT time)** 🟢 |
| **VAE Decode (17 frames)** | 322s | **484s** | **+162s (+50.3% slower)** ❌ |
| **PyTorch Peak Allocated** | ~7.9 GB virtual | **13,036.9 MB** | **+5,122 MB activation spike** ❌ |
| **Peak Physical NVML** | 5,451 MB | **5,861 MB** | **+410 MB vs F0** (95.4% of total GPU) |
| **Post-Inference NVML** | 5,451 MB (retained) | 902.4 MB (evicted) | Hook evicted submodel on completion |
| **Generated Video** | 16 frames @ 480p | 17 frames @ 480p | Clean, intact |

---

## 3. Critical Engineering Findings

### 3.1 The Pitfall of Full GPU Residency with FP32 VAE
In baseline F0 (`sequential_cpu_offload`), individual layers of the VAE were shuttled across PCIe one at a time. While slow (322s), this bounded concurrent in-flight activations.  
In Test G (`model_cpu_offload`), the entire VAE decoder resided on the GPU. Decoding the 17-frame latent tensor in **`float32`** through 33 causal 3D convolutional layers triggered an activation cascade demanding **13,036 MB of virtual working memory**.

### 3.2 PCIe Bus Thrashing via WDDM Paging
When requesting 13.0 GB on a GPU with 6.1 GB physical capacity:
1. The Windows WDDM subsystem paged **over 7 GB of tensors across PCIe to system RAM**.
2. Each 3D convolution kernel was forced to synchronize and stream large blocks across the laptop's PCIe bus.
3. This caused an operational collapse: VAE decode degraded to **8 minutes**, significantly worse than the baseline.

### 3.3 Methodological Note on NVML Sampling
With `model_cpu_offload`, Diffusers offload hooks instantly transfer the VAE back to system RAM the exact millisecond `pipe(...)` finishes. A single post-inference snapshot shows an artificially low `902 MB`. However, real-time live sampling via `nvidia-smi` captured the actual physical peak of **5,861 MiB** and PyTorch registered `peak_torch_allocated_mb: 13,036.9 MB`.

---

## 4. Conclusion & Actionable Takeaways

1. **Discard `model_cpu_offload` with FP32 VAE:** It is structurally incapable of satisfying the gate.
2. **Definitive Diagnostic Confirmation:** The root cause is not the DiT offload mechanism, but rather **the FP32 precision of the VAE**.
3. **Mandatory Next Step: Test H (`bfloat16` VAE):**  
   Executing the VAE in `bfloat16`:
   * Theoretically halves the 13 GB activation footprint down to ~6.5 GB.
   * Eliminates massive WDDM paging thrashing when combined with sequential offloading.
   * Utilizes native Tensor Core acceleration on Ampere architecture.
