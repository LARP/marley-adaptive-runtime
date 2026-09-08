# Test F1.5 — Real Bottleneck Identification & Roadmap Dispatch

**Phase:** F1.5 — Bottleneck Classification  
**Date:** 2026-09-08T14:45:00Z  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Total Physical VRAM: 6,144 MB · Hard Gate: 4,800 MB)  
**Input Telemetry:** [`logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json) (Canonical Phase F1 17f/30st Run)  
**Output Telemetry:** [`logs/f1_5_bottleneck_decomposition.json`](../logs/f1_5_bottleneck_decomposition.json)  
**Status:** 🟢 **COMPLETE**

---

## 1. Executive Summary

Phase F1.5 ingests the canonical dynamic profiling telemetry from Phase F1 (32 live traced components, 30 steps, 17 frames @ 480p) to perform the **quantitative decomposition of VRAM consumption across the 5 architectural categories** mandated by ROADMAP §3.

Based on empirical evidence, this report issues the **binding dispatch decisions** for downstream phases (F1.7, F2, F3, and F5).

### Key Empirical Findings:
1. **Allocator Fragmentation is Negligible (44.9 MB / < 1.0% of Gate):** PyTorch's native caching allocator operates with extreme efficiency during sequential offload. The Phase F2 trigger (> 15% fragmentation) is **refuted**. Phase F2 is formally **BYPASSED**.
2. **Double-Buffered Asynchronous Prefetching is 100% Safe:** Staging a DiT block in advance requires exactly **88.6 MB**, whereas current physical headroom under the 4,800 MB gate is **+1,576.3 MB**. Prefetch consumes only **5.6%** of available headroom. Combined with Phase F0.6's measured **79.9–96.3% PCIe/compute overlap**, Phase F3 is formally **GREENLIT**.
3. **VAE is Activation-Dominated, but Contained:** VAE peak residency is 2,112.1 MB on just 242.0 MB of weights. Tiled BF16 decoding successfully prevents the historical +4.1 GB crash.
4. **Text Encoder Dominates Static Weight Volume:** The T5-XXL text encoder comprises 14,758.5 MB (83.6% of all model weights), creating an operational incentive for Phase F1.7 selective quantization.

---

## 2. Quantitative Memory Decomposition (5 Categories)

| # | Category | Measured Footprint | Nature & Architectural Behavior |
| :---: | :--- | :---: | :--- |
| **1** | **Static Model Weights** | **17,658.6 MB Total**<br>(88.6 MB active DiT / 242 MB VAE) | Text Encoder (14,758.5 MB) + 30 DiT Blocks (2,658.0 MB) + VAE (242.0 MB). Under sequential offload, only ~88.6 MB resides on GPU concurrently during denoising. |
| **2** | **DiT Self/Cross-Attention Activations** | **~550 – 600 MB**<br>(Peak DiT NVML: 1,743.5 MB) | Uniform across all 30 blocks (1,645.7 – 1,743.5 MB). Per-block activation delta is minimal (mean 22.85 MB). Baseline OS/driver absorbs ~1,100 MB. |
| **3** | **VAE Latent Decoding Activations** | **~770 MB pure activations**<br>(Peak VAE NVML: 2,112.1 MB) | 3D causal convolutional activations. Heavily activation-dominated (residency is 8.7× weights). Completely contained within budget via 256×256 micro-tiling. |
| **4** | **Memory Allocator Fragmentation** | **44.9 MB**<br>(0.94% of 4.8 GB Gate) | Difference between PyTorch reserved (2,066.0 MB) and allocated (2,021.1 MB). Well below the 15% (720 MB) threshold. |
| **5** | **Host-to-Device Prefetch Buffers** | **88.6 MB buffer needed**<br>(1,576.3 MB headroom available) | Staging 1 subsequent DiT block requires 88.6 MB. Leaves +1,487.7 MB of untouched safety buffer below the 4,800 MB gate. |

---

## 3. Conditional Trigger Evaluations & Kill Gate Decisions

### 3.1 Phase F2 — Static Slab Allocator
- **Roadmap Criterion:** Triggered *strictly if* memory allocator fragmentation or caching overhead accounts for **> 15%** (720 MB) of the 4.8 GB budget.
- **Measured Fragmentation:** **44.9 MB** (**0.94%** of budget / 2.17% of allocated memory).
- **Verdict:** 🟢 **TRIGGER REJECTED — PHASE F2 BYPASSED**  
  *Rationale:* Implementing a custom fixed-ring slab allocator would add significant architectural overhead with near-zero practical memory savings (< 45 MB). PyTorch's native caching allocator is fully sufficient.

### 3.2 Phase F3 — Budgeted Asynchronous Scheduler
- **Roadmap Criterion:** Activated if Phase F0.6 demonstrates **≥ 10% overlap** under WDDM AND Phase F1.5 confirms double-buffered prefetching does not inflate peak residency past the 4.8 GB boundary.
- **Measured Metrics:**
  - WDDM PCIe / compute overlap: **79.9% – 96.3%** (measured in [`TEST_F0.6_wddm_overlap.md`](./TEST_F0.6_wddm_overlap.md)).
  - Prefetch buffer requirement: **88.6 MB** (1 DiT block).
  - Available gate headroom: **+1,576.3 MB**.
  - Safety margin remaining after prefetch: **+1,487.7 MB**.
- **Verdict:** 🟢 **TRIGGER APPROVED — PHASE F3 GREENLIT**  
  *Rationale:* Both conditions are overwhelmingly satisfied. Overlapping the 88.6 MB PCIe transfer of block $i+1$ with the ~10.4s compute of block $i$ will directly reduce the 455s inference latency without risking VRAM exhaustion.

### 3.3 Phase F1.7 — Selective & Adaptive Quantization
- **Roadmap Criterion:** Quantify opportunities to lower peak physical residency by ≥ 20% or dramatically compress host RAM weight footprints.
- **Target Analysis:**
  - Text encoder weights alone account for **14,758.5 MB** of host RAM and drove the global peak NVML residency (**3,223.7 MB**) during prompt ingestion.
  - DiT weights total **2,658.0 MB** across 30 blocks.
- **Verdict:** 🟢 **APPROVED IN PARALLEL**  
  *Focus:* INT8/FP8/NF4 quantization of the T5-XXL text encoder to compress host RAM demand, combined with quantized linear projections in DiT.

### 3.4 Phase F5 — Temporal VAE Stitcher
- **Roadmap Criterion:** Activated only if VAE decoding is an uncontained OOM bottleneck.
- **Verdict:** ⚪ **STANDBY / DEFERRED**  
  *Rationale:* The VAE memory bottleneck was already completely solved in Phase F0.5 (Test J) by causal spatial-temporal tiling (`pipe.vae.enable_tiling()`), which reduced peak residency to 2,112 MB. No custom stitching layer is required at 480p.

---

## 4. Master Dispatch Order

| Priority | Target Phase | Implementation Scope | Expected Impact |
| :---: | :--- | :--- | :--- |
| **1** | **Phase F3 (Budgeted Async Scheduler)** | Double-buffered CUDA stream pipeline overlapping H2D transfer of block $i+1$ with compute of block $i$. | Eliminates PCIe weight transfer latency from the DiT loop; cuts inference time from 455s. |
| **2** | **Phase F1.7 (Selective Quantization)** | 8-bit / 4-bit quantization of T5-XXL text encoder and non-critical DiT projection layers. | Reduces system RAM footprint by >10 GB and shrinks per-block PCIe transfer payload from 88 MB to ~44 MB. |
| **3** | **Phase F4 (Adaptive Decision Engine)** | Unifies F1 telemetry profiles, F3 async streams, and dynamic VRAM observation into an AOT execution policy. | Core orchestrator for multi-resolution adaptive execution. |
| **—** | **Phase F2 (Static Slab Allocator)** | *Bypassed / Discarded.* | Zero need identified (fragmentation < 1%). |

---

## 5. Artifacts Generated
- **Analysis Tool:** [`f1_5_bottleneck_classifier.py`](../f1_5_bottleneck_classifier.py)
- **Decomposition Telemetry:** [`logs/f1_5_bottleneck_decomposition.json`](../logs/f1_5_bottleneck_decomposition.json)
- **Formal Report:** [`results/TEST_F1.5_bottleneck_classification.md`](./TEST_F1.5_bottleneck_classification.md)
