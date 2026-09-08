# Test Report — Phase F5-A: Canonical 33-Frame VAE Decode Probe
## Formal Verdict: RETIRE F5 — Resolved by Existing Tiled VAE

**Phase:** F5 — Temporal VAE Stitcher (Probe F5-A)  
**Date:** 2026-09-08  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,143.5 MB total · 4,800 MB hard gate)  
**Host:** Windows 11 (WDDM concurrency active)  
**Model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`AutoencoderKLWan`, BF16)  
**Workload (Canonical F5):** 832×480 (480p) · **33 frames exact** · Latent `(1, 16, 9, 60, 104)`  
**Configuration:** Native `enable_tiling()` with 256×256 spatial tiles & causal temporal caching  
**Telemetry Artifact:** [`logs/f5_vae_probe_33f.json`](../logs/f5_vae_probe_33f.json)  
**Directive Authority:** [`docs/F5_DIRECTOR_RESOLUTION_01.md`](../docs/F5_DIRECTOR_RESOLUTION_01.md) · [`docs/F5_CONSULTANT_RATIFICATION_01.md`](../docs/F5_CONSULTANT_RATIFICATION_01.md)  
**Status:** 🟢 **CERTIFIED PASS — RETIRE F5-C**

---

## 1. Executive Summary

In strict accordance with the *Decision-First* protocol agreed between the Project Director and the External Technical Consultant, Phase F5-A was executed to establish whether scaling the temporal axis to **33 frames** produces an activation VRAM cliff or latency degradation that requires building a custom Temporal VAE Stitcher (`marley/ops/vae_stitch.py`).

The empirical evidence is definitive:
- **Peak Physical VRAM (NVML 50ms):** **2,109.0 MB** (Hard Gate ≤ 4,800 MB: **PASS**, Target ≤ 4,000 MB: **MET** with **+2,691 MB free headroom**).
- **VAE Decode Latency:** **26.99 seconds** (Engineering target ≤ 150 s: **MET with 82% margin**).
- **Integrity:** 0 NaNs, 0 Infs, output tensor `[1, 3, 33, 480, 832]` matching exact dimensions.
- **Host Process RAM:** 1,402.6 MB RSS (negligible system footprint, zero thrashing).

Because the existing tiled VAE path (`torch.bfloat16` + `enable_tiling()` 256×256) decodes 33 frames effortlessly within ~2.1 GB of VRAM in under 27 seconds, **no Temporal Stitcher is needed**.

---

## 2. Scorecard & Kill Gates

| Gate / Target | Threshold | Measured (F5-A) | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak VRAM (Hard Gate)** | ≤ 4,800 MB | **2,109.0 MB** | 🟢 **PASS (+2,691 MB margin)** |
| **Engineering Target VRAM** | ≤ 4,000 MB | **2,109.0 MB** | 🟢 **MET** |
| **VAE Decode Latency Target** | ≤ 150.0 s | **26.99 s** | 🟢 **MET (5.5× faster than budget)** |
| **Tensor Integrity** | 0 NaNs / 0 Infs | **0 / 0** | 🟢 **PASS** |
| **Output Shape Verification** | `[1, 3, 33, 480, 832]` | `[1, 3, 33, 480, 832]` | 🟢 **MATCH** |
| **Host Process RAM (RSS)** | Audited | **1,402.6 MB** | 🟢 **HEALTHY** |

---

## 3. Physical Analysis: Why 33 Frames Did Not Cause a Cliff

The VAE decoder in `AutoencoderKLWan` utilizes causal 3D convolutions with internal temporal slicing and spatial tiling:
1. **Spatial Tiling (256×256):** Limits the 2D receptive field to small $32 \times 32$ latent patches.
2. **Causal Temporal Slicing:** Rather than materializing all 33 full-frame activation volumes simultaneously in GPU memory, the decoder streams through the 9 temporal latent slices iteratively, passing history via compact hidden state caches (`feat_cache`).
3. **Peak Memory Confinement:** Peak memory is dominated by the convolutional workspace of a single spatial tile window ($~1.05\text{ GB}$ PyTorch alloc), remaining almost invariant to sequence length ($17\text{f} \rightarrow 33\text{f}$).

---

## 4. Decision & Architectural Action

Following Section 3 of `docs/F5_DIRECTOR_RESOLUTION_01.md`:

```text
[ F5-A: CANONICAL 33f PROBE ] ──► VRAM = 2,109 MB (<= 4,800) & Decode = 27 s (<= 150)
                                      │
                                      ▼
                        🟢 RETIRE F5 — RESOLVED BY EXISTING TILED VAE
                                      │
                                      ▼
                             DIRECT ADVANCE TO F6
```

- **F5-B (Temporal Characterization):** Not required (latency is already 27 s).
- **F5-C (`marley/ops/vae_stitch.py`):** **CANCELLED / RETIRED**. Writing a temporal chunker would introduce unnecessary complexity, seam blending issues, and potential temporal flicker without any VRAM benefit.
- **Phase F5 Closure:** Formally closed as **RETIRED (Resolved by Tiling)**.

---

*Marley Runtime — Phase F5 Technical Report · en memoria de Marley 🐾.*
