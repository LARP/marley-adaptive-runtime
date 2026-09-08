# Marley Runtime — Central Benchmark & Test Registry

Dedicated directory for test documentation, telemetry capture, and performance analysis across runtime phases (Phase F0.5 onward).

---

## 1. Master Experiment Matrix (Phase F0.5 — VAE & Offload Optimization)

**Phase F0.5 Gate Target:** Reduce physical peak VRAM (NVML) to **≤ 4,800 MB** on RTX 3050 6GB Laptop GPU.  
**Baseline F0 Reference:** Peak NVML = **5,451 MB** (+651 MB violation) · VAE Decode = **322s** (Sequential Offload, FP32 VAE).

| Test ID | Description / Strategy | Configuration | Peak NVML (50ms) | Denoise Time | VAE Decode | Total Wall-Clock | Gate F0.5 | Report |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0 (Ref)** | Sequential CPU Offload + FP32 VAE | `res=480p, 16f, DiT=fp16, VAE=fp32, offload=cpu` | 5,451 MB | 297s | 322s | 619.8s (10.3m) | ❌ +651 MB | [F0 Baseline](../logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json) |
| **Test G** | Model CPU Offload (submodel level) | `res=480p, 17f, DiT=fp16, VAE=fp32, offload=model_cpu` | **5,861 MB** (live peak) | **212s** (6.3s/it) | **484s** | **696.3s (11.6m)** | ❌ +1,061 MB (WDDM 13GB paging) | [TEST_G_model_cpu_offload.md](./TEST_G_model_cpu_offload.md) |
| **Test H** | Sequential Offload + bfloat16 VAE | `res=480p, 17f, DiT=fp16, VAE=bf16, offload=cpu` | **6,028 MB** (peak) / 4,839 post | **274s** (9.1s/it) | **65s** 🟢 | **339.0s (5.65m)** 🟢 | ⚠️ Alloc 4,366 MB (VAE -80% time) | [TEST_H_bfloat16_vae.md](./TEST_H_bfloat16_vae.md) |
| **Test I** | Sequential Offload + float16 VAE | `res=480p, 17f, DiT=fp16, VAE=fp16, offload=cpu` | *Skipped* (same memory footprint as BF16) | — | — | — | *Superseded by Test J* | Skipped |
| **Test J** | **Sequential Offload + BF16 + VAE Tiling** | `res=480p, 17f, DiT=fp16, VAE=bf16, vae_tiling=True` | **2,902.0 MB** 🟢 | **270s** (9.0s/it) | **58s** 🟢 | **328.2s (5.47m)** 🟢 | 🟢 **PASS (-1,898 MB headroom)** | [TEST_J_vae_tiling_bf16.md](./TEST_J_vae_tiling_bf16.md) |
| **Test K** | Custom Chunked VAE | *Not required* | — | — | — | — | *Gate F0.5 passed by Test J* | Standby |
| **Test L** | Chunked VAE + GPU Residency | *Future exploration* | — | — | — | — | *Deferred to Phase F4* | Standby |

---

## 2. Test Documentation Architecture

Each test run contains an individual technical report in this directory:
- `TEST_<ID>_<strategy>.md`: Document covering the technical hypothesis, reproducible command line, multi-layer memory telemetry, bottleneck analysis, and gate verdict.
- Generated `.mp4` video outputs are stored locally in `logs/` and excluded from the Git repository due to binary size constraints.
