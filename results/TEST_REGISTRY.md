# Marley Runtime — Central Test Registry

Dedicated directory for test documentation, telemetry capture, and performance analysis across runtime phases (Phase F0.5 onward).

This file is the **index** of all test reports in `results/`. Each row below lists a report file in
**chronological order**, with a one-line summary of what it covers and its gate status.

---

## 1. Test Reports (in order)

| # | Report file | Phase | Subject | Key result | Status |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 0 | [`TEST_F0_baseline_ref.md`](./TEST_F0_baseline_ref.md) | F0 | Reproducible Baseline Reference (`f0_baseline_real.py`) | Peak NVML 5,451 MB; VAE decode 322s; total 619.8s | ❌ Gate +651 MB |
| 1 | [`TEST_G_model_cpu_offload.md`](./TEST_G_model_cpu_offload.md) | F0.5 | Model-level CPU offload (`--offload model_cpu`) | Peak NVML 5,861 MB; VAE paging 13 GB virtual | ❌ Gate +1,061 MB |
| 2 | [`TEST_H_bfloat16_vae.md`](./TEST_H_bfloat16_vae.md) | F0.5 | Sequential offload + `bfloat16` VAE | VAE decode 322s → 65s | ⚠️ Alloc 4,366 MB |
| 3 | [`TEST_J_vae_tiling_bf16.md`](./TEST_J_vae_tiling_bf16.md) | F0.5 | Sequential offload + BF16 VAE + spatial-temporal tiling | Peak NVML 2,902 MB; VAE decode 58s | 🟢 **PASS** (−1,898 MB) |
| 4 | [`TEST_F0.6_wddm_overlap.md`](./TEST_F0.6_wddm_overlap.md) | F0.6 | WDDM `cudaMemcpyAsync` H2D vs. Tensor-Core `matmul` overlap | Overlap 79.9–96.3% | 🟢 **PASS** (F3 active) |
| 5 | [`TEST_F1_lifetime_profiler.md`](./TEST_F1_lifetime_profiler.md) | F1 | Tensor Lifetime Profiler (per-DiT-block residency, real model) | Peak NVML 3,223.7 MB; 32 components traced | 🟢 **PASS** (−1,576 MB) |
| 6 | [`TEST_F1.5_bottleneck_classification.md`](./TEST_F1.5_bottleneck_classification.md) | F1.5 | Real Bottleneck Identification & Roadmap Dispatch | Allocator frag 44.9 MB (<1%); Prefetch safe (88.6 MB vs 1.57 GB); F3 Greenlit, F2 Bypassed | 🟢 **COMPLETE** |
| 7 | [`TEST_F1.7_selective_quantization.md`](./TEST_F1.7_selective_quantization.md) | F1.7 | Selective & Adaptive Quantization | T5 8-bit (-50% / -7.38 GB); DiT 8-bit (-48.2%); Cosine Sim 0.9996; Gate $\ge 20\%$ PASS | 🟢 **PASS** |
| 8 | [`TEST_F3_async_scheduler.md`](./TEST_F3_async_scheduler.md) | F3 | Budgeted Asynchronous Scheduler (Double-buffer) | First run: +10.4% (optimistic sample). **Certified by [`TEST_F3_verification.md`](./TEST_F3_verification.md): mean +7.2%** (6.0–8.4%); overlap 100% measured; VRAM 2,584 MB; 0 NaNs | 🟢 **PASS** (§7) |
| 9 | [`TEST_F3_verification.md`](./TEST_F3_verification.md) | F3 | Final Measurement Verification (Enmiendas 1–3) | External wall-clock speedup mean **+7.2%** (stable over 5 A/B reps); stall 0 ms; peak 2,584 MB | 🟢 **CERTIFIED PASS** → F3+INT8 |
| 10 | [`TEST_F3_INT8_benchmark.md`](./TEST_F3_INT8_benchmark.md) | F3+INT8 | Isolation: transfer-volume reduction (INT8 linear proj, FP16 compute, scheduler frozen) | Async-vs-Sync **+13.2%**; payload −49.9%; overlap 98.1%; peak **2,076 MB** (−508 vs FP16); cos ≥ 0.9999; 0 NaNs | 🟢 **PASS** |
| — | [`F3_INT8_CLOSURE_01.md`](../docs/F3_INT8_CLOSURE_01.md) | F3+INT8 | Formal closure acta (freeze as F4 baseline) | +13.2% Async-vs-Sync; 46.5 MB/block; 2,076 MB; cos ≥ 0.9999; precise intra-session interpretation | 🟢 **CLOSED / FROZEN** |
| 11 | [`TEST_F4_adaptive_benchmark.md`](./TEST_F4_adaptive_benchmark.md) | F4 | Adaptive Memory Decision Engine (decision loop, A/B/C/D same-session + pressure test) | CORE VALIDATED: Safety 2,882 MB/0 NaN; Adaptive Gate PASS (pressure→Memory Safe→recovery, no oscillation); overhead 0.1 ms; D vs best static B = −3.40% (no-pressure, D stayed INT8) | 🟢 **CORE VALIDATED / PERF. PENDING** |
| 12 | [`TEST_F5_vae_probe_33f.md`](./TEST_F5_vae_probe_33f.md) | F5 | Canonical 33-frame VAE Decode Probe (Decision-First Protocol) | Peak NVML **2,109 MB** (Gate ≤ 4,800); Decode **26.99 s** (Target ≤ 150 s); 0 NaNs; 1,402 MB RAM; No temporal stitcher needed | 🟢 **RETIRE F5 (Resolved by Tiling)** |
| 13 | [`docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md`](../docs/F6_REPRODUCIBILITY_VARIABILITY_REPORT_01.md) | F6 | Multirun Core 4×3 Reproducibility & Variability Benchmark ($n=3, df=2$) | 12/12 corridas PASS $\le 4,800$ MB (Peak 2,741–4,047 MB), cadencia ~14.25 s/p, 0 NaNs | 🟢 **CERTIFIED & FROZEN** |
| 14 | [`docs/F7_STAGE_B_FULL_VALIDATION_REPORT_01.md`](../docs/F7_STAGE_B_FULL_VALIDATION_REPORT_01.md) | F7 | 720p Full 30-Step E2E Validation & Inter-Process Causal Intervention | Peak NVML **4,996.0 MB** en Paso 6 (+196.0 MB violación). 480p ratificado como único estándar | 🔴 **CLOSED FAIL** |
| 15 | [`docs/F8_SCREENING_AB_REPORT_01.md`](../docs/F8_SCREENING_AB_REPORT_01.md) | F8 | 10-Step Clean Processes A/B Screening (Selective DiT Quantization) | Delta Peak **-5.0 MB** ($\approx 0\text{ MB} < 50\text{ MB}$). Hipótesis falsada. Línea 720p definitivamente cerrada | 🔴 **NULL RESULT (720p CLOSED)** |

> **Notes:**
> - **F0 (Ref)** is documented in [`TEST_F0_baseline_ref.md`](./TEST_F0_baseline_ref.md) with ground-truth telemetry from [`logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json`](../logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json).
> - `Test I` and `Test K` produced **no report file** (skipped / not required) — see §2 matrix.
> - The order above is chronological by test **date**, which is the project's natural reading order
>   (`F0 → F0.5 → F0.6 → F1 → F1.5 → F1.7`). Detailed scorecards for each phase follow in §2.

---

## 2. Master Experiment Matrix (Phase F0.5 — VAE & Offload Optimization)

**Phase F0.5 Gate Target:** Reduce physical peak VRAM (NVML) to **≤ 4,800 MB** on RTX 3050 6GB Laptop GPU.  
**Baseline F0 Reference:** Peak NVML = **5,451 MB** (+651 MB violation) · VAE Decode = **322s** (Sequential Offload, FP32 VAE).

| Test ID | Description / Strategy | Configuration | Peak NVML (50ms) | Denoise Time | VAE Decode | Total Wall-Clock | Gate F0.5 | Report |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0 (Ref)** | Sequential CPU Offload + FP32 VAE | `res=480p, 16f, DiT=fp16, VAE=fp32, offload=cpu` | 5,451 MB | 297s | 322s | 619.8s (10.3m) | ❌ +651 MB | [TEST_F0](./TEST_F0_baseline_ref.md) |
| **Test G** | Model CPU Offload (submodel level) | `res=480p, 17f, DiT=fp16, VAE=fp32, offload=model_cpu` | **5,861 MB** (live peak) | **212s** (6.3s/it) | **484s** | **696.3s (11.6m)** | ❌ +1,061 MB (WDDM 13GB paging) | [TEST_G](./TEST_G_model_cpu_offload.md) |
| **Test H** | Sequential Offload + bfloat16 VAE | `res=480p, 17f, DiT=fp16, VAE=bf16, offload=cpu` | **6,028 MB** (peak) / 4,839 post | **274s** (9.1s/it) | **65s** 🟢 | **339.0s (5.65m)** 🟢 | ⚠️ Alloc 4,366 MB (VAE -80% time) | [TEST_H](./TEST_H_bfloat16_vae.md) |
| **Test I** | Sequential Offload + float16 VAE | `res=480p, 17f, DiT=fp16, VAE=fp16, offload=cpu` | *Skipped* (same memory footprint as BF16) | — | — | — | *Superseded by Test J* | Skipped |
| **Test J** | **Sequential Offload + BF16 + VAE Tiling** | `res=480p, 17f, DiT=fp16, VAE=bf16, vae_tiling=True` | **2,902.0 MB** 🟢 | **270s** (9.0s/it) | **58s** 🟢 | **328.2s (5.47m)** 🟢 | 🟢 **PASS (-1,898 MB headroom)** | [TEST_J](./TEST_J_vae_tiling_bf16.md) |
| **Test K** | Custom Chunked VAE | *Not required* | — | — | — | — | *Gate F0.5 passed by Test J* | Standby |
| **Test L** | Chunked VAE + GPU Residency | *Future exploration* | — | — | — | — | *Deferred to Phase F4* | Standby |

---

## 3. Phase F0.6 — Windows WDDM Concurrency Evaluation 🟢 PASS

**Gate Target:** Measured `cudaMemcpyAsync` (H2D) / `matmul` overlap under WDDM ≥ **10%** to keep Phase F3 active.  
**Result:** Overlap of **79.9–96.3%** (512 / 1024 / 2048 MB) → **Phase F3 stays active**.

| Workload (H2D copy) | Copy Alone | Matmul Alone | Serial Sum | Concurrent | Overlap | Gate F0.6 | Report |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 512 MB (32 × 16 MB) | 43.7 ms | 38.7 ms | 82.4 ms | 43.4 ms | **89.3%** | 🟢 PASS | [TEST_F0.6](./TEST_F0.6_wddm_overlap.md) |
| 1024 MB (128 × 8 MB) | 87.2 ms | 69.2 ms | 156.4 ms | 86.8 ms | **79.9%** | 🟢 PASS | [TEST_F0.6](./TEST_F0.6_wddm_overlap.md) |
| 2048 MB (128 × 16 MB) | 174.4 ms | 172.0 ms | 346.4 ms | 178.4 ms | **96.3%** | 🟢 PASS | [TEST_F0.6](./TEST_F0.6_wddm_overlap.md) |

---

## 4. Phase F1 — Tensor Lifetime Profiler 🟢 COMPLETE

**Gate Target:** Verify per-component & per-DiT-block lifecycle/residency profiling is feasible on the real model (with pure static analytical fallback preserved), and confirm physical residency stays ≤ 4,800 MB.

**Result:** All 32 components traced live · **Peak NVML = 3,223.7 MB** (≤ 4,800 MB) · DiT blocks uniform 1.65–1.74 GB stream · VAE decode activation-dominated (2,112 MB) · text encoder weight-dominated (3,224 MB).

| Component | Kind | Static weights | Peak live NVML | Report |
| :--- | :---: | :---: | :---: | :---: |
| text_encoder[0] | text_encoder | 14,758.5 MB | **3,223.7 MB** | [TEST_F1](./TEST_F1_lifetime_profiler.md) |
| 30 × diT_block[i] | diT_block | 88.6 MB each | 1,645.7–1,743.5 MB | [TEST_F1](./TEST_F1_lifetime_profiler.md) |
| vae (tiled decode) | vae | 242.0 MB | **2,112.1 MB** | [TEST_F1](./TEST_F1_lifetime_profiler.md) |

**Feed-forward:** Uniform DiT residency → low fragmentation prior (F2 unlikely to trigger). VAE residency is activation-driven → sharpens F1.5 decomposition. Physical peak 3.2 GB keeps headroom for F3/F4.

---

## 5. Phase F1.5 — Real Bottleneck Identification & Roadmap Dispatch 🟢 COMPLETE

**Gate Target:** Quantify memory consumption across the 5 categories from ROADMAP §3, evaluate conditional triggers for F2 and F3, and issue formal dispatch.

**Result:**
- Allocator fragmentation is only **44.9 MB (< 1.0% of Gate)** → **Phase F2 (Slab Allocator) BYPASSED / DISCARDED**.
- Phase F3 double-buffering prefetch requires **88.6 MB** against **+1,576.3 MB** free headroom (consumes 5.6% of headroom) → **Phase F3 (Async Scheduler) GREENLIT / APPROVED**.
- Report: [`TEST_F1.5_bottleneck_classification.md`](./TEST_F1.5_bottleneck_classification.md) · Telemetry: [`logs/f1_5_bottleneck_decomposition.json`](../logs/f1_5_bottleneck_decomposition.json).

| Category | Measured Footprint | Nature | Roadmap Decision |
| :--- | :---: | :--- | :--- |
| **1. Static Weights** | 17,658.5 MB total (88.6 MB active) | T5 Text Encoder is 83.6% of weights | Target for Phase F1.7 selective quantization |
| **2. DiT Activations** | ~550–600 MB (1,743.5 MB peak NVML) | Uniform across 30 blocks (delta < 98 MB) | Predictable streaming surface for Phase F3/F4 |
| **3. VAE Activations** | ~770 MB pure activations (2,112 MB peak) | Activation-dominated (8.7× weights) | Contained via 256×256 micro-tiling (Test J) |
| **4. Allocator Frag.** | **44.9 MB** (0.94% of 4.8 GB Gate) | Negligible caching pool delta | **Phase F2 Trigger Refuted (Bypassed)** |
| **5. Prefetch Buffers** | **88.6 MB** (Leaves +1,487.7 MB safety buffer) | Fully safe under 4,800 MB Gate | **Phase F3 Trigger Approved (Greenlit)** |

---

## 6. Phase F1.7 — Selective & Adaptive Quantization 🟢 PASS

**Gate Target:** Lower memory footprint by $\ge 20\%$ through selective quantization while preserving numerical fidelity (Cosine Similarity $\ge 0.99$).

**Result:**
- **Text Encoder (T5-XXL / 14,758.5 MB):** 8-bit quantization achieved **-50.0% (-7,379.2 MB)** weight reduction with **0.999607** cosine similarity and zero NaNs $\rightarrow$ 🟢 **PASS (+30% above gate)**.
- **DiT Blocks (30 blocks):** 8-bit projection quantization shrinks per-block payload from **88.6 MB $\rightarrow$ 45.9 MB (-48.2%)**, saving **-76.9 GB** in cumulative PCIe transfer across 30 diffusion steps.
- Report: [`TEST_F1.7_selective_quantization.md`](./TEST_F1.7_selective_quantization.md) · Telemetry: [`logs/f1_7_quantization_benchmark.json`](../logs/f1_7_quantization_benchmark.json).

| Component | FP16 Footprint | Quantized (8-bit) | Net Savings | Fidelity (Cosine Sim) | Gate ($\ge 20\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Encoder (T5-XXL)** | 14,758.5 MB | **7,379.2 MB** | **-7,379.2 MB (-50.0%)** | **0.999607** | 🟢 **PASS** |
| **Single DiT Block** | 88.6 MB | **45.9 MB** | **-42.7 MB (-48.2%)** | Near-lossless | 🟢 **PASS** |
| **Total DiT (30 blocks)** | 2,658.0 MB | **1,377.0 MB** | **-1,281.0 MB (-48.2%)** | Near-lossless | 🟢 **PASS** |

---

## 7. Phase F3 — Budgeted Asynchronous Scheduler 🟢 CERTIFIED PASS

**Gate Target:** Wall-clock speedup $> 0.0\%$, Effective Overlap $\ge 5.0\%$, Peak NVML $\le 4,800\text{ MB}$, and Zero NaNs/Infs on Wan2.1 DiT blocks.

**Certified result (measurement-corrected, [`TEST_F3_verification.md`](./TEST_F3_verification.md)):** repeated A/B (5 reps, order alternated, warmup discarded) with external wall-clock and real overlap/stall instrumentation confirms a **stable speedup of mean +7.2%** (range +6.0% to +8.4%, σ ±1.1%), **overlap 100% measured** (0.00 ms stall, backed by per-block timelines), **2,584 MB peak VRAM** and **0 NaNs/Infs**. All gates PASS → **F3 CERTIFIED, advance to F3 + INT8**.

> **Supersedes the first-run report** [`TEST_F3_async_scheduler.md`](./TEST_F3_async_scheduler.md) whose +10.4% was an optimistic single sample and whose "100% overlap" was an instrumentation artifact (now corrected; the 100% is re-confirmed empirically).

- Report: [`TEST_F3_async_scheduler.md`](./TEST_F3_async_scheduler.md) (first run) · Certified: [`TEST_F3_verification.md`](./TEST_F3_verification.md) · Telemetry: [`logs/f3_verification_benchmark.json`](../logs/f3_verification_benchmark.json) · Methodology: [`docs/F3_MEASUREMENT_VERIFICATION_03.md`](../docs/F3_MEASUREMENT_VERIFICATION_03.md)

| Metric | Condition A (Marley Sync) | Condition B (Marley Async) | Gate Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Wall-Clock Denoise (5 steps)** | 18,127.4 ms | **16,824.1 ms (+7.2%)** | $> 0.0\%$ speedup | 🟢 **PASS** |
| **Effective Overlap** | 0.0% | **100.0%** (measured, 0 ms stall) | $\ge 5.0\%$ | 🟢 **PASS** |
| **Peak VRAM (NVML)** | — | **2,584 MB** | $\le 4,800\text{ MB}$ | 🟢 **PASS** |
| **Numerical Fidelity** | Clean | **0 NaNs / 0 Infs** | Zero corruption | 🟢 **PASS** |
| **Consistency / Reproducibility** | σ ±245.9 ms | σ ±290.6 ms · speedup σ ±1.1 pp | Stable reps | 🟢 **PASS** |

---

## 8. Strategic Governance & Technical Directives

- **[External Technical Directive (SD & ComfyUI Expert Advisory — knowledge acquired through conversation with an AI agent)]:** Binding directive approved by human supervision on 2026-09-08. Establishes the freezing of prior optimization stack before Phase F3, reaffirms Phase F2 bypass, restricts NF4 to extreme low-memory mode, and defines the mandatory 7-metric scorecard for Phase F3 evaluation.

---

## 9. Phase F4 — Adaptive Memory Decision Engine 🟢 CORE VALIDATED (PERF. PENDING)

**Gate Targets (authorized contract [`docs/F4_TEST_SPEC_01.md`](../docs/F4_TEST_SPEC_01.md) v3, §4.4):**
- **Hard Gate:** Peak NVML ≤ 4,800 MB · 0 NaN/Inf · policy correct.
- **Adaptive Gate:** pressure detection + replan + correct switch + headroom recovery ≥ +1.5 GB (deterministic `--pressure-test`), no oscillation.
- **Engineering Target:** Peak NVML ≤ 4,000 MB (not a gate).
- **Optimization Target:** D vs best same-session `{A,B,C}` ≥ 5% (target only, not a validity gate).

**Status:** CORE SYSTEM VALIDATED (Safety + Adaptive Gates PASS); performance certification pending.
Report: [`TEST_F4_adaptive_benchmark.md`](./TEST_F4_adaptive_benchmark.md) · Telemetry: [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json) · Contract: [`docs/F4_TEST_SPEC_01.md`](../docs/F4_TEST_SPEC_01.md).

### Same-session A/B/C/D — 30 steps × 3 reps (external wall-clock)

| Condition | Strategy | Mean (ms) | Min (ms) | Max (ms) | Overlap | NaN/Inf |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Sync FP16 | 113,538.7 | 112,695.5 | 114,992.8 | 0.0% | 0/0 |
| **B** | Async FP16 | **105,441.5** | 104,864.5 | 106,205.0 | 100.0% | 0/0 |
| **C** | Async INT8 | 109,157.9 | 108,946.6 | 109,475.6 | 98.1% | 0/0 |
| **D** | Adaptive (engine) | 109,022.8 | 108,875.6 | 109,275.0 | ~88.4% | 0/0 |

- **Best static = B (Async FP16).** **D vs best = −3.40%** (Optimization Target not met; D stayed INT8 because no pressure ever demanded a switch — a *valid* no-pressure negative).
- Decision overhead (D) = **0.16 ms** · switch 0 · replan 0 · **peak 2,882 MB** · 0 NaN/Inf.
- Key finding (Consejero §3): **Async INT8 is ≈3.5% slower than Async FP16** at this length (dequant on critical path) → reinforces F4's purpose (pick FP16 when no pressure); static INT8 is not globally optimal.

### Adaptive Gate — `--pressure-test` (deterministic)

Profile sequence (8 windows): `performance → memory_safe → memory_safe → performance → performance → …`
Replan 2 (enter window 1 / exit window 3) · switch 2 · overhead 0.07 ms · **oscillation False → Adaptive Gate PASS**.

**Honest caveat:** simulated +600 MB kept free VRAM > 1,500 MB, so `memory_safe` took its
FP16/conservative/evict branch, not the INT8-tighten branch. Hysteresis/dwell/regime-switch are
demonstrated; exercising the INT8-tighten path needs free VRAM < 1,500 MB.

### Decision point (Consejero §7) — reserved to Director

A: retain F4 if consistent benefit (esp. under pressure) · B: retain if robustness justifies
complexity · C: else accept and use **Async INT8 / Performance** static. No-pressure run alone is
not sufficient to classify F4; decision based on evidence.

**Artifacts (F4 decision loop):** [`marley/core/policies.py`](../marley/core/policies.py) (pure selector, 3-state residency) ·
[`marley/core/adaptive.py`](../marley/core/adaptive.py) (EMA observer + hysteresis) · [`f4_adaptive_benchmark.py`](../f4_adaptive_benchmark.py).
Frozen F3/F3+INT8 primitives (`marley/ops/async_stream*.py`) **unchanged** (diff empty).

---

## 10. Phase F5 — Temporal VAE Stitcher (Probe F5-A) 🟢 RETIRED (Resolved by Tiling)

**Canonical F5 Workload & Decision Protocol ([`docs/F5_ROADMAP_PLAN_01.md`](../docs/F5_ROADMAP_PLAN_01.md)):**
- Workload: 832×480 @ **33 frames exact** · Latent `(1, 16, 9, 60, 104)` · BF16 · Spatial tiling 256×256 + causal caching.
- Decision rule: If Peak VRAM ≤ 4,800 MB & Decode ≤ 150 s $\rightarrow$ **RETIRE F5-C** (resolved by existing tiled VAE) $\rightarrow$ Advance to F6.

**Status:** 🟢 **CERTIFIED PASS — RETIRE F5-C**.
Report: [`TEST_F5_vae_probe_33f.md`](./TEST_F5_vae_probe_33f.md) · Telemetry: [`logs/f5_vae_probe_33f.json`](../logs/f5_vae_probe_33f.json) · Script: [`f5_vae_probe_33f.py`](../f5_vae_probe_33f.py).

| Metric | Target / Gate | Measured (F5-A) | Verdict |
| :--- | :---: | :---: | :---: |
| **Peak VRAM (NVML 50ms)** | ≤ 4,800 MB (Hard Gate) | **2,109.0 MB** | 🟢 **PASS (+2,691 MB margin)** |
| **Engineering Target VRAM** | ≤ 4,000 MB | **2,109.0 MB** | 🟢 **MET** |
| **VAE Decode Duration** | ≤ 150.0 s | **26.99 s** | 🟢 **MET (5.5× faster than budget)** |
| **PyTorch Allocated Peak** | — | **1,056.9 MB** | 🟢 Stable |
| **Host Process RAM (RSS)** | Audited | **1,402.6 MB** | 🟢 Negligible system footprint |
| **Integrity (NaNs / Infs)** | 0 / 0 | **0 / 0** | 🟢 **PASS** |
| **Output Shape** | `[1, 3, 33, 480, 832]` | `[1, 3, 33, 480, 832]` | 🟢 **MATCH** |

**Conclusion & Action:**
Because the existing tiled VAE path (`torch.bfloat16` + `enable_tiling()` 256×256) decodes 33 frames effortlessly within ~2.1 GB of VRAM in under 27 seconds, writing a temporal chunker (`marley/ops/vae_stitch.py`) is **unnecessary**. F5-C is **CANCELLED / RETIRED** as resolved by existing tiling. Phase F5 is formally closed; project advances directly to **Phase F6 (480p / 33f End-to-End Pipeline)**.

---

## 11. Test Documentation Architecture

Each test run contains an individual technical report in this directory:
- `TEST_<ID>_<strategy>.md`: Document covering the technical hypothesis, reproducible command line, multi-layer memory telemetry, bottleneck analysis, and gate verdict.
- Generated `.mp4` video outputs are stored locally in `logs/` and excluded from the Git repository due to binary size constraints.
- This file (`TEST_REGISTRY.md`) is the central index/scorecard and is the entry point to all reports.
