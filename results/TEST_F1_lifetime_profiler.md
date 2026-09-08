# Test F1 — Tensor Lifetime Profiler (real Wan2.1-T2V-1.3B)

**Phase:** F1 — Tensor Lifetime Profiler
**Date:** 2026-09-08T14:13:45Z (main run)
**Status:** 🟢 **COMPLETE**
**Scripts:** [`f1_lifetime_profiler.py`](../f1_lifetime_profiler.py) · [`marley/profiler/lifetime.py`](../marley/profiler/lifetime.py)

---

## 1. Identification of the Test

| Field | Value |
| :--- | :--- |
| Test name | **F1 — Tensor Lifetime Profiler** |
| Phase | F1 (ROADMAP v5) |
| Objective | Trace lifecycle events + per-component / per-DiT-block peak residency across text encoder, DiT transformer and VAE on the real model, and verify dynamic profiling is tractable (or fall back to static). |
| Final status | 🟢 **COMPLETE** |

### Execution history (all runs that produced F1 artefacts)

| # | Config | Log | Telemetry / Video | Instrumentation state | Role |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 5f / 2st / group-blocks | [`f1_lifetime_480p_5f_fp16_vae_bf16_cpu_2st_grp_20260908_104717.log`](../logs/f1_lifetime_480p_5f_fp16_vae_bf16_cpu_2st_grp_20260908_104717.log) | — (crashed on `output.frames` guard) | early smoke | **Discarded** |
| 2 | **17f / 30st** | [`f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_104921.log`](../logs/f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_104921.log) | [`f1_..._105742_telemetry.json`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_105742_telemetry.json) | no VAE recurse (`vae.forward_calls=0`), DiT grouped | **Superseded** (incomplete VAE instrumentation) |
| 3 | 5f / 2st | [`f1_lifetime_480p_5f_fp16_vae_bf16_cpu_2st_20260908_105851.log`](../logs/f1_lifetime_480p_5f_fp16_vae_bf16_cpu_2st_20260908_105851.log) | [`f1_..._110122_telemetry.json`](../logs/f1_480p_5f_fp16_vae_bf16_cpu_2st_20260908_110122_telemetry.json) | VAE recurse active (`vae.forward_calls=3810` ✓), DiT grouped | **Instrumentation validation** |
| 4 | **17f / 30st** | [`f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_110517.log`](../logs/f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_110517.log) | [`f1_..._111345_telemetry.json`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json) + [`f1_..._111345.mp4`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345.mp4) | 30 individual DiT blocks + VAE recurse (`vae.forward_calls=9570` ✓) | **MAIN / canonical** |

> [!IMPORTANT]
> Only run **#4 (111345, 17f/30st)** is used to declare the F1 gate. Runs #2 and #3 are referenced
> solely to document the instrumentation work; their numbers are **not** mixed into the gate verdict.
> Run #1 produced no artefact and is excluded.

---

## 2. Environment

| Component | Detail |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, CC 8.6) |
| Physical VRAM | 6,144 MB (6.0 GB) |
| Effective gate | 4,800 MB (NVML physical residency) |
| CPU / RAM | Intel Core i7-13650HX (14c/20t) · 24 GB DDR5 (system-level, from ROADMAP spec) |
| OS / Driver | Windows 11 · Driver 581.86 · CUDA 13.0 (runtime 12.4, WDDM) |
| Python stack | Python 3.10.11 · PyTorch 2.6.0+cu124 · diffusers 0.40.0 |
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

---

## 3. Exact Command (main run #4)

```
.venv\Scripts\python.exe f1_lifetime_profiler.py --res 480p --frames 17 --steps 30 --vae-dtype bf16 --vae-tiling --offload cpu
```

### Configuration

| Option | Value |
| :--- | :--- |
| Resolution | 480p (832 × 480) |
| Frames | 17 |
| Denoise steps | 30 |
| Scheduler | diffusers pipeline default (from `WanPipeline`) |
| DiT / text-encoder dtype | FP16 |
| VAE dtype | BF16 |
| VAE tiling | Enabled |
| CPU offload | `enable_sequential_cpu_offload()` |
| Guidance scale | 5.0 |
| Seed | 42 (CPU generator) |
| Profiler interval | 50 ms NVML sampler |
| DiT granularity | individual blocks (`group_blocks=False`) |

---

## 4. Performance

Metrics below are from the **main run #4 (17f/30st)** unless stated. Per-component wall time is
measured by the profiler hooks (includes the sampler's CPU-side overhead only where a forward is
active; DiT times are aggregated per component over 30 steps × 2 CFG calls = 60 forwards/block).

| Metric | Value |
| :--- | :---: |
| Total wall-clock inference | **455.1 s** (~7.6 min) |
| Text encoder total forward time | 67.4 s (2 calls; max 64.1 s) |
| DiT: aggregate forward time (sum over 30 blocks) | 313.2 s |
| DiT: per-block mean / max forward time | 10.44 s / 10.80 s (60 calls each) |
| VAE total forward time | 55.9 s (micro-calls aggregated) |

> The reported DiT aggregate (~313 s) is the sum of per-block hook times and therefore overlaps with
> the true denoise wall-clock; it is an instrumentation bound, not a stage decomposition of the
> pipeline. Stage-accurate decomposition is deferred to F1.5.

---

## 5. Memory

Metrics below are from the **main run #4 (17f/30st)**.

| Memory layer | Peak | Gate (4,800 MB) | Margin |
| :--- | :---: | :---: | :---: |
| **NVML physical residency (global)** | **3,223.7 MB** | ≤ 4,800 MB | **+1,576.3 MB** |
| PyTorch max allocated | 2,021.1 MB | < 4,800 MB | +2,778.9 MB |
| PyTorch max reserved | 2,066.0 MB | < 4,800 MB | +2,734.0 MB |
| RAM / Working Set | `N/A — not measured` | — | — |

```
margen = 4,800 MB - 3,223.7 MB = 1,576.3 MB
```

> **Layering (ROADMAP §2):** NVML physical residency (3,224 MB) is the ground-truth gate. PyTorch
> allocated (2,021 MB) / reserved (2,066 MB) are auxiliary and are **not** interchangeable with NVML:
> the ~1.2 GB delta is WDDM driver residency/paging. Static model weight bytes (17,658 MB aggregate)
> and per-forward activation deltas are separate figures reported in §6; none of them equal the peak
> physical residency.

---

## 6. Per-Component Profile (main run #4, 17f/30st)

For each component: static weights (analytical), live NVML peak attributed while that component was
the active driver of execution, and per-forward torch allocation delta.

| Component | Kind | Forward calls | Static weights | torch alloc Δ / call | **NVML peak attributed** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| text_encoder[0] | text_encoder | 2 | 14,758.5 MB | 12.1 MB | **3,223.7 MB** |
| diT_block[0] … diT_block[29] | diT_block (30) | 60 each | 88.6 MB each | 22.85–23.15 MB | 1,645.7–1,743.5 MB |
| vae | vae | 9,570 | 242.0 MB | 96.0 MB | **2,112.1 MB** |

### Static vs. live residency

| Category | Static footprint | Peak live NVML (sequential offload) | Interpretation |
| :--- | :---: | :---: | :--- |
| Text encoder (UMT5) | 14,758.5 MB | 3,223.7 MB | Largest weights; never resident whole |
| 30 × DiT transformer blocks | 2,658.0 MB (30 × 88.6) | ~1.65–1.74 GB (one block active) | Uniform, flat residency |
| VAE decoder | 242.0 MB | 2,112.1 MB | Small weights; decode **activations** dominate |
| **Total static** | **17,658.5 MB** | **3,223.7 MB peak live** | Offload keeps live ≪ static |

---

## 7. Lifetime Profiler — What It Observed

### Hooks & instrumentation
- `TensorLifetimeProfiler` attaches a `register_forward_pre_hook` + `register_forward_hook` pair per
  component. On enter it records a timestamp + `torch.cuda` allocated/reserved and pushes the
  component onto a stack; on exit it records elapsed time and deltas and pops.
- **NVML attribution:** a background sampler reads physical device residency every 50 ms and assigns
  it to the top of the enter/exit stack (the component actively driving execution). Under sequential
  CPU offload only the executing submodule is resident, so a block's attributed residency ≈ its own
  weights + activations.
- **Static analytical fallback:** `static_weights()` sums `param.numel() * element_size()` per dtype
  with no CUDA/NVML dependency — always available as the F1 kill-gate fallback.

### Micro-invocations & the VAE decode finding
- Components instrumented: 1 text encoder + 30 DiT blocks + VAE (leaves).
- DiT: **60 forwards/block** (30 steps × 2 CFG calls).
- VAE: **9,570 micro-invocations** (tiled decode dispatches through many leaf ops).

> **Instrumentation issue identified & validated (not an inference failure).**
> In run #2 the top-level VAE reported `vae.forward_calls = 0`. The cause is architectural, not a
> model fault: **Diffusers performs VAE decoding via `vae.decode(...)`, which does not invoke the
> VAE module's own `__call__`/`forward`**, so a hook on the VAE's top-level `forward` never fires.
> The profiler was corrected to hook every **leaf submodule** of the VAE (`recurse=True`).
> This was validated in run #3 (5f/2st: `vae.forward_calls = 3810` ✓) and confirmed in the main
> run #4 (17f/30st: `vae.forward_calls = 9570` ✓). VAE residency (2,112 MB) is therefore captured.

### Behaviour observed
- **Allocations/deallocations:** DiT forward deltas are small and uniform (≈23 MB/call) → the PyTorch
  caching allocator recycles buffers within each block; no single block spikes.
- **VAE decode** shows the largest per-forward delta (≈96 MB) with many short micro-calls — consistent
  with activation-dominated decode.

### Limitations of the profiler
- DiT aggregate wall time is a sum of per-block hook windows (overlaps true denoise time); not a clean
  stage decomposition → F1.5.
- NVML attribution is stack-based and exact only while a single component drives the GPU (true under
  sequential offload); under concurrent/multi-stream execution attribution would be approximate.
- torch reserved delta quantization: deltas reflect caching-allocator block growth, not true per-tensor
  high-water marks; NVML remains the ground truth.

---

## 8. Gate Result — 4,800 MB

Metrics are from the **main run #4 (17f/30st)** only.

| Metric | Result | Gate | State |
| :--- | :---: | :---: | :---: |
| Peak Global NVML | 3,223.7 MB | ≤ 4,800 MB | 🟢 **PASS** |
| PyTorch max allocated | 2,021.1 MB | < 4,800 MB | 🟢 PASS |
| PyTorch max reserved | 2,066.0 MB | < 4,800 MB | 🟢 PASS |
| OOM / crashes | 0 | 0 | 🟢 PASS |

### F1 kill-gate
ROADMAP v5, Phase F1 kill gate: *"If fine-grained dynamic profiling proves intractable due to runtime
overhead or driver virtualization, fall back to static analytical memory modeling."*

| Criterion | Gate | Measured | State |
| :--- | :---: | :---: | :---: |
| Dynamic tracing tractable? | Must not be intractable | All 32 components traced live | 🟢 PASS |
| Static analytical fallback | Must remain available | `static_weights()` present & computed | 🟢 PASS |

---

## 9. Public / External Technical Analysis

### What was demonstrated
- **End-to-end execution** of Wan2.1-T2V-1.3B at 480p / 17f / 30 steps under sequential CPU offload
  completed without OOM; a video was produced.
- **A Tensor Lifetime Profiler works on the real model**: per-component and per-DiT-block lifecycle and
  residency were captured live with acceptable overhead (dynamic path retained).
- **Physical peak = 3,223.7 MB**, with **1,576 MB** of headroom under the 4,800 MB gate.
- **DiT residency is uniform** (1.65–1.74 GB across all 30 blocks) → low fragmentation prior.
- **VAE decode is activation-dominated**: 2,112 MB residency on only 242 MB of weights.
- **Text encoder is weight-dominated**: 14.7 GB static, but streamed (3,224 MB live) under offload.
- The diffusers `vae.decode()` instrumentation gap was identified, fixed, and validated.

### What was NOT demonstrated
- **720p was not tested.** Only 480p evidence exists here.
- **Real-time performance** was not demonstrated (455 s for 17f).
- **No guarantee** that other resolutions/frame counts/configurations will fit in 4.8 GB.
- **F4 (adaptive engine) is not resolved**; F1 only supplies profiling, not an adaptive policy.
- **No claim that marley-runtime outperforms other runtimes** — no comparative benchmark here.
- **F2/F3 are not activated.** F1 supplies priors; their triggers are evaluated in F1.5+.
- DiT aggregate timing is an instrumentation sum, not a validated stage decomposition.

### Reproducibility
```
Hardware : RTX 3050 6GB Laptop · Windows 11 · driver 581.86
Model    : Wan-AI/Wan2.1-T2V-1.3B-Diffusers
Command  : .venv\Scripts\python.exe f1_lifetime_profiler.py --res 480p --frames 17 \
             --steps 30 --vae-dtype bf16 --vae-tiling --offload cpu
Artefacts: logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json
           logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345.mp4
           logs/f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_110517.log
```
Running the profiler requires the model to be cached (~5 GB) and the 4.8 GB NVML physical gate to be
tracked via the continuous sampler. Re-running yields the same command/config; peak values may vary by
a few MB run-to-run due to WDDM clock/driver residency.

---

## 10. Limitations & Anomalies

| Item | Detail |
| :--- | :--- |
| VAE top-level `forward_calls=0` (run #2) | Instrumentation artifact: `vae.decode()` bypasses top-level `forward`. Fixed via leaf recurse hooks; validated in runs #3–#4. **Not** an inference failure. |
| DiT grouped in runs #2/#3 | Pre-fix discovery aggregated the 30-block `ModuleList` as one group. Fixed to enumerate the 30 `WanTransformerBlock`s. Main run #4 uses individual blocks. |
| Run #1 (smoke) crash | Crashed on an `output.frames` truthiness guard; produced no artefact; excluded. |
| DiT timing overlap | Per-block times summed ≠ true denoise decomposition. |
| NVML attribution accuracy | Exact under sequential offload (single active driver); approximate under concurrency. |
| RAM / Working Set | `N/A — not measured`. |
| One seed / one prompt | Only seed 42 and a single prompt were profiled. |

---

## 11. Verdict — F1

**Verdict: 🟢 PASS**

Justification (all gate-relevant metrics from main run #4, 17f/30st, which has **complete VAE
instrumentation** — no conditional required):

1. **Dynamic lifetime profiling is tractable** on the real model — all 32 components traced live, so
   the F1 kill-gate fallback was **not** required (though it remains implemented).
2. **Physical peak 3,223.7 MB ≤ 4,800 MB** gate → PASS with 1,576 MB margin.
3. **Zero OOM/crashes** on the declaring run.
4. The VAE instrumentation gap (found in run #2) was **corrected and validated** in run #3 and is
   **active in the declaring run #4** (`vae.forward_calls=9570`), so VAE residency is genuine.
5. Exit code 0 was observed, but the PASS is asserted against the **F1 gate criteria** above, not
   merely the exit status.

---

## 12. Roadmap Implications

New information Marley Runtime now holds that it did not before:

| Capability | Enables |
| :--- | :--- |
| Per-DiT-block live residency (uniform 1.65–1.74 GB) | **F1.5** decomposition of the DiT stream; informs **F2** trigger prior (low fragmentation → slab allocator unlikely) |
| VAE residency attributed as **activation-dominated** (2,112 MB on 242 MB weights) | **F1.5** VAE activation budget; **F5** stitcher design input |
| Text encoder weight-dominated (3,224 MB ceiling) | **F1.7** selective-quantization target ranking; **F1.5** |
| Demonstrated WDDM-aware NVML attribution methodology | reusable telemetry primitive for **F3/F4** runtime observation |
| 1,576 MB headroom measured at 480p/17f | budget ceiling that **F3** double-buffering must not exceed (F1.5 will verify) |

How it feeds the **F4 Adaptive Decision Engine**: the per-block cost (time) and residency (MB) tables
produced here are exactly the "analytical cost profiles … updated via lightweight runtime observations"
that ROADMAP F4 specifies (free VRAM headroom, layer compute elapsed time). A per-block scheduler can
rank blocks by measured residency rather than estimate them, using NVML headroom as its live gate.

**Not pre-advanced:** F2, F3, F4, F5 and F6 remain pending their own triggers/gates. F1 provides
evidence and primitives only; F1.5 is the next gate to produce a prioritized dispatch.

---

## 13. References & Artefacts

- Report: [`results/TEST_F1_lifetime_profiler.md`](./TEST_F1_lifetime_profiler.md)
- Telemetry JSON (main): [`logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345_telemetry.json)
- Video (main): [`logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345.mp4`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_111345.mp4)
- Log (main): [`logs/f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_110517.log`](../logs/f1_lifetime_480p_17f_fp16_vae_bf16_cpu_30st_20260908_110517.log)
- Instrumentation-validation telemetry (5f/2st): [`logs/f1_480p_5f_fp16_vae_bf16_cpu_2st_20260908_110122_telemetry.json`](../logs/f1_480p_5f_fp16_vae_bf16_cpu_2st_20260908_110122_telemetry.json)
- Superseded 17f/30st telemetry (incomplete VAE instrumentation): [`logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_105742_telemetry.json`](../logs/f1_480p_17f_fp16_vae_bf16_cpu_30st_20260908_105742_telemetry.json)
- Profiler source: [`marley/profiler/lifetime.py`](../marley/profiler/lifetime.py) · [`marley/profiler/__init__.py`](../marley/profiler/__init__.py)
