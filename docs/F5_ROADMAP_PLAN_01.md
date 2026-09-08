# Experimental Plan — Phase F5 (Temporal VAE Stitcher · re-scoped)

## DRAFT for external-agent validation — NOT yet approved.

**Phase:** F5 — Temporal VAE Stitcher
**Date:** 2026-09-08
**Status:** 🟢 **RATIFIED & APPROVED (Decision-First)** — F5-A authorized by Director & Consultant.
**Relationship to roadmap:** F0.5 Test J (spatial-temporal 256×256 tiling, BF16) already resolved the
VAE **spatial** memory spike at 480p/17f (peak 2,902 MB; VAE decode 58 s). F5's *original* trigger
(VAE as primary OOM bottleneck) is therefore **largely satisfied**. This plan **re-scopes** F5 to
its unaddressed remainder that becomes relevant at the F6 primary target **480p / 33 frames**:
**temporal** VAE decoding of long sequences under the fixed 4.8 GB gate.

---

## 1. Background & motivation (evidence)

| Fact | Source |
| :--- | :--- |
| VAE decode was the F0 bottleneck (322 s, +4,187 MB over gate) | `TEST_F0_baseline_ref.md` |
| BF16 + spatial-temporal tiling cut peak to 2,902 MB and decode to 58 s (Test J, PASS) | `TEST_J_vae_tiling_bf16.md` / `TEST_REGISTRY.md` §2 |
| VAE peak is **activation-dominated** (2,112 MB on 242 MB weights) | `TEST_F1_lifetime_profiler.md` |
| Native causal temporal caching is already used by the tiling path | F0.5 reports |

**Canonical F5 Workload (Frozen):**
- Resolution: **832×480** (480p)
- Output frames: **33 frames exact**
- Latent shape: **(1, 16, 9, 60, 104)** (where $[33-1]/4 + 1 = 9$ temporal latents)
- Precision: **torch.bfloat16**
- Tiling: native `enable_tiling()` with **256×256** spatial tiles and causal temporal caching active.

**Null hypothesis (falsifiable):** *"At 480p/33f under the 4.8 GB gate, the existing
tiled VAE decode fits VRAM (≤ 4,800 MB) and meets the latency budget (≤ 150 s); no additional temporal chunking/stitcher
is required."*

---

## 2. Scope control

- **In scope:** VAE temporal-decode investigation at 480p/33f; measurement probe (F5-A); (conditional)
  temporal characterization (F5-B); (strictly conditional) temporal chunking + overlapping blending (F5-C).
- **Out of scope during F5:** any change to `BudgetedAsyncStreamer` / `INT8BudgetedStreamer`
  (frozen F3/F4 primitives); spatial tiling internals (reuse native); DiT denoising; NF4;
  the F4 adaptive engine (frozen, unchanged).
- Reuse existing: native `enable_tiling()`, BF16 VAE, NVML sampler (50 ms), the three-layer
  telemetry model, the same-session measurement discipline, host RAM/pinned tracking.

---

## 3. Objective Kill Gates & Transition Criteria

| Gate / Metric | Threshold / Condition | Consequence |
| :--- | :---: | :--- |
| **Peak VRAM (Hard Gate)** | ≤ 4,800 MB | Hard kill gate. If > 4,800 MB or OOM $\rightarrow$ triggers F5-B/F5-C |
| **Engineering Target VRAM** | ≤ 4,000 MB | Design target for operational safety |
| **VAE Decode Latency Target** | ≤ 150 s | Engineering target for 33f. If ≥ 150 s $\rightarrow$ investigate scaling |
| **Integrity Gate** | NaN / Inf = 0, valid output | Mandatory for PASS |
| **Host / Pinned Memory** | Audited RSS & pinned | No memory thrashing tolerated |
| **Temporal Chunker Value (H3)** | VRAM saving < 20% | Cancel stitcher build / retire F5-C |
| **Baseline OOM Condition** | Baseline > 4,800 MB | Chunker evaluated by Viability Restoration (`OOM → PASS`) |

---

## 4. Execution Protocol (3-Stage Decision Flow)

```text
[ F5-A: CANONICAL 33f PROBE ] (f5_vae_probe_33f.py)
              │
              ├──► Peak VRAM ≤ 4,800 MB, Decode ≤ 150 s, 0 NaNs
              │    └──► 🟢 RETIRE F5 — RESOLVED BY EXISTING TILED VAE ──► Direct advance to F6
              │
              └──► Peak VRAM > 4,800 MB (OOM) OR Latency ≥ 150 s
                   │
                   ▼
         [ F5-B: TEMPORAL CHARACTERIZATION ] (17f → 25f → 33f curve)
                   │
                   ▼
         [ F5-C: TEMPORAL STITCHER ] (Strictly conditional build: only if H3 justified)
```

---

## 5. Artifacts

- `f5_vae_probe_33f.py` — measurement probe (F5-A, authorized).
- `marley/ops/vae_stitch.py` — temporal chunker + blender (only in outcome F5-C, conditional).
- `logs/f5_*.json`, `results/TEST_F5_*.md`, registry row.

---

## 6. Closure of Open Questions (Ratified Consensually)

1. **Gating vs F4:** F5-A probe is decoupled from F4 retention and runs immediately.
2. **Expectation on 33f:** If 33f fits (≤ 4.8 GB & ≤ 150 s), F5 concludes with `RETIRE F5`.
3. **Reference for Quality:** Boundary Temporal Discontinuity (luminance & frame delta at seam) plus spatial PSNR/SSIM.
4. **Registry alignment:** Test Registry will record F5 as either `RETIRE (Resolved by Tiling)` or `CHUNKER VALIDATED`.

---

*Marley Runtime — Phase F5 Roadmap · en memoria de Marley 🐾.*
