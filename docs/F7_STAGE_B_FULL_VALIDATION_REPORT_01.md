# F7 Stage B — 720p Full Re-evaluation Report (30 Steps)

**Date:** 2026-09-10T02:32:19.847809  
**Protocol Reference:** [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md) §6  
**Classification:** **🔴 COMPLETE REJECTION**  

---

## 1. Executive Scorecard

> **Official Matrix Verdict:** Escenario 4: Both Gate A and Gate B failed.

| Gate / Metric | Pre-registered Limit | Observed Value | Result |
| :--- | :---: | :---: | :---: |
| **Gate A (Physical VRAM)** | $\le 4800.0$ MB | **4996.0 MB** | 🔴 FAIL |
| **Gate B (Net Delta $\Delta_{S0}$)** | $\le 3850.0$ MB | **4046.3 MB** | 🔴 FAIL |
| **Operational Delta $\Delta_{S1}$** | N/A (Analytical) | **3912.5 MB** | Observed |
| **VAE Decode & MP4** | 33 frames valid, 0 NaNs | True | 🟢 OK |

## 2. Milestone Telemetry Breakdown

- **S0 Idle Baseline:** 949.8 MB (spread: 121.4 MB)
- **Control A Delta:** 500.0 MB
- **S1 Operational Baseline:** 1083.5 MB
- **S2 Workload Peak:** 4996.0 MB
- **S3 Post-Workload Plateau:** 1650.9 MB (spread: 105.7 MB)
- **Control B Delta:** 508.3 MB
- **S4 Final Baseline:** 1667.5 MB (spread: 75.1 MB)

## 3. Epistemological and Engineering Governance

The run failed to satisfy both gates simultaneously. Gate A observed 4996.0 MB vs 4,800.0 MB limit. Gate B observed 4046.3 MB vs 3,850.0 MB limit.
