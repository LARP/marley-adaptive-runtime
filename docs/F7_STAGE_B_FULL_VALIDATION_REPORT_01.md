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

### 3.1 Lo que NO se demostró
- No se demostró cumplimiento del Gate A de 4,800.0 MB (déficit cuantificado de **+196.0 MB**).
- No se demostró cumplimiento del Gate B de 3,850.0 MB (déficit cuantificado de **+196.3 MB**).
- No se autoriza la certificación ni la integración de la resolución 1280×720 (720p) en la rama de producción.

### 3.2 Lo que SÍ se demostró
- **Estabilidad Operacional Completa:** Ejecución ininterrumpida de 30 pasos DiT continuos sin OOM y sin activar el kill-switch.
- **Cadencia Nominal Sostenida:** **58.47 s/paso** (denoise total: 1,754.2 s).
- **Control del Allocator de PyTorch:** Reserva estabilizada de forma plana en **~3.77 GB** (`reserved` final: 3,766.0 MB; `alloc` final: 601.3 MB), demostrando la eficacia del drenaje de inicio de paso y la liberación en costura (`seam release`).
- **Ausencia de Fuga Monotónica:** El pico físico absoluto ocurrió en el **Paso 6 (4,996.0 MB)** y descendió en los pasos finales (4,714 – 4,773 MB en pasos 24–30).
- **Integridad Perceptual y Funcional:** Decodificación VAE tileada completada en 75.70 s con 0 NaNs/Infs y video MP4 válido exportado.

### 3.3 Pregunta de Investigación Abierta
> *«¿Qué evento o conjunto de eventos produce el sobrepico de residencia física de aproximadamente 196 MB alrededor del Paso 6, y por qué dicho componente deja de requerirse o deja de permanecer físicamente residente en los pasos posteriores?»*
