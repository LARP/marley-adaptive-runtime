# Informe de Reproducibilidad y Variabilidad Multirun — Phase F6

**Para:** Director de Proyecto & Consejero Técnico Externo  
**De:** Equipo de Arquitectura e Implementación / Marley Runtime  
**Fecha de Inicio de Campaña:** 2026-09-08T22:04:19.537538  
**Total de Corridas Auditadas:** 12 repeticiones contrabalanceadas ($n=3$ por condición, $n-1 = 2$ grados de libertad)  
**Tiempo Total de Campaña:** 122.14 minutos (7328.4 s)  
**Estado Global:** 🟢 **REPRODUCIBILIDAD Y ESTABILIDAD CERTIFICADAS**

---

## 1. Condición Documental y Trazabilidad de Hardware/Software

| Parámetro | Valor de Auditoría |
| :--- | :--- |
| **Commit / Hash Exacto** | `c8dace716c4c892ea4cafc1359f8efd64d26997c` (Rama `main`) |
| **Python Runtime** | Python `3.10.11` |
| **PyTorch / CUDA** | PyTorch `2.6.0+cu124` / CUDA `12.4` |
| **Hardware GPU** | `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (WDDM 3.1) |
| **Resolución & Formato** | `832x480` @ `33` frames (30 steps, FPS=16) |
| **Prompt Canónico** | *"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."* |
| **Negative Prompt** | *"blurry, low quality, watermark, deformed"* |
| **Seed / Guidance** | Seed=`42`, Guidance Scale=`5.0` |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` |
| **Protocolo de Aislamiento** | Aislamiento del contexto de ejecución y liberación de los recursos asociados al proceso (subprocess) |
| **Control Térmico** | Pausa experimental de 60s entre corridas con telemetría basal previa/posterior |

---

## 2. Resumen Consolidado (Media, Mediana, Desviación Estándar y CV%)

| Modo de Streaming | Métrica | Media (μ) | Mediana (p50) | Desv. Estándar (σ) | Coef. Variación (CV%) | Rango [Min – Max] | Criterio Individual de Aceptación |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **sync** | Wall-Clock ($s$) | 531.65 s | 524.07 s | ±16.12 s | **3.03%** | [520.72 – 550.17] s | < 600.0 s individual 🟢 |
| | DiT Denoise ($s$) | 450.94 s | 443.95 s | ±12.22 s | **2.71%** | [443.82 – 465.05] s | Indicador de variabilidad |
| | Cadencia ($s/p$) | 15.03 s/p | 14.8 s/p | ±0.41 s/p | **2.71%** | [14.79 – 15.5] s/p | ≤ 15.0 s/p objetivo |
| | Peak VRAM ($MB$) | 2777.4 MB | 2721.9 MB | ±98.21 MB | **3.54%** | [2719.5 – 2890.8] MB | **≤ 4,800.0 MB 100% indiv.** 🟢 |
| | | | | | | | |
| **async_fp16** | Wall-Clock ($s$) | 515.57 s | 509.59 s | ±13.6 s | **2.64%** | [505.98 – 531.13] s | < 600.0 s individual 🟢 |
| | DiT Denoise ($s$) | 429.66 s | 422.03 s | ±15.3 s | **3.56%** | [419.67 – 447.27] s | Indicador de variabilidad |
| | Cadencia ($s/p$) | 14.32 s/p | 14.07 s/p | ±0.51 s/p | **3.56%** | [13.99 – 14.91] s/p | ≤ 15.0 s/p objetivo |
| | Peak VRAM ($MB$) | 2768.3 MB | 2719.1 MB | ±85.3 MB | **3.08%** | [2719.0 – 2866.8] MB | **≤ 4,800.0 MB 100% indiv.** 🟢 |
| | | | | | | | |
| **async_int8** | Wall-Clock ($s$) | 512.21 s | 508.56 s | ±9.93 s | **1.94%** | [504.62 – 523.44] s | < 600.0 s individual 🟢 |
| | DiT Denoise ($s$) | 433.24 s | 428.78 s | ±8.52 s | **1.97%** | [427.88 – 443.06] s | Indicador de variabilidad |
| | Cadencia ($s/p$) | 14.44 s/p | 14.29 s/p | ±0.29 s/p | **1.98%** | [14.26 – 14.77] s/p | ≤ 15.0 s/p objetivo |
| | Peak VRAM ($MB$) | 2741.4 MB | 2739.0 MB | ±4.6 MB | **0.17%** | [2738.5 – 2746.7] MB | **≤ 4,800.0 MB 100% indiv.** 🟢 |
| | | | | | | | |
| **adaptive** | Wall-Clock ($s$) | 527.07 s | 526.95 s | ±24.05 s | **4.56%** | [503.09 – 551.18] s | < 600.0 s individual 🟢 |
| | DiT Denoise ($s$) | 427.63 s | 422.84 s | ±9.63 s | **2.25%** | [421.34 – 438.72] s | Indicador de variabilidad |
| | Cadencia ($s/p$) | 14.25 s/p | 14.09 s/p | ±0.32 s/p | **2.26%** | [14.04 – 14.62] s/p | ≤ 15.0 s/p objetivo |
| | Peak VRAM ($MB$) | 4047.83 MB | 4007.0 MB | ±70.73 MB | **1.75%** | [4007.0 – 4129.5] MB | **≤ 4,800.0 MB 100% indiv.** 🟢 |
| | | | | | | | |

---

## 3. Telemetría de Hardware y Control de Condiciones Térmicas (NVML)

| Modo | Temp. Basal Pre (°C) | Temp. Final Post (°C) | Clock GPU Post (MHz) | Control Térmico |
| :--- | :---: | :---: | :---: | :--- |
| **sync** | 58 °C | 76.33 °C | 210 MHz | Pausa experimental de 60s aplicada para amortiguar acumulación térmica |
| **async_fp16** | 62.67 °C | 75 °C | 220 MHz | Pausa experimental de 60s aplicada para amortiguar acumulación térmica |
| **async_int8** | 62.33 °C | 75.33 °C | 210 MHz | Pausa experimental de 60s aplicada para amortiguar acumulación térmica |
| **adaptive** | 61.33 °C | 74.33 °C | 210 MHz | Pausa experimental de 60s aplicada para amortiguar acumulación térmica |

*Nota metodológica:* El cooldown de 60 segundos constituye una medida de control experimental y mitigación de acumulación térmica, no una garantía de isotermia absoluta entre ejecuciones.

---

## 4. Comparativa Relativa entre Arquitecturas y Análisis de Variabilidad

- **Async FP16 frente a Sync (Baseline):** **-3.02%** en latencia total media (515.57s vs 531.65s), confirmando la ventaja del solapamiento PCIe/compute.
- **Async INT8 frente a Async FP16:** **-0.65%** en wall-clock medio durante esta campaña (512.21s vs 515.57s). La diferencia es pequeña (-3.36s) y, con $n=3$ por condición ($n-1 = 2$ grados de libertad), no debe interpretarse como evidencia de una ventaja de rendimiento estadísticamente estable. El resultado confirma que INT8 no introdujo una penalización significativa de wall-clock bajo las condiciones de esta campaña, manteniéndose Async FP16 como baseline estático de referencia.
- **Adaptive frente a Async FP16:** **+2.23%** en régimen nominal (527.07s vs 515.57s). Adaptive no se valida por superar necesariamente al mejor modo estático, sino por mantener un coste nominal razonable mientras proporciona capacidad de adaptación dinámica ante contingencias o perturbaciones de memoria.
- **Observación sobre la Variabilidad de Adaptive:** Adaptive presentó la mayor variabilidad relativa de wall-clock de la campaña ($CV=4.56\%$, frente a 1.94%–3.03% de los modos estáticos). Aunque todas las ejecuciones cumplieron holgadamente los hard gates, esta dispersión queda registrada como observación para F7 para monitorizar si vuelve a manifestarse bajo mayores demandas de memoria o resolución.

---

## 5. Dictamen Final de Reproducibilidad y Transición a F7

> [!IMPORTANT]
> **CUMPLIMIENTO INDIVIDUAL DEL 100% DE LOS HARD GATES:**  
> - **Peak NVML:** El 100% de las corridas individuales cumplió estrictamente $\le 4,800.0\text{ MB}$ de VRAM física (Rango global observado: [2,719.0 – 4,129.5] MB).  
> - **Integridad Numérica:** 0 NaNs / Infs en el 100% de las repeticiones individuales.  
> - **Integridad de Salida:** 100% de las ejecuciones generaron un archivo MP4 válido y correctamente decodificable.  
> - **Línea Base para 720p:** Los resultados de F6 proporcionan una referencia empírica para establecer el presupuesto inicial de F7. No obstante, la viabilidad de 720p no debe extrapolarse matemáticamente a partir de los resultados de 480p y deberá determinarse mediante medición directa en hardware.
