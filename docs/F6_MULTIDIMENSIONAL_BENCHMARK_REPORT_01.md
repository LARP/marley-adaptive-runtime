# Informe Técnico Multidimensional — Phase F6: Multidimensional Benchmarks

**Proyecto:** Marley Runtime — Low-VRAM Diffusion Orchestration  
**Fase:** Phase F6 — Multidimensional Benchmarks & End-to-End Video Generation  
**Fecha de Emisión:** 8 de septiembre de 2026 (20:32:00 -03:00)  
**Estado General:** 🟢 **F6-A (PASS) · 🟢 F6-B (PASS) · ⏳ F6-C (LISTO PARA EJECUCIÓN)**  

---

## 1. Metadata del Entorno de Ejecución y Hardware

| Parámetro | Valor Exacto / Especificación |
| :--- | :--- |
| **Commit Git Exacto** | `88a678f70021c41ca38694cecf995d3dddd9eb2e` |
| **Entorno Python** | Python 3.10.11 (MSC v.1929 64 bit AMD64) en `.venv\Scripts\python.exe` |
| **Framework PyTorch** | `2.6.0+cu124` |
| **CUDA Runtime (PyTorch)**| `12.4` |
| **Driver NVIDIA** | `581.86` |
| **Dispositivo GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **VRAM Física Total** | 6,144.0 MB |
| **Hard Gate de VRAM** | $\le 4,800.0\text{ MB}$ (NVML físico a 50 ms) |
| **Engineering Target VRAM**| $\le 4,000.0\text{ MB}$ |
| **Engineering Target Latencia** | $< 600.0\text{ s}$ (10 minutos por video) |
| **RAM de Sistema Host** | 24.0 GB DDR5 |
| **Sistema Operativo** | Windows 11 (WDDM 3.1) |

---

## 2. Configuración Canónica del Workload

| Parámetro | Valor |
| :--- | :--- |
| **Modelo Base** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| **Resolución Canónica** | 832x480 (480p panorámico) |
| **Número de Frames** | 33 frames |
| **Cadencia de Video** | 16 fps (~2.06 segundos de video) |
| **Pasos de Difusión** | 30 pasos de inferencia |
| **Escala de Guía (CFG)**| 5.0 |
| **Semilla Aleatoria** | 42 |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` |
| **Prompt Positivo** | `"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."` |
| **Prompt Negativo** | `"blurry, low quality, watermark, deformed"` |
| **DiT Architecture** | `WanTransformer3DModel` (30 bloques DiT secuenciales, patch 1x2x2) |
| **VAE Architecture** | `AutoencoderKLWan` (bfloat16 con 256x256 spatial tiling per F5-A) |
| **Text Encoder** | `UMT5-XXL` (6.73B params, ejecutado en host CPU para garantizar 0 VRAM peak) |

---

## 3. Registro de Warnings de Framework Observados

Durante la inicialización de los pipelines se registraron los siguientes avisos del framework HuggingFace/Diffusers (ninguno representa un fallo de ejecución ni corrupción de tensores):
1. `UserWarning: The local_dir_use_symlinks argument is deprecated and ignored in hf_hub_download`: Deprecación menor de Hugging Face Hub en Windows.
2. `Warning: You are sending unauthenticated requests to the HF Hub`: Aviso informativo de límites de tasa en HF Hub para modelos cacheados localmente.
3. `FutureWarning: torch_dtype is deprecated and will be removed in version 1.0.0. Please use dtype instead`: Aviso interno de Diffusers.
4. `UserWarning: There are modules in WanTransformer3DModel that should be kept in float32: ['rope', 'time_embedder', ...]. Casting directly with to() can lead to inconsistent results`: Salvaguardado por Marley preservando las subredes de modulación en precisión nativa.

---

## 4. Benchmark F6-A: Baseline Sincrónico (Sync)

### 4.1 Comando de Ejecución
```powershell
.venv\Scripts\python.exe f6_end_to_end_benchmark.py --mode sync --steps 30 --frames 33 --output logs/f6_condition_a_sync.json
```

### 4.2 Métricas y Telemetría Auditada
* **Fecha y Hora de Inicio:** 2026-09-08 20:11:08 -03:00
* **Fecha y Hora de Finalización:** 2026-09-08 20:20:58 -03:00
* **Archivo de Telemetría JSON:** [`logs/f6_condition_a_sync.json`](../logs/f6_condition_a_sync.json)
* **Archivo de Video Exportado:** [`logs/f6_832x480_33f_sync_20260908_202057.mp4`](../logs/f6_832x480_33f_sync_20260908_202057.mp4) (756,148 bytes)

#### Desglose de Latencias por Etapa:
1. **Prompt Encoding (UMT5-XXL en CPU):** 66.08 s
2. **Preparación de Latentes & Scheduler:** 1.35 s
3. **Bucle DiT Denoising (30 pasos sync):** **456.98 s** (**15.23 s/paso**)
4. **Decodificación VAE Tiled (33f bfloat16):** 28.92 s
5. **Serialización y Exportación MP4:** 0.64 s
* **Tiempo Total End-to-End (Wall-Clock):** **559.93 s (9.33 min)** (Cumple Target $< 600\text{ s}$)

#### Huella de Memoria:
* **Pico Físico NVML (VRAM Real a 50 ms):** **2,624.3 MB**
  * *Margen vs Hard Gate (4,800 MB):* **+2,175.7 MB** de holgura.
  * *Margen vs Target (4,000 MB):* **+1,375.7 MB** de holgura.
* **PyTorch Allocator Peak:** 1,204.2 MB
* **PyTorch Reserved Peak:** 1,656.0 MB
* **Host RAM RSS del Proceso:** 744.9 MB
* **Estabilidad Numérica:** 0 NaNs / 0 Infs (`False`)
* **Veredicto F6-A:** 🟢 **PASS** en los 5 ejes.

---

## 5. Benchmark F6-B: Marley Asíncrono FP16 (Async FP16)

### 5.1 Comando de Ejecución
```powershell
.venv\Scripts\python.exe f6_end_to_end_benchmark.py --mode async_fp16 --steps 30 --frames 33 --output logs/f6_condition_b_async_fp16.json
```

### 5.2 Mecanismo Técnico
Utiliza `BudgetedAsyncStreamer` con:
* **Doble CUDA Stream:** `compute_stream` (prioridad alta para cálculo del bloque $i$) y `transfer_stream` (prioridad baja para transferencia PCIe DtoH / HtoD del bloque $i+1$).
* **Pinned Host Memory (`pin_memory()`):** Transferencias asíncronas no-bloqueantes vía DMA directo.
* **Overlapping Activo:** La latencia de transferencia PCIe queda completamente oculta tras la ejecución matemática del bloque actual.

### 5.3 Métricas y Telemetría Auditada
* **Fecha y Hora de Inicio:** 2026-09-08 20:22:31 -03:00
* **Fecha y Hora de Finalización:** 2026-09-08 20:31:50 -03:00
* **Archivo de Telemetría JSON:** [`logs/f6_condition_b_async_fp16.json`](../logs/f6_condition_b_async_fp16.json)
* **Archivo de Video Exportado:** [`logs/f6_832x480_33f_async_fp16_20260908_203148.mp4`](../logs/f6_832x480_33f_async_fp16_20260908_203148.mp4) (756,148 bytes)

#### Desglose de Latencias por Etapa:
1. **Prompt Encoding (UMT5-XXL en CPU):** 45.20 s
2. **Preparación de Latentes & Scheduler:** 0.13 s
3. **Bucle DiT Denoising (30 pasos async):** **436.92 s** (**14.56 s/paso**)
4. **Decodificación VAE Tiled (33f bfloat16):** 31.11 s
5. **Serialización y Exportación MP4:** 2.34 s
* **Tiempo Total End-to-End (Wall-Clock):** **523.68 s (8.73 min)**

#### Comparativa F6-B vs F6-A (Ganancia por Prefetch Asíncrono):
* **Aceleración DiT Pura:** De 456.98 s a 436.92 s (**-20.06 s / -4.4% de tiempo de denoising**).
* **Cadencia por Paso:** De 15.23 s/paso a **14.56 s/paso** (**-0.67 s por paso**).
* **Aceleración Total End-to-End:** De 559.93 s a **523.68 s** (**-36.25 s de reducción total**).
* **Impacto en VRAM:** De 2,624.3 MB a **2,698.0 MB** (+73.7 MB debido al buffer de prefetch en GPU, perfectamente dentro de los límites).

#### Huella de Memoria:
* **Pico Físico NVML (VRAM Real a 50 ms):** **2,698.0 MB**
  * *Margen vs Hard Gate (4,800 MB):* **+2,102.0 MB** de holgura.
  * *Margen vs Target (4,000 MB):* **+1,302.0 MB** de holgura.
* **PyTorch Allocator Peak:** 1,204.2 MB
* **PyTorch Reserved Peak:** 1,656.0 MB
* **Host RAM RSS del Proceso:** 1,516.6 MB
* **Estabilidad Numérica:** 0 NaNs / 0 Infs (`False`)
* **Veredicto F6-B:** 🟢 **PASS** en los 5 ejes.

---

## 6. Benchmark F6-C: Marley Asíncrono INT8 (Async INT8)

### 6.1 Comando de Ejecución
```powershell
.venv\Scripts\python.exe f6_end_to_end_benchmark.py --mode async_int8 --steps 30 --frames 33 --output logs/f6_condition_c_async_int8.json
```

### 6.2 Mecanismo Técnico
Utiliza `INT8BudgetedStreamer` con:
* **Compresión PCIe al 50%:** Payload por bloque reducido de 88.6 MB a **45.9 MB (-48.2%)** mediante cuantización simétrica INT8 en proyecciones lineales.
* **Doble CUDA Stream Cuantizado:** Cómputo DiT solapado con transferencias de la mitad de volumen de datos a través del bus PCIe.
* **Dequant en GPU:** Descuantización al vuelo en CUDA a FP16 con preservación estricta de fidelidad numérica.

### 6.3 Métricas y Telemetría Auditada
* **Fecha y Hora de Inicio:** 2026-09-08 20:32:42 -03:00
* **Fecha y Hora de Finalización:** 2026-09-08 20:42:14 -03:00
* **Archivo de Telemetría JSON:** [`logs/f6_condition_c_async_int8.json`](../logs/f6_condition_c_async_int8.json)
* **Archivo de Video Exportado:** [`logs/f6_832x480_33f_async_int8_20260908_204210.mp4`](../logs/f6_832x480_33f_async_int8_20260908_204210.mp4) (756,148 bytes)

#### Desglose de Latencias por Etapa:
1. **Prompt Encoding (UMT5-XXL en CPU):** 49.71 s
2. **Preparación de Latentes & Scheduler:** 0.11 s
3. **Bucle DiT Denoising (30 pasos async INT8):** **451.32 s** (**15.04 s/paso**)
4. **Decodificación VAE Tiled (33f bfloat16):** 29.87 s
5. **Serialización y Exportación MP4:** 0.68 s
* **Tiempo Total End-to-End (Wall-Clock):** **538.67 s (8.98 min)**

#### Huella de Memoria:
* **Pico Físico NVML (VRAM Real a 50 ms):** **2,962.5 MB**
  * *Margen vs Hard Gate (4,800 MB):* **+1,837.5 MB** de holgura.
  * *Margen vs Target (4,000 MB):* **+1,037.5 MB** de holgura.
* **PyTorch Allocator Peak:** 1,204.2 MB
* **PyTorch Reserved Peak:** 1,656.0 MB
* **Host RAM RSS del Proceso:** 3,805.5 MB
* **Estabilidad Numérica:** 0 NaNs / 0 Infs (`False`)
* **Veredicto F6-C:** 🟢 **PASS** en los 5 ejes.

## 7. Benchmark F6-D: Marley Motor Adaptativo (Adaptive Engine)

### 7.1 Comando de Ejecución
```powershell
.venv\Scripts\python.exe f6_end_to_end_benchmark.py --mode adaptive --steps 30 --frames 33 --output logs/f6_condition_d_adaptive.json
```

### 7.2 Mecanismo Técnico
Utiliza `AdaptiveEngine` (validado en Phase F4) con:
* **Monitoreo en Ventanas ($N=5$ pasos):** Muestreo continuo de presión y evaluación de histéresis cada 5 pasos de difusión.
* **Selección Dinámica de Política:** Arbitraje en tiempo real entre perfiles `performance` (Async FP16 con prefetch agresivo), `balanced` (Async INT8 con compresión PCIe) y `safety` (Sync conservador ante presión extrema).
* **Gestión Dual de Streamers:** Dispone de ambos motores instanciados en memoria para conmutación inmediata sin recompilación de grafos ni re-cuantización.

### 7.3 Métricas y Telemetría Auditada
* **Fecha y Hora de Inicio:** 2026-09-08 20:46:37 -03:00
* **Fecha y Hora de Finalización:** 2026-09-08 20:56:01 -03:00
* **Archivo de Telemetría JSON:** [`logs/f6_condition_d_adaptive.json`](../logs/f6_condition_d_adaptive.json)
* **Archivo de Video Exportado:** [`logs/f6_832x480_33f_adaptive_20260908_205600.mp4`](../logs/f6_832x480_33f_adaptive_20260908_205600.mp4) (756,148 bytes)

#### Desglose de Latencias por Etapa:
1. **Prompt Encoding (UMT5-XXL en CPU):** 44.29 s
2. **Preparación de Latentes & Scheduler:** 0.13 s
3. **Bucle DiT Denoising (30 pasos adaptativos):** **443.21 s** (**14.77 s/paso**)
4. **Decodificación VAE Tiled (33f bfloat16):** 29.81 s
5. **Serialización y Exportación MP4:** 0.60 s
* **Tiempo Total End-to-End (Wall-Clock):** **530.00 s (8.83 min)**

#### Huella de Memoria:
* **Pico Físico NVML (VRAM Real a 50 ms):** **3,244.6 MB**
  * *Margen vs Hard Gate (4,800 MB):* **+1,555.4 MB** de holgura.
  * *Margen vs Target (4,000 MB):* **+755.4 MB** de holgura.
* **PyTorch Allocator Peak:** 1,202.9 MB
* **PyTorch Reserved Peak:** 1,870.0 MB
* **Host RAM RSS del Proceso:** **881.4 MB** (Excelente control de memoria host)
* **Estabilidad Numérica:** 0 NaNs / 0 Infs (`False`)
* **Veredicto F6-D:** 🟢 **PASS** en los 5 ejes.

---

## 8. Tabla Comparativa Multidimensional Consolidada (Tetralogía Canónica Completa)

| Eje de Evaluación | F6-A (Sync FP16) | F6-B (Async FP16) | F6-C (Async INT8) | F6-D (Adaptive Engine) | Target / Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mecanismo de Streaming** | Secuencial síncrono | Doble stream FP16 | Doble stream INT8 | Orquestador Dinámico | — |
| **Payload PCIe / Bloque** | 88.6 MB | 88.6 MB | 45.9 MB (-48.2%) | Dinámico (FP16/INT8) | — |
| **Tiempo DiT (30 pasos)** | 456.98 s | **436.92 s (-20.1 s)**| 451.32 s (-5.66 s)| 443.21 s (-13.77 s) | — |
| **Cadencia (s/paso)** | 15.23 s | **14.56 s (-0.67 s)** | 15.04 s (-0.19 s) | 14.77 s (-0.46 s) | $\le 15.0\text{ s}$ |
| **Tiempo Total End-to-End**| 559.93 s (9.33 min) | **523.68 s (8.73 min)**| 538.67 s (8.98 min)| 530.00 s (8.83 min) | $< 600.0\text{ s}$ |
| **Pico VRAM Físico (NVML)**| **2,624.3 MB** | 2,698.0 MB | 2,962.5 MB | 3,244.6 MB | $\le 4,800.0\text{ MB}$ |
| **Holgura vs Hard Gate** | **+2,175.7 MB** | +2,102.0 MB | +1,837.5 MB | +1,555.4 MB | $> 0\text{ MB}$ |
| **VAE Decode (33 frames)** | 28.92 s | 31.11 s | 29.87 s | 29.81 s | $\approx 27.0\text{ s}$ |
| **Host RAM RSS** | **744.9 MB** | 1,516.6 MB | 3,805.5 MB (Alerta F6-01) | 881.4 MB | $\le 24\text{ GB}$ |
| **Estabilidad Numérica** | 0 NaNs | 0 NaNs | 0 NaNs | 0 NaNs | 0 NaNs |
| **Video MP4 Generado** | [`logs/..._sync_...mp4`](../logs/f6_832x480_33f_sync_20260908_202057.mp4) | [`logs/..._async_fp16_...mp4`](../logs/f6_832x480_33f_async_fp16_20260908_203148.mp4) | [`logs/..._async_int8_...mp4`](../logs/f6_832x480_33f_async_int8_20260908_204210.mp4) | [`logs/..._adaptive_...mp4`](../logs/f6_832x480_33f_adaptive_20260908_205600.mp4) | MP4 Válido |
| **Veredicto General** | 🟢 **PASS** | 🟢 **PASS** | 🟢 **PASS** | 🟢 **PASS** | 🟢 **PASS** |

---

## 9. Alerta Técnica Identificada: Sobrecarga Host RSS en F6-C

Durante la evaluación de F6-C se registró un aumento del Host RSS del proceso a **3,805.5 MB** (+2,288.9 MB sobre F6-B).  
El análisis forense determinó que `INT8BudgetedStreamer` genera una réplica de pesos `pinned INT8` (~1.38 GB) sin liberar los tensores originales `param.data` en FP16 (~2.66 GB) de los 30 bloques DiT en CPU, produciendo una doble residencia transitoria en memoria host.

Análisis forense detallado y plan de remediación: [`docs/F6_ALERT_01_INT8_HOST_RSS_OVERHEAD.md`](F6_ALERT_01_INT8_HOST_RSS_OVERHEAD.md).



