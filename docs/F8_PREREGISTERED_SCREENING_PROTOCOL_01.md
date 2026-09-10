# PROTOCOLO PRE-REGISTRADO OFICIAL DE F8-SCREENING (A/B PAREADO DE 10 PASOS)

**Documento:** `F8_PREREGISTERED_SCREENING_PROTOCOL_01.md`  
**Fecha de Congelación:** 10 de septiembre de 2026  
**Autor:** Director de Proyecto — Marley Runtime  
**Autoridad / Revisor:** Consejero Técnico Externo  
**Estado:** **PRE-REGISTRADO Y CONGELADO ANTES DE EJECUCIÓN**  
**Referencias:** Dictamen de Aprobación del Consejero (`F8_CONSULTANT_VERDICT_SCREENING_01.md`); Carta de Justificación de la Dirección (`F8_DIRECTOR_JUSTIFICATION_LETTER_TO_CONSULTANT_01.md`).  

---

## 1. Declaración de Misión y Pregunta Científica Central

El objetivo de la **Fase F8-Screening** es determinar empíricamente si la cuantización selectiva de las matrices de proyección lineal de los 30 bloques DiT (Wan2.1-T2V-1.3B) reduce la presión de memoria física en el dispositivo de forma suficiente como para eliminar o atenuar drásticamente el sobrepico localizado de **~196 MB** observado en el Paso 6 durante F7-D5.

### Pregunta Falsable Pre-Registrada:
> *«¿La cuantización selectiva de las proyecciones DiT en precisión reducida produce una reducción reproducible del pico físico NVML de al menos 196.0 MB respecto al Control A durante una ejecución a 720p de 10 pasos, manteniendo la cadencia en $\le 65.0\text{ s/paso}$ y sin divergencia numérica significativa en los tensores latentes?»*

---

## 2. Matriz Cuantitativa de Decisión Vinculante

El resultado del screening gobernará de forma automática e inapelable el siguiente paso del proyecto según cuatro umbrales mutuamente excluyentes fijados antes de correr la prueba:

| Nivel de Resultado | Reducción del Pico NVML ($\Delta_{\text{Peak}} = \text{Peak}_A - \text{Peak}_B$) | Decisión de Gobernanza Inmutable |
| :---: | :---: | :--- |
| 🟢 **Resultado Fuerte** | $\Delta_{\text{Peak}} \ge \mathbf{196.0\text{ MB}}$ | **GO:** Se autoriza de inmediato la validación completa de 30 pasos (F8 Stage B) para certificación 720p. |
| 🟡 **Resultado Prometedor** | $\mathbf{100.0\text{ MB}} \le \Delta_{\text{Peak}} < \mathbf{196.0\text{ MB}}$ | **GO CONDICIONAL:** Se justifica estudiar una segunda micro-intervención complementaria antes de correr 30 pasos. |
| 🔴 **Resultado Débil** | $\Delta_{\text{Peak}} < \mathbf{50.0\text{ MB}}$ | **NO-GO:** Se descarta la cuantización como solución viable para 720p en 6 GB; no se amplía la intervención. |
| 🔴 **Resultado Nulo** | $\Delta_{\text{Peak}} \approx \mathbf{0.0\text{ MB}}$ ($\pm 25\text{ MB}$) | **CIERRE DEFINITIVO:** Cierre formal e irrevocable de la línea 720p en hardware de 6 GB; 480p ratificado como único estándar. |

---

## 3. Arquitectura Metrológica Obligatoria: Dos Procesos Limpios Desacoplados

En estricto cumplimiento de la resolución del Consejero, **se prohíbe taxativamente ejecutar ambas condiciones en el mismo proceso de Python**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 1: CONDICIÓN A (CONTROL FP16 LIMPIO)                             │
│  - Proceso A nuevo (PID_A)                                             │
│  - Muestreo S0_A (30s @ 1Hz) -> Inferencia 10 pasos FP16               │
│  - Terminación forzada con os._exit(0)                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 2: ENFRIAMIENTO Y REPOSO DEL SISTEMA OPERATIVO                   │
│  - Ventana obligatoria de 60 segundos de reposo                        │
│  - Recolección pasiva de páginas y recuperación de estado basal WDDM   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 3: CONDICIÓN B (TRATAMIENTO CUANTIZADO LIMPIO)                   │
│  - Proceso B nuevo (PID_B) sin herencia de memoria                     │
│  - Muestreo S0_B (30s @ 1Hz) -> Inferencia 10 pasos con Cuantización   │
│  - Comparación objetiva A vs B y cálculo de similitud coseno           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Especificación Técnica de las Condiciones Experimentales

### 4.1. Workload Compartido e Inmutable
* **Modelo Base:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
* **Resolución:** **1280 x 720** (720p)
* **Frames:** **33 frames**
* **Pasos DiT:** **10 pasos exactos** (capturando la ventana crítica de los pasos 5 a 7)
* **Prompt Canónico:** *"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."*
* **Negative Prompt:** *"blurry, low quality, watermark, deformed"*
* **Seed:** `42` (Determinista)
* **Guidance Scale:** `5.0`
* **Scheduler:** Flow-Match / Euler idéntico
* **Intervención Base de Memoria:** Modo `adaptive` con vaciado en costura `cond → uncond` y drenaje de arranque de paso (`empty_cache()`).
* **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,144 MB físicos, Windows 11 WDDM).

### 4.2. Condición A (Control Limpio — Línea Base F7)
- **Precisión:** Bloques DiT en FP16 nativo (`torch.float16`).
- **Offload:** Secuencial en CPU, transferencia por bloque a GPU.
- **Objetivo Metrológico:** Reproducir el comportamiento del Paso 6 observado en F7.

### 4.3. Condición B (Tratamiento Limpio — Proyecciones DiT Cuantizadas)
- **Precisión:** Proyecciones lineales de los 30 bloques DiT (`q, k, v, out` y proyecciones FFN) en precisión reducida validada en **Fase F1.7**.
- **Salvaguarda de Buffers:** La de-cuantización / cómputo se realiza de forma compacta por bloque en GPU, sin crear copias duplicadas completas en VRAM.
- **Métrica de Fidelidad:** Se extraen los tensores latentes resultantes en el Paso 10 y se calcula la similitud coseno respecto al Control A:
  $$\text{Cosine Similarity} = \frac{\langle \mathbf{z}_A, \mathbf{z}_B \rangle}{\|\mathbf{z}_A\|_2 \|\mathbf{z}_B\|_2}$$
  - **Criterio de Tolerancia Numérica:** Similitud coseno $\ge \mathbf{0.9900}$ (cero NaNs / Infs).

---

## 5. Instrumentación y Telemetría por Paso

En ambas condiciones se registrará de forma continua y sincronizada:
1. **Física del Dispositivo:** NVML instantáneo a 20 ms de intervalo; detección de `Peak NVML` absoluto.
2. **Allocator de PyTorch:** Muestreo exacto al final de cada paso ($t \in [1, 10]$) de:
   - `Allocated (MB)`: Memoria de tensores vivos.
   - `Reserved (MB)`: Memoria total del pool de PyTorch.
3. **Métricas de Rendimiento:**
   - Tiempo de ejecución de cada paso ($s/\text{step}$).
   - Límite máximo de cadencia: media $\le \mathbf{65.0\text{ s/paso}}$.
4. **Kill-Switch de Emergencia:** Aborto automático si en cualquier condición la VRAM física supera **5,800.0 MB**.

---

## 6. Salidas y Artefactos de la Prueba

1. **Telemetría Cruda JSON:** `logs/f8_screening_ab_telemetry.json` (incluyendo la serie temporal completa de A y B).
2. **Reporte Formal:** `docs/F8_SCREENING_AB_REPORT_01.md` con la tabla comparativa paso a paso y el veredicto automático de la matriz.

---

**ESTADO: PROTOCOLO PRERREGISTRADO OFICIAL — CONGELADO Y LISTO PARA RUNNER**
