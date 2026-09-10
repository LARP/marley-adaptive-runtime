# PROTOCOLO PRE-REGISTRADO DE LA CAMPAÑA F7-720p (REEVALUACIÓN Y ATRIBUCIÓN)

**Documento:** `F7_720P_PREREGISTERED_PROTOCOL_01.md`  
**Fecha de Congelación:** 10 de septiembre de 2026  
**Autor:** Director de Proyecto — Marley Runtime  
**Revisor / Autoridad:** Consejero Técnico Externo  
**Estado:** **PRE-REGISTRADO Y CONGELADO ANTES DE EJECUCIÓN**  
**Referencias:** Dictamen Final del Consejero (knowledge acquired through conversation with an AI agent); Telemetría F7-D6 (`logs/f7_d6_attribution_probe_telemetry.json`).  

---

## 1. Declaración de Misión y Propósito

El objetivo de la **Campaña F7-720p** es resolver de manera empírica, replicable y causal la viabilidad de inferencia a resolución de alta definición (**1280x720 @ 33 frames**) sobre el hardware objetivo (**NVIDIA GeForce RTX 3050 6GB Laptop GPU**, 6,144 MB físicos bajo Windows 11 WDDM).

La campaña responde a dos preguntas científicas desacopladas:
1. **Pregunta de Certificación de Seguridad Física (Capa A):** ¿Puede el sistema completo ejecutar el workload objetivo sin superar el límite de **4,800.0 MB** de VRAM física en ningún instante?
2. **Pregunta de Eficiencia Algorítmica y Atribución Causal (Capas B, C y D):** ¿Cuál es el incremento neto inducido por el runtime ($S2 - S0$), cuánta memoria administra el allocator de PyTorch y qué componente causa la retención residual observada post-workload?

---

## 2. Taxonomía de Medición de Cuatro Capas (Modelo Autorizado)

Se medirá de forma continua a **1 Hz** el balance de masa de memoria a través de cuatro capas independientes:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  CAPA A: Residencia Física Global del Dispositivo (NVML used)           │
│  Límite Oficial de Seguridad: Peak <= 4,800.0 MB                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│ CAPA B: Incremento Neto del Workload│   │ CAPA D: Residencia Posterior (S3/S4)│
│ Δ_S0 = S2 - S0 (Cota de Impacto)    │   │ Meseta post-workload tras release   │
│ Δ_S1 = S2 - S1 (Incremento Operac.) │   │ Intervención inter-proceso          │
└──────────────────┬──────────────────┘   └─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  CAPA C: Allocator de PyTorch (torch.cuda)                             │
│  Peak Reserved (Bloques retenidos) | Peak Allocated (Tensores vivos)   │
└────────────────────────────────────────────────────────────────────────┘
```

* **Capa A — Residencia Física del Dispositivo:** Lectura de `nvmlDeviceGetMemoryInfo().used`. Captura la carga total sobre el hardware (incluyendo Windows DWM, aplicaciones del usuario, driver y Marley).
* **Capa B — Incremento Neto Asociado al Workload:** 
  - $\Delta_{S0} = S2 - S0$ (Impacto adicional de Marley sobre el baseline previo al contexto).
  - $\Delta_{S1} = S2 - S1$ (Impacto adicional sobre el estado con modelo cargado).
* **Capa C — Allocator PyTorch:** `torch.cuda.max_memory_reserved()` y `torch.cuda.max_memory_allocated()`. Mide la huella interna de tensores y pools de bloques de PyTorch.
* **Capa D — Residencia Posterior de Sesión:** Muestreo en $S3$ (120 s) y $S4$ (60 s). Mide la meseta de retención tras invocar `empty_cache()`.

---

## 3. Matriz Formal de Certificación

De acuerdo con la resolución vinculante del Consejero Técnico Externo, la evaluación de la campaña no se basará en un juicio único, sino en una **matriz formal de dos compuertas**:

| Resultado | Gate A (Seguridad Física $\le 4,800\text{ MB}$) | Gate B (Eficiencia Incremental $\Delta_{S0} \le 3,850\text{ MB}$) | Clasificación Oficial | Consecuencia en Gobernanza |
| :---: | :---: | :---: | :--- | :--- |
| **Escenario 1** | 🟢 **PASS** | 🟢 **PASS** | **🟢 720p CERTIFIED FULL PASS** | Integración en `pipeline.py` autorizada. Hito 720p cerrado. |
| **Escenario 2** | 🟢 **PASS** | 🔴 **FAIL** | **🟡 FUNCTIONAL / INEFFICIENT** | Funciona físicamente, pero con sobrecosto algorítmico. Revisión requerida. |
| **Escenario 3** | 🔴 **FAIL** | 🟢 **PASS** | **🔴 GATE A FAIL / 🟢 B FAVORABLE** | Algoritmo eficiente, pero el entorno del sistema viola la seguridad física. Integración NO autorizada. |
| **Escenario 4** | 🔴 **FAIL** | 🔴 **FAIL** | **🔴 COMPLETE REJECTION** | Fallo total en ambas dimensiones. Algoritmo no viable. |

> **REGLA DE ORO DE GOBERNANZA:**  
> Ningún resultado en la Capa B puede sustituir o anular un fallo en la Gate A oficial. Si el pico del dispositivo supera 4,800.0 MB, la corrida es **FAIL de certificación**, sin excepciones ni reinterpretaciones retroactivas.

---

## 4. Estructura Experimental de la Campaña (Tres Etapas)

La campaña se ejecutará en tres etapas secuenciales estrictamente controladas:

```text
FASE 1: ETAPA C (INTERVENCIÓN INTER-PROCESO)
  - Medir causa raíz de los ~438 MB
  - Monitor externo B (1 Hz) vigila mientras Proceso A se ejecuta y MUERE
       ↓
FASE 2: ETAPA A (REPRODUCCIÓN CONTROLADA)
  - 3 pasos DiT idénticos a F7-D6 para verificar reproducibilidad
  - Control A/B <= 10%, S0 estable <= 150 MB
       ↓
FASE 3: ETAPA B (REEVALUACIÓN 720p COMPLETA)
  - 30 pasos DiT completos a 720p / 33 frames
  - Modo adaptive con liberación seam cond→uncond y drenaje step-start
  - Evaluación contra Gate A (4,800 MB) y Gate B (3,850 MB)
```

---

## 5. Especificación Detallada de la Etapa C: Intervención Causal Inter-Proceso

Esta etapa determina de forma experimental qué provoca la retención de los ~438 MB post-workload.

### 5.1. Arquitectura de Dos Procesos
1. **Proceso Monitor (B):** Proceso ligero en Python inerte, sin contexto CUDA, ejecutado en segundo plano. Registra telemétricamente cada 1 segundo:
   - NVML device-wide `used`, `free`, `total`.
   - Lista de procesos activos en GPU vía NVML (`Compute` y `Graphics`).
   - Timestamps de alta precisión.
2. **Proceso de Carga (A):**
   - Ejecuta el protocolo validado: $S0$ (30 s) $\rightarrow$ CUDA init + warm-up (10 MB) $\rightarrow$ Control A (500 MB) $\rightarrow$ Pipeline init $\rightarrow$ $S1$ $\rightarrow$ 3 pasos DiT $\rightarrow$ `empty_cache()` $\rightarrow$ $S3$ (30 s, confirmación de meseta de ~1,480 MB).
   - En el segundo 30 de $S3$, llama inmediatamente a `os._exit(0)` / `sys.exit(0)`, forzando la terminación inmediata del proceso y la destrucción de todos los handles del sistema operativo y del contexto CUDA.
3. **Monitoreo Post-Muerte:** El Proceso Monitor (B) continúa registrando en silencio durante **60 segundos posteriores a la muerte confirmada del Proceso A**.

### 5.2. Criterios de Falsación e Interpretación Pre-Registrados

| Resultado Observado en Monitor B tras muerte de A | Conclusión Causal Pre-Registrada |
| :--- | :--- |
| **H1: Caída inmediata a $S0$ ($\le S0 + 50\text{ MB}$ en $\le 5\text{ s}$)** | **Residencia dependiente del ciclo de vida del proceso.** Se prueba empíricamente que los ~438 MB pertenecían al contexto CUDA y recursos del proceso, liberados limpiamente por el kernel al terminar. |
| **H2: Caída diferida hacia $S0$ (tarda entre 5 y 60 segundos)** | **Liberación diferida del sistema operativo.** Se prueba que WDDM aplica una recolección de páginas retrasada post-mortem. |
| **H3: Permanencia de la meseta ($\approx 1,480\text{ MB}$ tras 60 s sin proceso A)** | **Residencia externa al proceso.** La retención sobrevive a la destrucción del contexto; evidencia de persistencia en el subsistema gráfico global o DWM. |

---

## 6. Especificación Detallada de la Etapa B: Reevaluación 720p (30 Pasos)

### 6.1. Workload Pre-Registrado Inmutable
* **Modelo:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
* **Resolución:** **1280 x 720** (720p)
* **Frames:** **33 frames** (~2.06 segundos a 16 fps)
* **Pasos de Difusión:** **30 pasos DiT completos**
* **Modo de Ejecución:** `adaptive` con:
  - Liberación forzada en costura `cond → uncond` (`empty_cache()` quirúrgico).
  - Drenaje al inicio de cada paso de difusión.
  - Offload secuencial de bloques DiT en CPU.
* **Precisión:** DiT en FP16 (`torch.float16`); VAE y Text Encoder en BF16 (`torch.bfloat16`).
* **VAE:** Decodificación espacialmente tileada (256x256 tiles) con decodificación al final del denoise.
* **Prompt Canónico:**
  > *"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."*
* **Negative Prompt:** *"blurry, low quality, watermark, deformed"*
* **Seed:** `42` (Determinista)
* **Guidance Scale:** `5.0`

### 6.2. Protocolo de Ejecución
```text
S0 (60 s @ 1Hz)
   ↓
torch.cuda.init() + warm-up 10 MB + empty_cache()
   ↓
Control A (~500 MB) + empty_cache()
   ↓
Carga de Pipeline Wan2.1
   ↓
S1 operational baseline (post-init)
   ↓
Wan2.1 Denoise (30 pasos completos con telemetría per-step)
   ↓
VAE Decode (33 frames @ 720p tileado)
   ↓
empty_cache()
   ↓
S3 post-workload (120 s @ 1Hz)
   ↓
Control B (~500 MB) + empty_cache()
   ↓
S4 final observation (60 s @ 1Hz)
```

---

## 7. Criterios de Aborto y Seguridad Operacional

Para proteger la integridad del hardware y prevenir congelamientos del sistema:
1. **Kill-Switch de Emergencia:** Si en cualquier instante de la difusión o decodificación la lectura de NVML físico alcanza **5,800.0 MB** (umbral de seguridad crítica a 344 MB del límite físico de 6,144 MB), el runner abortará inmediatamente el proceso (`SAFE_ABORT_TRIGGERED`), liberará memoria y registrará la telemetría hasta el fallo.
2. **NaN Guard:** Si el modelo genera tensores con `NaN` o `Inf`, la corrida se detendrá y se marcará como fallo numérico.

---

## 8. Gobernanza y Estado del Documento

* Este protocolo queda **completamente congelado** bajo el commit correspondiente.
* **F7-D5 permanece formalmente como NEGATIVE** hasta que la Etapa B de este protocolo arroje sus resultados oficiales.
* Cualquier discrepancia entre lo observado y las hipótesis pre-registradas será reportada fielmente sin alteraciones retrospectivas de los umbrales.

---

**ESTADO: PROTOCOLO PRE-REGISTRADO Y CONGELADO — LISTO PARA IMPLEMENTACIÓN DEL RUNNER**
