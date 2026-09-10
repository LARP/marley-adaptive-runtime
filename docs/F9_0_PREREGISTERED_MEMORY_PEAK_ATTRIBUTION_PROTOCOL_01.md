# PROTOCOLO PREREGISTRADO OFICIAL — F9-0
## Preregistered Memory Peak Attribution Diagnostic

**Documento:** `F9_0_PREREGISTERED_MEMORY_PEAK_ATTRIBUTION_PROTOCOL_01.md`
**Fecha de congelación:** 10 de septiembre de 2026
**Autor:** Dirección del Proyecto — Marley Runtime
**Autoridad externa:** Consejero Técnico — Runtime y Gestión de Memoria
**Estado:** 🟡 **F9-0 — PREREGISTERED / FROZEN — PENDIENTE DE AUTORIZACIÓN DE EJECUCIÓN**
**Naturaleza:** **Diagnóstico, no optimización.** No consume slot de validación Wan. No reabre F7 ni F8.
**Referencias:** Carta F9 del Consejero (`docs/private/F9_CONSULTANT_LETTER_GENERATION_01.md`); MSG de generación/congelamiento F9-0 (`docs/private/F9_CONSULTANT_MSG_PREREGISTRATION_01.md`); Charter F9 (`F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md`); **F7-D5 oficial** (`F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md` + `logs/f7_d5_full_30step_validation_telemetry.json`); F7-D6 (`F7_D6_METRIC_ATTRIBUTION_REPORT_01.md`); ROADMAP §2.

> Este preregistro incorpora las **ocho observaciones críticas** de la Dirección, aceptadas por el
> Consejero, convirtiendo sus condiciones en **definiciones operacionales reproducibles** (§2).
> No altera la arquitectura de F9 ni contradice la aprobación del Consejero Externo.

---

## 1. Misión y pregunta falsable

Determinar cuantitativamente qué componentes explican el **pico físico** observado en
**Wan2.1-T2V-1.3B — 1280×720 — 33 frames — 30 steps**, tomando **F7-D5** como referencia
histórica única y congelada (§2.7), y separar: pesos, activaciones, workspace de kernels, memoria
del allocator, fragmentación, CUDA/Driver/Runtime Overhead y otros/transitorios.

**Pregunta falsable preregistrada:**

> *¿Es posible explicar de forma reproducible al menos el **90 %** del presupuesto atribuible
> (definido en §2.1) del pico físico NVML de una corrida canónica 1280×720/33f/30steps mediante
> las siete categorías mutuamente excluyentes de §2.2, con una instrumentación NVML de resolución
> suficiente (≥10 Hz) y GPU sin otros contextos de cómputo?*

---

## 2. Definiciones operacionales (ocho resoluciones)

### 2.1 Denominadores de atribución (obs. #1)

Se reportarán **tres cantidades separadas** y **no se sumarán entre sí**:

| Símbolo | Definición | Rol |
| :--- | :--- | :--- |
| `Peak_NVML_raw` | Máximo de `nvmlDeviceGetMemoryInfo().used` (device-wide) durante la corrida. | Métrica física primaria / gate 4,800 MB. |
| `S0_idle` | Media estable del baseline idle **antes** de inicializar contexto CUDA (ventana de 60 s, 1 Hz). | Residencia OS/WDDM/escritorio **no atribuible** a Marley. |
| `Delta_induced` | `Peak_NVML_raw − S0_idle`. | **Presupuesto atribuible**: **pico inducido por el workload**. Device-wide growth; NO propiedad por proceso. |

**Denominador de la atribución del 90 %:** el **pico inducido por el workload** = `Peak NVML − S0`
= `Delta_induced` (§2.1). Esto es obligatorio porque F7-D6 demostró que `Peak_NVML_raw` incluye un
baseline idle de ~962 MB no atribuible a Marley; atribuir el 90 % del pico crudo es inalcanzable por
construcción. El **baseline idle de WDDM/DWM no se atribuye** al workload Marley: se reporta aparte
como `S0_idle`.

**Reporte obligatorio adicional (transparencia):** la fracción explicada de `Peak_NVML_raw` se
reportará siempre como métrica secundaria, sin sustituir al criterio de cierre.

**Coherencia con F7-D6:** se reutiliza la distinción de dos capas (A: device-wide raw; B:
incremental inducido) ya caracterizada por F7-D6. `usedGpuMemory = None` bajo WDDM significa
**contador no disponible**, no 0 MB. La atribución es un **modelo de descomposición**, no una
prueba de propiedad física por proceso.

### 2.2 Taxonomía mutuamente excluyente (obs. #2)

Las siete categorías forman una **partición jerárquica** de `Delta_induced`. **Cada byte se asigna
a exactamente una categoría**; se prohíbe el doble conteo.

| # | Categoría | Definición operacional | Método de medición |
| :-: | :--- | :--- | :--- |
| 1 | **Pesos residentes** | Bytes de parámetros presentes en GPU en un instante dado (streaming por bloque: no son el total del modelo). | Snapshots de `allocated` + modelo estático del schedule de streaming. |
| 2 | **Activaciones** | Tensores vivos no-peso en el instante de pico (latentes, embeds, buffers intermedios). | `allocated` menos pesos residentes; snapshots sincronizados. |
| 3 | **Workspace de kernels** | Memoria transitoria reservada por kernels (cuDNN/cuBLAS/aten), no visible como tensores. | Deltas NVML acotados por eventos CUDA en el instante del evento; ventana temporal. |
| 4 | **Memoria del allocator** | `reserved − allocated` (pool retenido pero no vivo). | `torch.cuda.memory_reserved − memory_allocated`. |
| 5 | **Fragmentación** | **Sub-slice** de la categoría 4 no reutilizable por fragmentación de layout (p. ej. `inactive_split`). **No es aditiva** respecto de 4. | `memory_stats()` del allocator (`inactive_split`, bloques no divididos). |
| 6 | **CUDA/Driver/Runtime Overhead** | `S1_operational − S0_idle`, medido **tras inicializar el contexto CUDA y lanzar el primer kernel, antes de cargar el modelo** (§2.4). | Instrumentación pre-carga (§2.4). |
| 7 | **Otros / transitorios** | Residuo no asignable a 1–6. | `Delta_induced − Σ(1..6)`. |

**Reglas de no-solapamiento:**

* Categoría **5 ⊂ 4** (la fragmentación es una porción del pool retenido, no un sumando extra).
* Categoría **3** solo puede reclamar bytes **no** ya contabilizados en 1, 2 ni 4, acotados por la
  ventana temporal del evento.
* Ante ambigüedad de asignación, el byte va a **7 (otros)**; nunca se reparte entre categorías.
* La suma `Σ(1..6)` puede ser menor que `Delta_induced`; la diferencia es exactamente 7.

### 2.3 Exclusividad de GPU — definición operacional (obs. #3)

En un portátil con display, DWM/WDDM **siempre** retiene memoria; exigir "cero residente" invalidaría
todas las corridas. Por tanto:

* **`exclusive = TRUE`** si y solo si:
  1. **No hay otro proceso con contexto CUDA de cómputo** (enumeración NVML
     `nvmlDeviceGetComputeRunningProcesses` / `GraphicsRunningProcesses`, excluyendo el PID propio); y
  2. El baseline idle `S0` es **estable**: `spread ≤ 150 MB` durante la ventana de 60 s.
* El baseline de escritorio/DWM **no invalida** la corrida: se **cuantifica** como `S0_idle` y se
  reporta.
* Si `exclusive = FALSE`: la corrida se conserva como **evidencia exploratoria** y **no cuenta**
  como corrida válida para la atribución final.
* Se registran temperatura, clocks y utilización durante toda la corrida (instrumento F7-D6).

### 2.4 Lower Bounds reproducibles (obs. #4)

Se reportan **tres cantidades distintas**, todas tratadas como **cotas analíticas/observadas
defendibles**, y **ninguna** como "mínimo físico absoluto":

**(a) Lower Bound Analítico (`LB_analitico`)** — memoria mínima estimada de los tensores vivos
necesarios para el grafo bajo las reglas del schedule. Se define como:

```text
LB_analitico = pesos_persistentes
             + max_i ( pesos_bloque_i )                      # streaming por bloque
             + max( conjunto_activaciones_simultaneas )      # punto más estrecho del schedule
```

* `pesos_persistentes`: buffers de modelo siempre residentes (patch_embedding, condition_embedder,
  norm_out, proj_out, buffers de tiling VAE…), medidos por snapshots.
* El término de activaciones se calcula **analíticamente** a partir de las formas del grafo.
* Se documenta la fórmula y todos sus inputs. Es una **estimación**, marcada como tal.

**(b) Minimum Allocated observado (`Min_allocated_obs`)** — mínimo de `torch.cuda.memory_allocated`
medido bajo el pipeline en puntos CUDA sincronizados (el instante de menor residencia de tensores
vivos observado). Es una **observación**, no una cota teórica.

**(c) Lower Bound Operativo (`LB_operativo`)** — referencia relevante para el gate de 4.8 GB:

```text
LB_operativo = LB_analitico + Overhead_CUDA/Driver/Runtime
```

donde `Overhead_CUDA/Driver/Runtime = S1_operational − S0_idle`, medido **después de inicializar el
contexto CUDA y lanzar el primer kernel, pero antes de que los datos del modelo estén residentes**
(§2.4, obs. #2/obs. #6).

**Contraste obligatorio:** se comparan `LB_analitico` y `Min_allocated_obs`; una divergencia grande
se documenta como incertidumbre del modelo analítico. Ninguno se presenta como mínimo físico
absoluto.

### 2.5 Resolución temporal e instrumentación (obs. #5)

* **Frecuencia NVML primaria: 20 ms (50 Hz).** Mínimo aceptable: **≥10 Hz**. Se registrará la
  **frecuencia real alcanzada**, el **coste del polling** y un test de **efecto observador**
  (comparación de pico con y sin muestreo a alta frecuencia).
* **Puntos de sincronización CUDA** para acotar eventos candidatos: costura `cond → uncond`,
  atención, concatenaciones, upcasts de precisión y creación/liberación de grandes buffers.
* Se registra la **ventana temporal del pico** (timestamp inicio/fin) y la serie completa.
* **Clasificación de resolución:** si la resolución obtenida no permite confiar en que el máximo
  observado corresponde al máximo real, el resultado se clasificará como
  **`POSIBLE SUBESTIMACIÓN DEL PICO`** y F9-0 **no podrá cerrarse** solo por alcanzar el 90 %
  nominal.

### 2.6 Régimen experimental, VAE y reutilización de F7-D6 (obs. #6)

* **VAE fuera de la atribución primaria:** F9-0 atribuye el pico del **bucle DiT** (déficit conocido).
  El pico del VAE (~2 GB) es un **evento separado** y se reporta aparte, sin mezclarse.
* **Reutilización del instrumento F7-D6** (validado como `INSTRUMENT VALIDATED`): baseline `S0`,
  `S1`, controles A/B de ~500 MB, `cudaMemGetInfo`, temperatura/clocks/utilización.
* **N ≥ 5 corridas independientes** del workload canónico, con **cooldown** entre corridas y orden
  de ejecución registrado. Se reportan mediana, dispersión e intervalo de confianza.
* **Campaña:** 30 steps 720p ≈ 44 min/corrida ⇒ >4 h para 5 corridas + cooldowns. La Dirección
  autoriza explícitamente el inicio de la campaña antes de lanzarla.

### 2.7 Referencia histórica única y verificada (obs. #7)

**Verificación realizada (2026-09-10) contra los artefactos oficiales:**

| Artefacto | Peak NVML oficial | Déficit vs gate 4,800 MB | Uso en F9-0 |
| :--- | :---: | :---: | :--- |
| **F7-D5** — `logs/f7_d5_full_30step_validation_telemetry.json` + `docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md` | **5,096.1 MB** | **+296.1 MB** | ✅ **Referencia única congelada de F9-0** |
| F7 Stage B — `logs/f7_stage_b_full_validation_telemetry.json` + `docs/F7_STAGE_B_FULL_VALIDATION_REPORT_01.md` | 4,996.0 MB | +196.0 MB | ❌ Artefacto **distinto**; **NO** se usa como referencia F7-D5 |

* **Referencia F9-0 congelada:** **F7-D5 = 5,096.1 MB** (`peak_nvml_used`), déficit **+296.1 MB**
  respecto del gate. Vinculada explícitamente a su telemetría y reporte oficiales.
* **Corrección de atribución:** el valor **4,996.0 MB / +196.0 MB** pertenece a **F7 Stage B**, una
  re-evaluación posterior e independiente, **no** a F7-D5. Se prohíbe usar simultáneamente ambos
  valores: F9-0 usa **solo** la referencia F7-D5 (5,096.1 MB).
* El déficit histórico se calcula **exclusivamente** a partir del valor oficial congelado:
  `5,096.1 − 4,800.0 = +296.1 MB`.
* F7-D5 y F7 Stage B **no se reclassifican**; son referencias inmutables.
* Gate duro vigente: **4,800.0 MB** (device-wide NVML). No se modifica.

### 2.8 Estado de F9-1…F9-4 (obs. #8)

F9-1, F9-2, F9-3 y F9-4 se declaran **placeholders estructurales inactivos**. No se redactan ni
congelan hasta el cierre de F9-0. Su contenido dependerá del mecanismo dominante identificado.

---

## 3. Workload canónico (inmutable durante F9-0)

* **Modelo:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
* **Resolución:** 1280 × 720 · **Frames:** 33 · **Steps:** 30
* **Prompt canónico:** *"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."*
* **Negative prompt:** *"blurry, low quality, watermark, deformed"*
* **Seed:** 42 · **Guidance:** 5.0 · **Scheduler:** Flow-Match/Euler (idéntico a F7).
* **Modo:** `adaptive` con vaciado en costura `cond → uncond` y drenaje de arranque de paso (config
  FAVORABLE de F7-D4/D5).
* **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Windows 11 WDDM).
* **Prohibido modificar** entre iteraciones: modelo, workload, resolución, frames, steps, seed y
  parámetros de ejecución.

---

## 4. Protocolo de ejecución

```text
[Por corrida i = 1..5]
  S0  baseline idle (60 s @ 1 Hz)  -> verifica exclusividad §2.3
  -> init contexto CUDA + primer kernel (warm-up)
  -> Control A (~500 MB) [instrumento F7-D6]
  -> S1 operational baseline (pre-carga de modelo)
  -> Overhead_CUDA/Driver/Runtime = S1 - S0        (§2.4)
  -> carga del pipeline
  -> workload canónico: 30 steps DiT (VAE medida aparte)
  -> Peak_NVML_raw, Delta_induced, snapshots de allocator y eventos CUDA
  -> S3 post-workload (60 s) + Control B (~500 MB)
  -> release / cooldown
```

**Iteraciones de instrumentación:** máximo **2**. Entre la 1.ª y la 2.ª solo pueden cambiar
frecuencia de muestreo, puntos de sincronización, snapshots e instrumentación. Nunca el workload.

---

## 5. Criterio de cierre y matriz de decisión

**F9-0 se cierra como `CLOSED — ATTRIBUTED`** cuando se cumple todo:

1. `Peak_NVML_raw` correctamente identificado y ventana temporal localizada.
2. Atribución ≥ **90 %** de `Delta_induced` por la partición de §2.2 (o justificación
   metodológica equivalente).
3. `LB_analitico`, `Min_allocated_obs` y `LB_operativo` presentados.
4. `Overhead_CUDA/Driver/Runtime` cuantificado.
5. Componente dominante del exceso determinado.
6. Recomendación trazable del primer mecanismo a investigar (prioridad §6 del charter).
7. Sin clasificación `POSIBLE SUBESTIMACIÓN DEL PICO` (§2.5).
8. Exclusividad verificada (§2.3) en las corridas válidas.

Si tras **2 iteraciones** no se alcanza el 90 %:

> **`ATTRIBUTION INCOMPLETE — DOCUMENTED`** — se documenta la fracción no explicada y sus fuentes
> plausibles. No se continúa indefinidamente con instrumentación sin autorización adicional.

| Resultado | Condición | Acción de gobernanza |
| :---: | :--- | :--- |
| 🟢 `CLOSED — ATTRIBUTED` | ≥90 % de `Delta_induced` + ventana localizada + sin subestimación | Dirección selecciona **un único** mecanismo → genera F9-1 |
| 🟡 `ATTRIBUTION INCOMPLETE — DOCUMENTED` | <90 % tras 2 iteraciones | Documentar fracción no explicada; Dirección decide |
| 🔴 `INSTRUMENT INVALID` | S0 inestable, sin exclusividad o subestimación del pico | Re-iterar instrumentación (máx. 2); no se cierra F9-0 |

---

## 6. Salidas y artefactos

1. **Telemetría cruda:** `logs/f9_0_attribution_telemetry.json` (series completas, snapshots,
   eventos CUDA, métricas de instrumento).
2. **Reporte formal:** `docs/F9_0_ATTRIBUTION_REPORT_01.md` con la partición de las 7 categorías,
   `Peak_NVML_raw`, `Delta_induced`, `LB_analitico`, `Min_allocated_obs`, `LB_operativo`, overhead
   y veredicto.
3. **Recomendación:** mecanismo prioritario trazable para F9-1.

---

## 7. Gobernanza

* F9-0 es **observacional y reversible**: no modifica el runtime, no cambia el gate y **no
  reclassifica F7-D5 / Stage B / F8**.
* F7 y F8 permanecen **CERRADAS**.
* Ninguna ejecución se lanza sin autorización explícita de la Dirección.
* La correspondencia con el Consejero reside en `docs/private/` y no se versiona.

---

## 8. Registro de verificación de referencia y revisión interna de consistencia

**Secuencia formal cumplida antes del congelamiento:**

1. **Crear F9** — charter generado (§ `F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md`).
2. **Generar preregistro F9-0** — este documento.
3. **Resolver/verificar la referencia F7-D5** — hecho §2.7: F7-D5 = **5,096.1 MB**
   (`peak_nvml_used`), déficit **+296.1 MB**, vinculado a
   `logs/f7_d5_full_30step_validation_telemetry.json` y `docs/F7_D5_FULL_30STEP_VALIDATION_REPORT_01.md`.
   Se excluye 4,996.0 MB (F7 Stage B) como referencia F7-D5.
4. **Revisión interna de consistencia** — las 8 observaciones están incorporadas como definiciones
   operacionales en §2.1–§2.8; sin contradicciones internas; coherente con ROADMAP §2 y con el
   instrumento validado F7-D6.
5. **Congelar preregistro** — estado `F9-0 — PREREGISTERED / FROZEN`.

**Checklist de las 8 observaciones:**

| # | Observación | Resolución en este documento |
| :-: | :--- | :--- |
| 1 | Denominador del 90 % = pico inducido (`Peak NVML − S0`) | §2.1 |
| 2 | Categorías mutuamente excluyentes, categoría primaria única | §2.2 |
| 3 | Exclusividad GPU realista (sin otros contextos CUDA + S0 estable) | §2.3 |
| 4 | Lower Bounds reproducibles (Analítico / Min. Allocated / Operativo) | §2.4 |
| 5 | Sandbox basado en DiT, no UNet, para G3 | Charter §3 |
| 6 | Atribución sobre DiT/diffusion, VAE separado, reutilizar F7-D6 | §2.6 |
| 7 | Referencia F7-D5 única verificada y vinculada a artefacto | §2.7, §8 |
| 8 | F9-1…F9-4 solo placeholders; únicamente F9-0 habilitado | §2.8 |

---

**ESTADO: F9-0 — PREREGISTERED / FROZEN — NO EJECUTADO. PENDIENTE DE AUTORIZACIÓN DE EJECUCIÓN.**
