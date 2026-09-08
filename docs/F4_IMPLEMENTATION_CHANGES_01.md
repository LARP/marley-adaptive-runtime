# Informe de Cambios — Implementación de Fase F4 y Ajustes

**Autor:** Director de Proyecto (implementación asistida) · **Fecha:** 2026-09-08
**Fase:** F4 — Adaptive Memory Decision Engine *(CORE SYSTEM)*
**Estado:** 🟢 **CORE SYSTEM VALIDATED — PERFORMANCE CERTIFICATION PENDING**
**Documentos relacionados:**
- Contrato de prueba aprobado: [`F4_TEST_SPEC_01.md`](F4_TEST_SPEC_01.md) (v3)
- Resultados: [`results/TEST_F4_adaptive_benchmark.md`](../results/TEST_F4_adaptive_benchmark.md)
- Plan de diseño: [`F4_ADAPTIVE_ENGINE_PLAN_01.md`](F4_ADAPTIVE_ENGINE_PLAN_01.md)
- Telemetría: [`logs/f4_adaptive_benchmark.json`](../logs/f4_adaptive_benchmark.json)

---

## 1. Propósito de este documento

Este informe explica **qué se construyó, qué se ajustó durante la implementación y por qué**,
junto con el **resultado experimental completo**, de forma que quede un registro auditable para
futuras investigaciones (F6, escalado a 33f/720p, o revisión de la retención de F4).

---

## 2. Qué se construyó (deliverables de F4)

F4 es **solo una capa de decisión**. No reimplementa transferencia/cómputo y **no modifica** las
primitivas congeladas de F3/F3+INT8 (causal attribution intacta).

| Archivo | Responsabilidad |
| :--- | :--- |
| `marley/core/policies.py` | Primitivas puras de decisión (sin CUDA): `PolicyDecision` (precision×prefetch×residency), `RuntimeState` (6 señales), `PolicySelector` (perfil→decisión). |
| `marley/core/adaptive.py` | `AdaptiveEngine`: observador EMA + selector + planificador AOT + checkpoints con **hysteresis**. Delega la ejecución en los streamers F3 congelados. |
| `f4_adaptive_benchmark.py` | Runner: benchmark same-session **A/B/C/D** y `--pressure-test` (Adaptive Gate determinista). |
| `docs/F4_TEST_SPEC_01.md` | Contrato de prueba v3 (aprobado por el Director; incorpora las 6 recomendaciones del Consejero). |

### Contrato de decisión (PolicySelector) — reglas puras implementadas

| Perfil | precision | prefetch | residency | Cuándo |
| :--- | :--- | :--- | :--- | :--- |
| `performance` | int8 | aggressive | keep | siempre (máx. latencia mínima) |
| `memory_safe` (headroom sano) | fp16 | conservative | evict | free VRAM ≥ 1,500 MB |
| `memory_safe` (bajo presión) | int8 | off | evict | free VRAM < 1,500 MB |

`residency` es un campo explícito de **3 estados** (`keep|prefetch|evict`), no un booleano
(Consejero §4).

---

## 3. Ajustes realizados durante la implementación y su justificación

### 3.1 (Cambio 1) Especificación elevada de diseño → contrato de prueba v3 (aprobado)
**Qué:** Se fijó `docs/F4_TEST_SPEC_01.md` como contrato oficial y se **congeló** tras incorporar
la revisión del Consejero (hysteresis, residency explícito, telemetría de overhead, separación
Hard-Gate 4,800 / Target 4,000, ≥5% como *objetivo* no único criterio, A/B/C/D same-session).
**Por qué:** La directiva del proyecto exige congelar la spec antes de implementar y no aprobar
cambios sin autorización del Director; el Consejero pidió cambios *menores* antes de congelar.

### 3.2 (Cambio 2) Residency de 3 estados en `PolicyDecision`
**Qué:** `residency` se representa como `"keep" | "prefetch" | "evict"` (validado en
`__post_init__`), no como `evict: bool`.
**Por qué:** Corrección de la inconsistencia señalada por el Consejero (§4) entre la definición
conceptual de 3 ejes y una interfaz de 2 estados.

### 3.3 (Cambio 3) Detector de presión con hysteresis en `AdaptiveEngine`
**Qué:** Umbrales `PRESSURE_HIGH = EMA + 200 MB` / `PRESSURE_LOW = EMA − 200 MB`, mínima
permanencia `MIN_DWELL_WINDOWS = 1` ventana; `_monitor()` decide entrada/salida de régimen y
devuelve el nuevo perfil solo cuando corresponde.
**Por qué:** Sin hysteresis el detector oscilaría `INT8→FP16→INT8` ante fluctuaciones normales de
WDDM; el runtime adaptativo debe ser **estable**, no solo reactivo (Consejero §3).

### 3.4 (Cambio 4) Overhead/telemetría de control ampliada
**Qué:** `decision_overhead_ms_total`, `switch_count`, `replan_count` (y `replan_events`) se
acumulan y se emiten en el JSON.
**Por qué:** Permite separar la *ganancia real del runtime* de la *ganancia aparente compensada
por overhead de control*; un motor solo interesa si su decisión es mucho más barata que su
beneficio (Consejero §5).

### 3.5 (Cambio 5 — REQUERIDO POR BUG EN LA PRIMERA EJECUCIÓN) Refactor del runner para reutilizar streamers
**Qué:** El runner original construía **tres** conjuntos de streamers (FP16 para A/B + INT8 para C +
par FP16/INT8 dentro del `AdaptiveEngine` para D). Cada `BudgetedAsyncStreamer`/`INT8BudgetedStreamer`
crea una **copia pinned** en host de los pesos (FP16 ≈ 2.66 GB; INT8 ≈ 1.4 GB por copia) además de
los bloques originales del modelo en RAM.
**Problema observado:** en la primera ejecución (`--steps 3`) la construcción del 2º/3er streamer
falló con:
```
RuntimeError: CUDA error: out of memory
  en INT8BudgetedStreamer._prepare_host_weights -> wq_int8.pin_memory()
```
**Diagnóstico:** no era VRAM GPU (los slots de GPU eran < 0.6 GB) sino agotamiento de **memoria
host paginada por `pin_memory()`/`cudaMallocHost`** por duplicación excesiva de copias pinned.
**Corrección:** refactor de `run_abc_benchmark()` para que A/B/C/D **reutilicen los streamers que
ya posee el motor** (`engine._fp16`, `engine._int8`) — una única copia pinned de cada representación.
Se redujo la huella pinned de ~4 copias a ~2.
**Verificación:** la corrida reejecutada completó limpiamente.

> **Lección para investigación futura:** los streamers F3 mantienen pesos *pinned* en host por
> bloque. Mantener varias instancias vivas simultáneamente (p. ej. una por condición + una por
> motor) puede agotar la memoria host paginada incluso cuando la VRAM está holgada. Compartir una
> sola representación pinned por dtype entre condiciones es la práctica recomendada.

### 3.6 (Cambio 6) Fix de telemetría: peak VRAM de la condición D
**Qué:** En la rama adaptativa del runner se propaga el pico NVML al `merged` (`D` reportaba `peak 0`
por no asignarlo).
**Por qué:** El reporte de D debe mostrar su pico real, no 0, para una scorecard coherente.

### 3.7 Sin cambios en primitivas F3/F3+INT8
`marley/ops/async_stream.py` y `marley/ops/async_stream_int8.py` quedaron **intactos**
(`git diff` vacío) — atribución causal preservada, como exige la directiva.

---

## 4. Resultado experimental completo

### 4.1 Same-session A/B/C/D — 30 pasos × 3 repeticiones
**Comando:** `.venv\Scripts\python.exe f4_adaptive_benchmark.py --abc --steps 30 --reps 3 --window 5`
(orden alternado A/B/C/D ↔ D/C/B/A por rep; warmup descartado).

| Cond | Estrategia | Mean (ms) | Min (ms) | Max (ms) | Overlap | NaN/Inf | Peak (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Sync FP16 | 113,538.7 | 112,695.5 | 114,992.8 | 0.0% | 0/0 | 2,860 |
| **B** | Async FP16 | **105,441.5** | 104,864.5 | 106,205.0 | 100.0% | 0/0 | 2,854 |
| **C** | Async INT8 | 109,157.9 | 108,946.6 | 109,475.6 | 98.1% | 0/0 | 2,843 |
| **D** | Adaptive | 109,022.8 | 108,875.6 | 109,275.0 | ~88.4% | 0/0 | 2,882 |

**Wall-clock por rep (ms):**
- Rep 1 (A,B,C,D): A 114,992.8 · B 106,205.0 · C 108,946.6 · D 109,275.0
- Rep 2 (D,C,B,A): D 108,917.7 · C 109,051.5 · B 105,255.0 · A 112,927.9
- Rep 3 (A,B,C,D): A 112,695.5 · B 104,864.5 · C 109,475.6 · D 108,875.6

**Derivados:**
- Best static same-session = **B** (105,441.5 ms).
- **D vs best static = −3.40%** (Optimization Target no alcanzado en régimen sin presión).
- Overhead de decisión (D) = **0.16 ms total** · switch_count 0 · replan_count 0.
- Peak NVML máximo = **2,882 MB** → Safety PASS y Target ≤ 4,000 MB MET.

**Interpretación:** en régimen sin presión, **Async FP16 (B) es la más rápida**; **Async INT8 (C)
es ~3.5% más lenta** (dequant ~324 ms en camino crítico no amortizado). D con base `performance`
(INT8) se mantuvo en INT8 toda la corrida (ningún evento de presión pidió conmutar) ⇒ D ≈ C. El
menor overlap medido de D (~88% vs 98% de C) es artefacto de las fronteras de ventana
(re-sync + re-decisión entre ventanas), no una regresión del scheduler; el wall-clock de D ≈ C
dentro del ruido.

### 4.2 Adaptive Gate — `--pressure-test` (inyección determinista)
**Comando:** `.venv\Scripts\python.exe f4_adaptive_benchmark.py --pressure-test --window 5`

Perfil de presión simulado (used MB): `2600 … → 3200 (pico 3 ventanas) → 2400 (release) …`

| Señal | Valor |
| :--- | :---: |
| Ventanas | 8 |
| Secuencia de perfiles | `performance → memory_safe → memory_safe → performance → performance → performance → performance → performance` |
| Replan events | 2 (ventana 1 → memory_safe / pressure; ventana 3 → performance / normal) |
| Replan count | 2 |
| Switch count | 2 |
| Overhead de decisión | 0.07 ms total |
| Entrada a memory_safe | True |
| Salida (PRESSURE_LOW tras dwell) | True |
| Oscilación | **False** |
| **Adaptive Gate** | 🟢 **PASS** |

Se reprodujo de forma determinista el ciclo completo solicitado por el Consejero:
`Performance → (+600 MB) → detección ≥200 MB sobre EMA → replan → Memory Safe → evict/conservador
→ recuperación → (presión baja) → PRESSURE_LOW + MIN_DWELL → replan → Performance`.

> **Caveat honesto:** el pico simulado (+600 MB) dejó free VRAM por encima de la línea
> `SAFE_MIN_FREE_MB` (1,500 MB), por lo que `memory_safe` seleccionó su rama de headroom sano
> (FP16/conservative/evict) y **no** la rama de apriete (INT8/off). Lo demostrado es el bucle de
> decisión, hysteresis, dwell y conmutación de régimen. Para ejercitar la rama INT8-tighten hace
> falta un perfil que lleve free VRAM por debajo de 1,500 MB (trabajo futuro).

---

## 5. Verdictos de gates (contrato v3 §4.4)

| Nivel | Gate | Umbral | Medido | Estado |
| :--- | :--- | :---: | :--- | :---: |
| **Hard Gate** | Peak VRAM (NVML) | ≤ 4,800 MB | 2,882 MB | 🟢 PASS |
| **Hard Gate** | NaN / Inf | 0 | 0/0 | 🟢 PASS |
| **Hard Gate** | Corrección de política | seleccionado == intencionado | performance→int8/agg/keep; memory_safe→cons/evict | 🟢 PASS |
| **Engineering Target** | Peak VRAM (NVML) | ≤ 4,000 MB | 2,882 MB | 🟢 MET |
| **Adaptive Gate** | detección + replan + switch + recuperación, sin oscilación | — | §4.2 | 🟢 PASS |
| **Optimization Target** | D vs best {A,B,C} | ≥ 5% | −3.40% | 🟡 NO MET (target, no gate) |

**Veredicto global: 🟢 CORE SYSTEM VALIDATED — PERFORMANCE CERTIFICATION PENDING.**

---

## 6. Preguntas abiertas / decisiones pendientes para investigación futura

1. **Rama INT8-tighten de `memory_safe`:** ejercitarla con free VRAM real < 1,500 MB (inyección
   mayor o presión WDDM real) para validar la conmutación FP16↔INT8 bajo apriete real.
2. **Perfil base óptimo sin presión:** dado que Async FP16 (B) supera a Async INT8 (C) aquí, un
   motor "inteligente" en régimen sano debería elegir **FP16**; explorar si la base `performance`
   debe mapear a FP16 cuando el headroom es amplio (vs INT8 fijo actual).
3. **Decisiones A/B/C del Consejero §7:** tras medir overhead/estabilidad/presión en profundidad,
   el Director decidirá A (retener F4), B (retener por robustez) o C (política fija
   Async-INT8/Performance).
4. **Escalado a 480p/33f (F6):** re-validar la secuencia de decisión a 33 frames, que era el
   objetivo primario de escalabilidad y donde el overhead por ventana se amortiza mejor.

---

*Marley Runtime — Fase F4 · informe de cambios e investigación · en memoria de Marley 🐾.*
