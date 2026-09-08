# Especificación de Prueba — Fase F4 (Adaptive Memory Decision Engine)

**Propósito:** Definir, de forma objetiva y verificable, QUÉ se construirá en F4 y CÓMO un
agente profesional podrá validar que la implementación cumple la especificación, reutilizando
las primitivas congeladas de F3 y F3+INT8 **sin reescribirlas**.

**Fecha:** 2026-09-08 · **Estado:** 🟡 **IMPLEMENTANDO** — contrato fijado y aprobado por el
Director (2026-09-08). Revisión del Consejero incorporada (v3).

> **Revisión del Consejero Técnico Externo (2026-09-08):** veredicto 🟢 **AUTORIZAR
> IMPLEMENTACIÓN DE F4** con cambios menores. Esta v3 incorpora las 6 recomendaciones de la
> carta antes de congelar el contrato de prueba: (1) hysteresis en el detector de presión,
> (2) `residency` explícito de 3 estados, (3) telemetría `decision_overhead_ms_total` /
> `switch_count` / `replan_count`, (4) separación formal **Hard Gate 4,800 MB** vs
> **Engineering Target 4,000 MB**, (5) **≥5% como objetivo de optimización**, no único
> criterio de validez (gates Safety/Adaptive/Optimization), (6) primitivas F3/F3+INT8
> estrictamente congeladas.
>
> ⚠️ **Política de autorización (Director):** ningún cambio posterior a esta especificación
> se aprueba sin autorización explícita del Director del Proyecto. Las primitivas F3/F3+INT8
> NO se tocan durante F4.
>
> ✅ **APROBACIÓN DEL DIRECTOR (2026-09-08):** esta especificación v3 queda **fijada como
> contrato oficial** de la Fase F4 y se autoriza su implementación conforme a los §1–§6.
> Registro: `docs/F4_TEST_SPEC_01.md` · Estado de F4 en ROADMAP: ⚪ Planned → 🟡 Implementando.

---

## 1. Alcance (definición de terminado)

F4 introduce **únicamente el bucle de decisión**. Queda EXPRESAMENTE fuera de alcance:
NF4, reescritura de schedulers, y ajustes ad-hoc de dequant (directiva de congelación §10 del plan
[`docs/F4_ADAPTIVE_ENGINE_PLAN_01.md`](F4_ADAPTIVE_ENGINE_PLAN_01.md)).

### 1.1 Entregables de código esperados (obligatorios)

| Ruta | Responsabilidad esperada |
| :--- | :--- |
| `marley/core/adaptive.py` | Motor de decisión: observador + selector de perfil + planificador AOT + checkpoints de re-planificación. |
| `marley/core/policies.py` | Objetos de estrategia FP16/INT8, prefetch y residencia (no duplicar lógica de los streamers). |
| `f4_adaptive_benchmark.py` | Runner con **validación same-session A/B/C** y modo de presión inyectada. |
| `logs/f4_adaptive_benchmark.json` | Telemetría de salida (estructura JSON definida abajo). |
| `results/TEST_F4_adaptive_benchmark.md` | Informe humano. |
| Fila nueva en `results/TEST_REGISTRY.md` | Registro maestro actualizado. |

### 1.2 Primitivas REUTILIZADAS (prohibido modificarlas)

- `marley/ops/async_stream.py` — `BudgetedAsyncStreamer` (+ `StreamMetrics`).
- `marley/ops/async_stream_int8.py` — `INT8BudgetedStreamer`.
- `NVMLSampler` patrón del runner F3 (50 ms).
- Protocolo de 7 métricas y Enmiendas 1–3 de F3.

---

## 2. Qué se va a construir (comportamiento funcional)

### 2.1 El motor de decisión

El motor observa el estado del runtime **por bloque / por paso** y elige, en cada checkpoint,
una triple decisión:

```
(precision ∈ {FP16, INT8}, prefetch ∈ {aggressive, conservative, off}, residency ∈ {keep, prefetch, evict})
```

Las **señales observadas** (inputs de decisión) son exactamente las 6 del plan §3:
1. Headroom VRAM libre/usado (NVML).
2. Latencia PCIe H2D acumulada reciente.
3. Tiempo de cómputo de capa (por bloque, timeline de eventos).
4. EMA de presión de memoria (WDDM de fondo).
5. Estado de prefetch / **stall real medido** (timeline `per_block_stall_s`).
6. **Coste de dequant** en el camino crítico (señal nueva de F3+INT8, ~324 ms).

Regla de mapeo señal → acción (sketch del plan §4, debe materializarse en una función pura
y testeable `decide(state) -> PolicyDecision`).

### 2.2 Perfiles operativos (directiva §5/§6)

| Perfil | precision | prefetch | residency | objetivo |
| :--- | :--- | :--- | :--- | :--- |
| `performance` | INT8 | aggressive | `keep` | mínima latencia, máx. VRAM permitida |
| `memory_safe` | FP16 o INT8 según presión | conservative | `evict` (temprano) | ≥ +1.5 GB headroom ante presión WDDM |

> `residency ∈ {keep, prefetch, evict}` se representa EXPLÍCITAMENTE como campo de 3 estados
> (no como booleano) para que la interfaz refleje fielmente la decisión de 3 dimensiones
> `precision × prefetch × residency` (carta del Consejero §4).

Granularidad de decisión: **bloque/capa** (no per-tensor, por directiva).

### 2.3 Modelo de ejecución

- Plan estático **Ahead-Of-Time (AOT)** generado tras el warmup del paso 1.
- Checkpoints de re-evaluación cada `N` pasos de denoise **o** ante desviación brusca de
  presión de memoria (event-triggered).
- En cada checkpoint se elige precision + prefetch + residencia para la siguiente ventana.

### 2.4 Definiciones cuantitativas (resolver las preguntas abiertas del plan §14)

Para que un agente pueda verificar, se FIJAAN estos valores (decisión de kickoff):

| Parámetro | Valor fijo |
| :--- | :--- |
| `N` (periodo de re-evaluación) | `5` pasos de denoise |
| Representación dual INT8 | **pre-computada en host** (evita re-cuantizar en caliente): el streamer INT8 ya guarda payload INT8 + escala FP16 en host → reutilizar |
| Overlap del dequant en F4 | permitido en el bucle de decisión (elegir INT8=anticipar dequant), **no** cambios de scheduler |

**Detector de presión con HYSTERESIS (carta §3):**

| Constante | Valor fijo |
| :--- | :--- |
| `PRESSURE_HIGH` | `EMA + 200 MB` → dispara entrada en régimen de presión |
| `PRESSURE_LOW` | `EMA − 200 MB` → dispara salida del régimen de presión |
| `MIN_DWELL_WINDOWS` | `1` ventana (`N=5` pasos) de permanencia mínima antes de volver a cambiar de política |

Regla anti-oscilación: una vez conmutada la política por presión, no se permite volver a la
política anterior hasta cumplir `MIN_DWELL_WINDOWS` **y** que la señal cruce el umbral
opuesto (`PRESSURE_LOW` para salir). Evita ciclos `INT8→FP16→INT8…` por fluctuación WDDM
normal (el runtime adaptativo debe ser **estable**, no solo reactivo).

---

## 3. Interfaces que el agente deberá verificar (contrato)

### 3.1 `marley/core/policies.py`

```python
@dataclass
class PolicyDecision:
    precision: str            # "fp16" | "int8"
    prefetch: str             # "aggressive" | "conservative" | "off"
    residency: str            # "keep" | "prefetch" | "evict"   (3 estados, carta §4)

@dataclass
class RuntimeState:           # 6 señales del §2.1
    free_vram_mb: float
    recent_h2d_ms: float
    layer_compute_ms: float
    ema_pressure_mb: float
    prefetch_stall_ms: float
    dequant_cost_ms: float

class PolicySelector:         # mapea perfil/norma -> PolicyDecision
    def select(self, profile: str, state: RuntimeState) -> PolicyDecision: ...
```

Criterio de aceptación: `select()` es **pura** (sin E/S, sin CUDA), determinista y cubre
los perfiles `performance` y `memory_safe` según las reglas del §2.2/§4. `residency` debe
aceptar EXACTAMENTE los tres valores `{"keep","prefetch","evict"}`; un constructor booleano
o un 4º valor inválido se considera **violación de contrato**.

### 3.2 `marley/core/adaptive.py`

```python
class AdaptiveEngine:
    def __init__(self, blocks, device, dtype, streamer_cls=INT8BudgetedStreamer): ...
    def plan_aot(self, warmup_state: RuntimeState) -> List[PolicyDecision]: ...   # una decisión por ventana de N pasos
    def checkpoint(self, state: RuntimeState) -> PolicyDecision: ...              # re-evaluación
    def should_replan(self, state: RuntimeState, current: PolicyDecision) -> bool: ...
    def run(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
            num_steps, num_frames=17, profile="performance",
            pressure_override_cb=None) -> StreamMetrics: ...
```

Criterio de aceptación:
- El motor **delega la ejecución** en `BudgetedAsyncStreamer` (FP16) o `INT8BudgetedStreamer`
  (INT8); nunca implementa su propio bucle de transferencia/compute.
- `pressure_override_cb` permite al runner **inyectar presión de VRAM** (necesario para el
  test dinámico de Milestone D / §4.3).
- Cambio INT8↔FP16 entre ventanas debe **conmutar de streamer sin re-cuantizar** en caliente
  (usa representación host ya disponible en `INT8BudgetedStreamer.host_weights`).

### 3.3 `f4_adaptive_benchmark.py` — CLI esperada

```
python f4_adaptive_benchmark.py [--steps 30] [--reps 3] [--profile performance]
python f4_adaptive_benchmark.py --pressure-test          # Milestone D (inyección)
python f4_adaptive_benchmark.py --abc                     # same-session A/B/C (Sync-FP16/Async-FP16/Async-INT8)
```

---

## 4. Procedimiento de verificación del agente (pass/fail objetivo)

### 4.0 Precondiciones

- CUDA disponible; Wan2.1-T2V-1.3B descargable (offline ya cacheado si aplica).
- Enmascarar stdout en win32 (mismo patrón `TextIOWrapper` del runner F3) si hace falta.

### 4.1 Test estático (sin GPU — unit)

El agente verificará, mediante import y llamada aislada:
1. `PolicySelector.select("performance", state_alta_presion)` → `INT8/aggressive`.
2. `PolicySelector.select("memory_safe", state_baja_presion)` → conservador + evict.
3. `select()` no toca CUDA ni imprime (determinismo).

**PASS** si los 3 criterios se cumplen.

### 4.2 Test de integración (GPU real, baseline) — same-session A/B/C/D

Ejecutar `--abc` en **una sola sesión**. El runner debe:
- Ejecutar las 4 condiciones en la misma sesión: `A=Sync FP16`, `B=Async FP16`,
  `C=Async INT8`, `D=Adaptive`.
- Aplicar la regla de **margen same-session** del plan §8 (carta §7):
  - `beneficio_scheduler = (A−B)/A`
  - `beneficio_int8 = (B−C)/B`
  - El motor D se compara SOLO contra `D vs mejor(A,B,C)` dentro de la misma sesión,
    nunca contra resultados históricos/cross-session.
- Registrar las 7 métricas F3 por condición + decisión correcta + overhead de
  decisión/re-planificación/conmutación (§5).

**El ≥5% NO es aquí el único criterio de validez** (carta §2). El veredicto se descompone en
tres niveles evaluados por separado (§4.4):

1. **Safety Gate** (siempre exigido): Peak NVML ≤ 4,800 MB · 0 NaN/Inf · política correcta.
2. **Adaptive Gate** (siempre exigido): el motor observa→decide→ejecuta→replanifica y
   recupera estabilidad bajo presión (§4.3). No requiere 5% de velocidad.
3. **Optimization Target** (objetivo): D supera a `mejor(A,B,C)` same-session en ≥ 5% de
   headroom **o** de velocidad. Si NO lo alcanza, el resultado **NO se declara FAIL** de
   inmediato: se informa como no-alcanzado y se evalúa si la complejidad de control aporta
   valor por adaptación (en cuyo caso puede retenerse) o si procede el **fallback a política
   fija** (Async INT8 / performance). El fallback solo se activa tras esta evaluación por el
   Director, no automáticamente por no llegar al 5%.

### 4.3 Test dinámico de presión (Milestone D) — con hysteresis

Con `--pressure-test`, durante la corrida el runner inyecta un aumento artificial de uso de
VRAM (p.ej. reserva/fragmento) y verifica:
1. Entrada en régimen: `used > EMA + PRESSURE_HIGH (200 MB)` dispara `should_replan() = True`.
2. El motor conmuta a la política más conservadora (perfil `memory_safe` /
   `residency=evict`), restaurando **headroom ≥ +1.5 GB**.
3. Salida estable: al remitir la presión, NO vuelve a `performance` hasta cumplir
   `MIN_DWELL_WINDOWS` y cruzar `EMA − PRESSURE_LOW (200 MB)` (sin oscilar).
4. Se registran el/los evento(s) de conmutación, su latencia/overhead y los contadores.

**PASS** del **Adaptive Gate** si se observan los 4 puntos (especialmente la estabilidad /
ausencia de oscilación del punto 3).

### 4.4 Umbrales / gates consolidados (carta §2 y §6)

| Nivel | Gate | Umbral |
| :--- | :--- | :---: |
| **Hard Gate** (bloqueante) | Peak VRAM (NVML) | ≤ 4,800 MB |
| **Hard Gate** (bloqueante) | NaN / Inf | ninguno |
| **Hard Gate** (bloqueante) | Corrección de política | perfil seleccionado == perfil intencionado |
| **Adaptive Gate** | Detección de presión + replanificación + conmutación correcta | sí (§4.3) |
| **Adaptive Gate** | Recuperación de headroom (escenario diseñado) | ≥ +1.5 GB |
| **Engineering Target** (no bloqueante) | Peak VRAM (NVML) | ≤ 4,000 MB |
| **Optimization Target** (objetivo) | D vs `mejor(A,B,C)` same-session (headroom o velocidad) | ≥ 5 % |

> **4,800 MB ≠ 4,000 MB** (carta §6): 4,800 es el **Hard Gate** operativo; 4,000 es un
> **Engineering Target** ambicioso (margen extra ante WDDM). Se reportan por separado en el
> informe final, sin tratarlos como equivalentes.

Fallback autorizado: **política estática determinista** (Async INT8 / performance), activado
tras evaluación del Director (no automático por no alcanzar el 5%).

---

## 5. Estructura JSON de salida esperada (`logs/f4_adaptive_benchmark.json`)

```json
{
  "phase": "F4",
  "device": "...", "vram_total_mb": 0.0, "num_blocks": 30,
  "num_steps": 0, "num_frames": 17, "num_reps": 0,
  "mode": "abc" | "pressure",
  "same_session": {
    "A_sync_fp16_ms": 0.0, "B_async_fp16_ms": 0.0, "C_async_int8_ms": 0.0,
    "scheduler_benefit_pct": 0.0, "int8_benefit_pct": 0.0,
    "best_static_condition": "A|B|C", "best_static_ms": 0.0
  },
  "engine": {
    "profile": "performance",
    "wall_ms": 0.0, "vs_best_static_pct": 0.0,
    "decisions": ["<PolicyDecision> por ventana"],
    "replan_events": [],
    "switch_overhead_ms_total": 0.0,
    "decision_overhead_ms_total": 0.0,   # carta §5: coste de observación+EMA+selección
    "switch_count": 0,                    # carta §5: nº de cambios de streamer/política
    "replan_count": 0,                    # carta §5: nº de re-evaluaciones fuera de AOT
    "pressure_events": [],
    "peak_vram_mb": 0.0
  },
  "pass": true,
  "gates": {
    "hard_gate_vram_mb_limit": 4800.0,          # carta §6: bloqueante
    "engineering_target_vram_mb": 4000.0,       # carta §6: objetivo no bloqueante
    "nan_inf": false,
    "adaptive_gate_headroom_recovered_gb": 1.5, # carta §2/§4.3
    "optimization_target_pct": 5.0              # carta §2/§4.4 (objetivo, no único criterio)
  }
}
```

> Nota de overhead (carta §5): para decidir si el motor vale, se separa la **ganancia real
> del runtime** de la **ganancia aparente compensada por overhead de control**. El motor solo
> es interesante si `decision_overhead_ms_total + switch_overhead_ms_total` es mucho menor que
> el beneficio que obtiene (`vs_best_static_pct`).

---

## 6. Definición de terminado (DoD) que el agente debe confirmar

- [ ] `marley/core/policies.py` y `marley/core/adaptive.py` existen y cumplen el contrato §3
      (incl. `residency ∈ {keep, prefetch, evict}` explícito).
- [ ] Las primitivas F3/F3+INT8 NO fueron modificadas (diff vacío sobre `marley/ops/async_stream*.py`).
- [ ] `--abc` same-session produce **A/B/C/D** y el margen del motor D se expresa relativo a
      `mejor(A,B,C)` same-session (nunca cross-session).
- [ ] **Safety Gate** (§4.4): Peak NVML ≤ 4,800 MB, 0 NaN/Inf, política correcta.
- [ ] **Adaptive Gate** (§4.3): detección con hysteresis (sin oscilación) + replanificación +
      recuperación de headroom ≥ +1.5 GB bajo presión.
- [ ] **Optimization Target** (§4.4): si D ≥ 5% sobre `mejor(A,B,C)` → señalado como alcanzado;
      si < 5% → NO se declara FAIL automático, se reporta overhead vs beneficio y se deja a
      decisión del Director el fallback a política fija.
- [ ] Telemetría JSON conforme a §5 incl. `decision_overhead_ms_total`, `switch_count`,
      `replan_count` y gates separados Hard 4,800 / Target 4,000; informe Markdown + fila de
      registry generados.

---

*Marley Runtime — Fase F4 · en memoria de Marley 🐾.*
