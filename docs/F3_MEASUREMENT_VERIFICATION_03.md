# F3 — Verificación Final de Medición
## Enmiendas de Instrumentación y Protocolo de Certificación

> ✅ **RESULTADO (2026-09-08):** batería de verificación ejecutada y superada → **F3 CERTIFIED PASS**.
> Reporte: [`results/TEST_F3_verification.md`](../results/TEST_F3_verification.md) ·
> Telemetría: [`logs/f3_verification_benchmark.json`](../logs/f3_verification_benchmark.json).

**Proyecto:** Marley Runtime
**Fase:** F3 — Budgeted Asynchronous Scheduler
**Fecha:** 8 de septiembre de 2026
**Status:** 🟢 **INSTRUMENTACIÓN CORREGIDA — F3 CERTIFICADO PASS** (verificada 2026-09-08)
**Resultado certificado:** mean **+7.2%** external wall-clock (rango +6.0% / +8.4%, σ ±1.1%) · overlap 100% medido (0 ms stall) · 2,584 MB NVML · 0 NaN/Inf — [`results/TEST_F3_verification.md`](../results/TEST_F3_verification.md)
**Origen:** Cartas cruzadas Director ↔ Consultor técnico externo (SD / ComfyUI / runtimes)

---

## 1. Contexto y trazabilidad

Tras la corrida oficial de F3 (`results/TEST_F3_async_scheduler.md`, `logs/f3_async_scheduler_benchmark.json`) y dos rondas de revisión del consultor técnico externo, se identificaron **deficiencias de instrumentación** que impedían certificar formalmente F3:

| Ronda | Hallazgo | Resolución |
| :---: | :--- | :--- |
| Carta 1 | F3-B no evalúa el gate de overlap en `check_kill_gates()` | Se incorpora `overlap_pct` al gate F3-B (tras corregir la medición) |
| Carta 1 | Speedup derivado de tiempo interno, no del wall-clock externo capturado | Wall-clock externo pasa a ser la métrica oficial |
| Carta 2 | El "100% de overlap" reportado merece comprobación | Auditoría de código → hallazgo §2 |
| Auditoría | `total_prefetch_latency` jamás se acumula → overlap = 100% por construcción | Corrección §3 (Enmienda 1) |

Este documento describe en detalle la corrección de instrumentación, los cambios de código, las fórmulas y el protocolo de certificación final. Es la fuente de verdad de la implementación actual.

---

## 2. Hallazgo de la auditoría — El "100% de overlap" era un artefacto

### 2.1 Código defectuoso (antes)

En `marley/ops/async_stream.py`, dentro de `execute_async_loop()`:

```python
total_prefetch_latency = 0.0                          # se inicializa…
…
latency_ev_start = torch.cuda.Event(enable_timing=True)   # se registra…
self.compute_stream.wait_event(self.copy_done_events[i + 1])  # …se espera…
…
# PERO NUNCA se computa elapsed ni se acumula:
# hidden_ratio = 1.0 - (total_prefetch_latency / total_h2d_time)   ← total_prefetch_latency ≡ 0
m.effective_overlap_pct = hidden_ratio * 100.0       # → 100,0% por construcción
```

El acumulador `total_prefetch_latency` permanecía en `0.0` en todo el flujo. La instrumentación de stall (evento `latency_ev_start` + `wait_event`) estaba registrada pero **incompleta**: nunca se media el intervalo entre el evento de espera y el `copy_done_events[i+1]`.

### 2.2 Consecuencia

```text
effective_overlap_pct = (1 − 0 / H2D) × 100 = 100,0%   (siempre, con H2D > 0)
```

El valor "100,0%" de la corrida oficial **no era una medición**; era una propiedad accidental de la fórmula. Además, `ev_compute_start/end` solo se leían tras el bucle completo (sobre la última iteración de eventos reutilizados), por lo que cualquier acumulación por pasos estaba subestimada.

### 2.3 Alcance del impacto

- **No invalida** el speedup preliminar de **+10,4%** (wall-clock `perf_counter` del bucle, independiente del artefacto).
- **Invalida** el "100% de overlap" como evidencia central de F3.
- Reafirma la necesidad de **medir stalls reales antes de activar el gate de overlap** (si el gate se hubiera conectado con esta medición, habría sido un gate vacío: siempre PASS).

---

## 3. Enmienda 1 — Medición real de overlap / stalls

### 3.1 Qué se mide ahora (post-corrección)

Por cada paso de denoise y por cada bloque prefetched se capturan tres cronologías GPU mediante eventos CUDA `enable_timing=True`:

| Señal | Eventos | Significado |
| :--- | :--- | :--- |
| **Stall real** | `stall_start_evs[i+1]` (compute) vs. `copy_done_events[i+1]` (copy) | Tiempo GPU que el compute stream espera de verdad a que el prefetch de `block[i+1]` termine. Si el prefetch ya estaba listo → delta negativo → clamp a 0 (sin stall). |
| **Copy (prefetch)** | `ev_copy_start[j]` (copy) vs. `copy_done_events[j]` (copy) | Duración GPU real de la copia H2D asíncrona de `block[j]` (j ≥ 1). |
| **Compute** | `ev_compute_start[i]` vs. `ev_compute_end[i]` (compute) | Duración GPU real del forward de `block[i]`. |

Los tres se **acumulan por bloque a lo largo de todos los pasos** (cronología `per_block_*`) y en totales (`total_prefetch_latency`, copy total, compute total). La medición de cada paso se ejecuta inmediatamente después del `torch.cuda.synchronize()` de fin de paso, **antes** de que la siguiente iteración re-registre los eventos (los arrays de eventos se reutilizan entre pasos).

### 3.2 Fórmulas

```text
Measured async copy time   = Σ_steps Σ_{j=1..N-1} copy_ms(j)      # trabajo H2D prefetcheable
Total prefetch latency     = Σ_steps Σ_{j=1..N-1} max(0, stall_ms(j))
Effective overlap          = (1 − total_prefetch_latency / measured_async_copy_time) × 100
```

Acotado a `[0, 100]`. Si `measured_async_copy_time == 0` → `0,0%`.

### 3.3 Cambio de código — `marley/ops/async_stream.py`

1. `StreamMetrics`: se añaden los campos de cronología `per_block_copy_s`, `per_block_compute_s`, `per_block_stall_s` (`List[float]`, `len == num_blocks`). El campo existente `prefetch_latency_s` pasa a contener el **stall real acumulado** (antes siempre 0).
2. `execute_async_loop()`:
   - Acumuladores nuevos: `measured_async_copy_s`, `measured_compute_s`, `per_block_*`.
   - El evento de stall por bloque se conserva en `stall_start_evs[j]` dentro del paso (antes se descartaba).
   - Tras el `torch.cuda.synchronize(device)` de cada paso se calculan y acumulan `elapsed_time` de (stall, copy, compute).
   - El bloque 0 (carga síncrona de inicio de paso) se registra en `per_block_copy_s[0]` con timing host.
   - `total_h2d_transfer_time_s` se define ahora como el **H2D medido** (copias GPU de bloques ≥ 1 + bloque 0 síncrono). El parámetro `sync_baseline_h2d_s` queda solo como *fallback* si la medición por eventos no estuviera disponible.
   - `effective_overlap_pct` deriva de la fórmula §3.2.
3. `execute_sync_loop()`: registra `per_block_copy_s` (timing host) para el timeline A/B; `overlap = 0,0%` por definición (sin cambios de semántica).

### 3.4 Auditoría

El "100%" —si se mantiene tras la corrección— quedará respaldado por:
```text
H2D total  (ms)      → Σ copies medidas
Stall total (ms)     → Σ stalls reales
per_block_copy_s     → timeline de copia por bloque
per_block_stall_s    → timeline de espera por bloque
per_block_compute_s  → timeline de cómputo por bloque
```

---

## 4. Enmienda 2 — Orden A/B alternado + warmup

### 4.1 Motivación

Mantener siempre el orden `A→B` asocia sistemáticamente cualquier deriva térmica / de caching / del driver con la condición B. Para que la distribución sea interpretable:

```text
Rep 1 → A/B     Rep 2 → B/A     Rep 3 → A/B     Rep 4 → B/A   …
```

con una **ejecución de warmup descartada** antes de la primera repetición medida.

### 4.2 Cambio de código — `f3_async_scheduler_benchmark.py`

- Nuevo argumento `--f3b-reps` (default 5) y `--no-f3b-warmup` (warmup activo por defecto).
- F3-B ejecuta N repeticiones alternando el orden real de ejecución por repetición.
- Independientemente del orden, `A = Marley Sync`, `B = Marley Async` (la asignación de variables no depende del orden de ejecución; el speedup se calcula siempre `(wall_sync − wall_async) / wall_sync`).
- Se ejecuta un warmup (Sync + Async) descartado para estabilizar clocks/estado del driver.
- Se reportan, por condición y para el speedup: **media, mínimo, máximo, desviación estándar**.

---

## 5. Enmienda 3 — Wall-clock externo como métrica oficial

### 5.1 Decisión

El speedup oficial usa exclusivamente el **wall-clock externo** capturado en el benchmark alrededor de cada condición:

```python
wall_sync  = t_sync_end  - t_sync_start
wall_async = t_async_end - t_async_start
speedup_pct = ((wall_sync - wall_async) / wall_sync) * 100.0
```

El tiempo interno `total_denoise_time_s` se conserva como **métrica diagnóstica**.

### 5.2 Consistencia interna/externa

Ambos temporizadores deben cubrir el mismo trabajo (activaciones en host → resultado en GPU, excluyendo carga de modelo y warmup). Se reportan ambos y se verifica su coherencia; una divergencia anómala entre el wall-clock externo y el interno es, en sí misma, un indicador de error de instrumentación.

### 5.3 Cambio de código — `f3_async_scheduler_benchmark.py`

- `speedup_pct` se calcula con los temporizadores externos por repetición.
- Los temporizadores externos e internos se registran ambos por condición en el JSON de salida.
- Independencia de estado: antes de cada condición se usan clones de los tensores de entrada (`hidden_states.clone()`, etc.) para eliminar cualquier contaminación A/B.

---

## 6. Kill Gates de F3-B (activación)

`check_kill_gates()` en F3-B recibe ahora **los cinco criterios**:

| Gate | Umbral | Base de evaluación |
| :--- | :---: | :--- |
| Peak VRAM (NVML) | `> 4.800 MB` → FAIL | Máximo sobre todas las repeticiones |
| Overlap real | `< 5,0%` → FAIL | Mínimo sobre todas las repeticiones |
| Wall-clock speedup | `< 0,0%` → FAIL | Mínimo sobre todas las repeticiones (cualquier regresión en una repetición = revisión) |
| NaN/Inf | cualquier detección → FAIL | OR sobre todas las repeticiones |
| Consistencia | std dev reportada | Métrica de diagnóstico, no gate |

El criterio de interpretación del overlap se mantiene:

| Overlap | Evaluación |
| ------: | :--- |
|     <5% | ❌ FAIL |
|   5–30% | ⚠️ Mínimo |
|  30–70% | 🟢 Bueno |
|    >70% | 🟢 Excelente |

---

## 7. Protocolo de certificación final

```text
F3 actual (congelado, sin cambios de arquitectura)
   ↓
1. Enmienda 1 — instrumentación de overlap/stall corregida
   ↓
2. Gate de overlap activado en F3-B (con medición real)
   ↓
3. Enmienda 3 — wall-clock externo como métrica oficial
   ↓
4. Enmienda 2 — N repeticiones con A/B alternado + warmup
   ↓
5. Reporte por condición: media / mínimo / máximo / desviación estándar
   ↓
6. ¿Speedup estable, reproducible y correctamente medido?
   ↓
   YES → 🟢 F3 PASS → F3 + INT8
   NO  → análisis de causas (sin rediseño)
```

Configuración congelada: mismo hardware (RTX 3050 6GB Laptop), mismo modelo (Wan2.1-T2V-1.3B, 30 bloques DiT), misma seed, mismos inputs/shapes, mismos frames (17), mismos steps, FP16, mismas versiones PyTorch/CUDA/driver, misma máquina.

---

## 8. Estructura de salida (JSON)

`logs/f3_async_scheduler_benchmark.json` — `f3b` pasa a contener:

```jsonc
{
  "pass": bool,
  "reps": int,
  "warmup": bool,
  "order_pattern": "AB/BA alternado",
  "aggregate": {
    "speedup_pct": {"mean": float, "min": float, "max": float, "std": float},
    "wall_clock_sync_ms":  {"mean": float, "min": float, "max": float, "std": float},
    "wall_clock_async_ms": {"mean": float, "min": float, "max": float, "std": float},
    "overlap_pct":         {"mean": float, "min": float, "max": float},
    "peak_vram_mb":        float,
    "nan_inf_detected":    bool,
    "stall_total_ms":      float
  },
  "per_rep": [ { "rep": int, "order": "A/B", "speedup_pct": float, "overlap_pct": float,
                 "wall_a_ms": float, "wall_b_ms": float, "internal_a_ms": float, "internal_b_ms": float,
                 "peak_vram_mb": float, "nan_inf": bool,
                 "timeline": { "per_block_copy_ms": [...], "per_block_stall_ms": [...],
                               "per_block_compute_ms": [...] } } ]
}
```

---

## 9. Archivos afectados

| Archivo | Cambio |
| :--- | :--- |
| `marley/ops/async_stream.py` | Enmienda 1: medición real de stall/copy/compute + cronología `per_block_*` |
| `f3_async_scheduler_benchmark.py` | Enmiendas 1–3: gate de overlap, wall-clock externo, clones de entrada, F3-B multi-rep A/B alternado + warmup + estadísticas |
| `README.md` | Estado F3 actualizado (verificación en curso) |
| `ROADMAP.md` | Fase F3 y tabla de gates actualizadas |
| `docs/F3_CALIBRATION_CHANGES_02.md` | Entradas de change log 8–10 (enmiendas) |
| `results/TEST_F3_async_scheduler.md` | Reporte de corrida (estado provisional) |

---

## 10. Autorización de ejecución

> [!IMPORTANT]
> **Regla estricta del proyecto: ninguna ejecución de benchmarks se lanza sin autorización escrita explícita del supervisor humano.**
> (regla registrada en `.agents/rules/execution_authorization.md`)

Esta fase queda en estado **listo para verificar**. La ejecución de la batería de repeticiones (N × Sync + Async) requiere autorización explícita.

Comando de verificación previsto:

```bash
python f3_async_scheduler_benchmark.py --steps 5 --f3a-reps 3 --f3b-reps 5 --output logs/f3_verification_benchmark.json
```
