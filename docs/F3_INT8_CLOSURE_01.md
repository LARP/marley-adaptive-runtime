# Acta de Cierre — F3 + INT8 (Isolation: Transfer-Volume Reduction)
## Formal Closure · Frozen as Baseline for Phase F4

**Proyecto:** Marley Runtime
**Fase cerrada:** F3 + INT8 — Isolation: Transfer-Volume Reduction
**Fecha:** 2026-09-08
**Clase de documento:** Acta de cierre (Director del Proyecto), conforme a la
recomendación del consultor técnico externo (Generación de vídeo IA / SD / ComfyUI).
**Estado:** 🟢 **CERRADO — PASS** · Baseline congelado · Avance a F4 autorizado.
**Evidencia:** [`results/TEST_F3_INT8_benchmark.md`](../results/TEST_F3_INT8_benchmark.md) ·
Telemetría [`logs/f3_int8_benchmark.json`](../logs/f3_int8_benchmark.json) ·
Plan [`F3_INT8_EXPERIMENT_PLAN_01.md`](F3_INT8_EXPERIMENT_PLAN_01.md)

---

## 1. Objeto

Cerrar formalmente la fase F3 + INT8 y congelar sus resultados como **baseline de
referencia** para la siguiente etapa (F4 Adaptive Decision Engine), de acuerdo con el
veredicto PASS del consultor técnico externo.

## 2. Resultados certificados (resumen del cierre)

| Indicador | Valor | Gate | Veredicto |
| :--- | :---: | :---: | :---: |
| Payload H2D por bloque | **46,53 MB** (vs 92,88 MB FP16) | reducción objetiva | 🟢 **−49,9%** |
| Speedup Async-vs-Sync | **+13,2%** (min +10,4 / máx +15,4) | > 0% | 🟢 **PASS** |
| Overlap efectivo | **98,1%** (97,9–98,3) | ≥ 5% | 🟢 **PASS** |
| Stall acumulado | ~324 ms (critical path: dequant) | medido | 🟢 observado |
| Peak VRAM (NVML) | **2.076 MB** (−508 MB vs FP16) | ≤ 4.800 MB | 🟢 **PASS** |
| Fidelidad dequant (float64) | cos medio **0,999959** / mín **0,999935** | ≥ 0,99 | 🟢 **PASS** |
| NaN / Inf | **0 / 0** | ninguno | 🟢 **PASS** |

Todas las repeticiones fueron positivas. Todos los kill gates se satisfacen.

## 3. Precisión metodológica adoptada (obligatoria en la documentación oficial)

Conforme a la directriz del consultor, queda **prohibido** afirmar que
"INT8 hace a Marley ~8% más rápido que F3 FP16" comparando tiempos absolutos entre
sesiones (varianza térmica/clock). La afirmación oficial es:

> **En su propia comparación A/B, F3 + INT8 eleva el margen Async-vs-Sync hasta +13,2%,
> reduce el volumen H2D en −49,9% y baja el peak de VRAM a 2.076 MB.**

Esta formulación es la única vigente para cualquier publicación o comparativa.

## 4. Baseline congelado (bloque de referencia)

```text
F3 + INT8 (frozen)
Async-vs-Sync      +13,2%  (min +10,4 / máx +15,4)
Payload/bloque     46,53 MB  (−49,9% vs FP16)
Overlap            98,1%
Stall              ~324 ms  (dequant en critical path)
Peak NVML          2.076 MB
Cosine (float64)   ≥ 0,999935
NaN / Inf          0 / 0
```

## 5. Congelamiento de arquitectura

Queda **congelada** la arquitectura F3 + INT8. No se introducirán en esta fase:
- nuevas variantes del scheduler;
- NF4 / 4-bit (requiere validación de deriva temporal completa);
- modificaciones adicionales del doble buffer;
- optimizaciones ad hoc de dequantización.

Motivo: preservar la atribución causal y permitir que F4 mida su propio aporte contra un
baseline estable.

## 6. Decisión y próximo paso

- **F3 + INT8:** 🟢 **PASS — CERRADO y congelado como baseline.**
- **F4 FP16 baseline:** permanece como referencia histórica (no se sustituye).
- **Siguiente:** 🟢 **F4 — Adaptive Decision Engine.** Plan: [`F4_ADAPTIVE_ENGINE_PLAN_01.md`](F4_ADAPTIVE_ENGINE_PLAN_01.md).

---

*Firmado: Agente Director del Proyecto. Conforme: Consultor técnico externo.*
