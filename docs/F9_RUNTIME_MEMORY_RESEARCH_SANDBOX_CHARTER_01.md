# F9 — Runtime Memory Research Sandbox · Charter

**Documento:** `F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md`
**Fecha:** 10 de septiembre de 2026
**Autor:** Dirección del Proyecto — Marley Runtime
**Autoridad externa:** Consejero Técnico — Runtime y Gestión de Memoria
**Estado:** 🟡 **GENERADA / PREREGISTRADA — NO EJECUTADA**
**Referencias:** Carta del Consejero de generación de F9 (`docs/private/F9_CONSULTANT_LETTER_GENERATION_01.md`); cierre de F7 (`F7_STAGE_B_FULL_VALIDATION_REPORT_01.md`); cierre de F8 (`F8_SCREENING_AB_REPORT_01.md`); metrología F7-D6 (`F7_D6_METRIC_ATTRIBUTION_REPORT_01.md`); ROADMAP §2 (modelo de tres capas).

---

## 1. Resolución de cierre previo

* **F7 — 720p Feasibility:** 🔴 **CERRADA (FAIL)**. No se reabre.
* **F8 — Quantization Screening:** 🔴 **CERRADA (RESULTADO NULO)**. No se reabre.
* **480p / 33 frames / 30 steps:** producción certificada e inmutable.
* **720p / 33 frames / 30 steps:** experimental, no certificado.

La línea de **cuantización sobre Wan queda descartada** como vía de solución del déficit.
La siguiente investigación es de **gestión dinámica de memoria / runtime**, no de precisión numérica.

---

## 2. Misión de F9

Investigar mecanismos de gestión dinámica de memoria capaces de reducir el **pico físico de
VRAM** mediante el control de:

* lifetime de tensores;
* coexistencia temporal de buffers;
* reutilización de memoria;
* workspaces;
* eviction / residency;
* scheduling;
* prefetch / overlap;
* recomputación y otras técnicas de control de residencia.

**Criterio de éxito:** producir conocimiento **reproducible** sobre cómo controlar la residencia y
coexistencia de memoria en un runtime generativo bajo un límite físico estricto. El objetivo no es
"menos VRAM en Stable Diffusion", sino descubrir **fenómenos de runtime** cuya abstracción sea
razonablemente transferible a un grafo DiT/video como Wan2.1-T2V-1.3B.

---

## 3. Sandbox experimental

* Stable Diffusion se usará como **laboratorio rápido de hipótesis**, nunca como sustituto de la
  validación final en Wan2.1.
* **Requisito de similitud estructural (definición operacional, obs. #5):** el sandbox debe emplear
  preferentemente un grafo **basado en DiT** (p. ej. SD3 / MMDiT) y no un UNet clásico, para que los
  fenómenos de *lifetime / coexistencia / workspace* sean estructuralmente análogos a Wan.
  La elección concreta de modelo y pipeline se fijará en el preregistro de cada subtarea F9-X.
* Superar el sandbox **no** implica transferibilidad automática a Wan.

---

## 4. Subtareas

Las subtareas se generan **únicamente cuando la evidencia de la anterior lo justifique**.

| Subtarea | Nombre | Estado |
| :--- | :--- | :--- |
| **F9-0** | Preregistered Memory Peak Attribution Diagnostic | 🟡 **Preregistrada — pendiente de autorización de ejecución** |
| **F9-1** | Primer mecanismo seleccionado por F9-0 (placeholder inactivo) | ⚪ Placeholder |
| **F9-2** | Screening experimental del mecanismo (placeholder inactivo) | ⚪ Placeholder |
| **F9-3** | Transferability / Wan diagnostic (placeholder inactivo) | ⚪ Placeholder |
| **F9-4** | Validación aislada en Wan — solo si corresponde (placeholder inactivo) | ⚪ Placeholder |

> F9-1…F9-4 son **placeholders estructurales**, no tareas activas. Su contenido se redacta y
> congela tras el cierre de F9-0 (obs. #8).

---

## 5. Ruta estratégica

```text
F9-0  Atribución del pico físico
  ↓   Identificación del fenómeno dominante
F9-X  Mecanismo en sandbox
  ↓   G1–G6
     Evidencia de transferibilidad (G3)
  ↓
     Una validación aislada en Wan
  ↓
     PASS / FAIL / G4-C
  ↓
     Decisión del Comité
```

> **MEDIR → ATRIBUIR → FORMULAR → EXPERIMENTAR → TRANSFERIR → VALIDAR → DECIDIR**

Esta estructura sustituye el ensayo indiscriminado de optimizaciones por un ciclo de hipótesis
falsables.

---

## 6. Gate de transferibilidad G3

Ningún mecanismo se promociona a Wan sin superar tres niveles:

1. **Fenómeno** — formulado como fenómeno de memoria (lifetime, coexistencia, workspace, residency,
   scheduling…). No basta con "esta implementación consume menos VRAM".
2. **Evidencia estructural** — el fenómeno debe existir conceptualmente también en el grafo de
   Wan2.1; cuando sea viable, se realizará una medición diagnóstica barata en Wan.
3. **Predicción falsable** — antes de la validación Wan se registrará una predicción cuantitativa
   (p. ej. "el mecanismo debería reducir el evento X en Y–Z MB"). Fuera del intervalo preregistrado
   ⇒ hipótesis fallida, aunque exista beneficio accidental.

---

## 7. Gates experimentales G1–G6

| Gate | Criterio |
| :--- | :--- |
| **G1 — Reproducibilidad** | N ≥ 3 corridas independientes en screening; N ≥ 5 recomendado para promoción. |
| **G2 — Explicación mecanística** | Evidencia de qué fenómeno de memoria se controla. |
| **G3 — Transferibilidad** | fenómeno → evidencia estructural → predicción falsable → Wan. |
| **G4 — Magnitud** | **G4-A:** ≥15 % de reducción reproducible. **G4-B:** <15 % pero reproducible y distinguible del ruido con mecanismo claro. **G4-C:** mecanismo bien caracterizado sin reducción útil del pico en Wan. El objetivo práctico es evaluar si cierra el déficit histórico de **F7-D5 = +296.1 MB** sobre el gate (referencia única verificada en el preregistro F9-0 §2.7). |
| **G5 — Correctness / Quality** | Sin corrupción, NaN/Inf ni regresión de calidad no aceptada (PSNR / SSIM / LPIPS / perceptual cuando corresponda). |
| **G6 — Coste / Portabilidad** | latencia, throughput, complejidad, riesgo de integración y esfuerzo estimado de portabilidad a Wan. |

---

## 8. Regla de composición

Los efectos de varios mecanismos **no se suman teóricamente**. Solo son componibles si:

1. afectan eventos de pico **diferentes**;
2. existe evidencia de que **no compiten** por la misma memoria;
3. la combinación se **mide experimentalmente**.

---

## 9. Promoción final hacia Wan

Un mecanismo que supere los gates recibe **un único slot de validación aislada** en
Wan2.1-T2V-1.3B con el pipeline canónico **1280×720 / 33 frames / 30 steps**, manteniendo mismo
seed, modelo, workload, gate y métrica primaria **NVML**. No se permite reajustar la técnica tras
el resultado para presentarla como la validación original. Una nueva optimización exige una nueva
hipótesis y un nuevo experimento.

---

## 10. Gobernanza

* F9-0 es la **puerta de entrada obligatoria** y tiene naturaleza de **diagnóstico, no optimización**.
* Ninguna corrida se ejecutará sin autorización explícita de la Dirección.
* F7 y F8 permanecen cerradas e inmutables.
* Los documentos de correspondencia residen en `docs/private/` y no se versionan.

---

*Marley Runtime is dedicated in loving memory to Marley 🐾.*
