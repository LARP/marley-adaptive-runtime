# Especificación Técnica y Gobernanza: Protocolo F7-D1 (Forensic Memory Profiling)

**Documento Oficial:** `docs/F7_D1_FORENSIC_SPEC_01.md`  
**Autorización:** Resolución Formal del Director de Proyecto (2026-09-09)  
**Evaluador:** Consejero Técnico Externo  
**Ejecutor:** Equipo de Arquitectura e Implementación / Antigravity AI  
**Fecha:** 9 de septiembre de 2026  
**Fase:** F7-D1 — Forensic Memory Profiling (Diagnóstico Observacional Aislado)  
**Baseline Relacionado:** F7-0 (720p Memory Feasibility Probe, Commit `83f138a`)  

---

## 1. Contexto y Relación con las Pruebas Anteriores

El presente protocolo forense surge directamente de la secuencia experimental de Marley Runtime:

```
[Phase F6: 480p Core 4x3 Multirun]
  │  12/12 Corridas PASS (Peak NVML <= 4,800 MB, Cadencia ~14.25 s/p)
  │  Certificación completa y congelamiento (Commit c6e4bfb)
  ▼
[Phase F7-0: 720p Memory Feasibility Probe]
  │  Workload: 1280x720, 33 frames, 5 pasos, Adaptive Engine
  │  Pico Físico NVML: 6,058.5 MB (Hard Gate <= 4,800 MB: ❌ FAIL, exceso +1,258.5 MB)
  │  PyTorch Allocated: 2,187.5 MB | PyTorch Reserved: 5,894.0 MB (Δ = 3,706.5 MB)
  │  Cadencia: 68.75 s/paso (degradación 4.82x vs F6) | VAE: 69.13 s
  │  Integridad: 0 NaNs/Infs (🟢 PASS) | Video MP4 Decodificable (🟢 PASS)
  │  Baseline F7-0 cerrado, congelado e inmutable
  ▼
[Phase F7-D1: Forensic Memory Profiling (AUTORIZADA)]
  │  Objetivo: Diagnosticar la anatomía del consumo de memoria y degradación temporal
  │  Regla de oro: CERO OPTIMIZACIONES DURANTE EL DIAGNÓSTICO
  │  Esclarecer causalidad para responder preguntas P1–P5 y evaluar hipótesis H1–H4
```

---

## 2. Marco Epistemológico Obligatorio

En cumplimiento de las directivas del Consejero Técnico y la resolución del Director, se establece una distinción tajante entre hechos empíricos e hipótesis:

### 2.1 Hechos Observados (Inmutables)
1. **Pico físico NVML:** 6,058.5 MB (rebasamiento de +1,258.5 MB).
2. **PyTorch Allocated final:** 2,187.5 MB.
3. **PyTorch Reserved final:** 5,894.0 MB.
4. **Diferencial interno:** $\Delta = 3,706.5\text{ MB}$.
5. **Cadencia DiT:** 68.75 s/paso promedio en 5 pasos.
6. **Telemetría de hardware:** 80 °C de temperatura GPU y 547 MHz de reloj.
7. **Integridad:** 0 NaNs/Infs y MP4 válido y reproducible.

### 2.2 Hipótesis de Trabajo (Sujetas a Validación Empírica en F7-D1)
* **H1 (Allocator):** La retención de bloques libres por el caching allocator de PyTorch es el factor dominante en la brecha entre tensores activos y memoria física.
* **H2 (WDDM):** La degradación de cadencia a 68.75 s/paso está vinculada a paginación de memoria compartida (Shared GPU Memory / WDDM paging) al superar el pool dedicado de 6 GB.
* **H3 (Comportamiento Térmico / Clock):** La frecuencia de 547 MHz responde a estrangulamiento térmico/energético o a ciclos de espera de memoria/PCIe.
* **H4 (Doble Residencia):** Existe retención temporal o duplicación de buffers durante los ciclos de transferencia en Adaptive/streaming.

---

## 3. Plan de Medición y Respuestas a las 5 Preguntas Diagnósticas

El script [`f7_d1_forensic_profiling.py`](../f7_d1_forensic_profiling.py) implementará la telemetría para responder:

1. **P1 — Localización Temporal del Pico NVML:**  
   Muestreo por paso y por fase (Prompt Encoding, DiT Prep, Step 1..5, VAE Decode) correlacionando tiempo de ejecución, timestamp y memoria física.
2. **P2 — Evento Detonante de PyTorch Reserved:**  
   Activación de historial de memoria estratégico (`torch.cuda.memory._record_memory_history`) acotado al Paso 1 (asignación espacial $90 \times 160$) y al instante pico, volcando snapshots nativos para análisis pericial.
3. **P3 — Composición Detallada de Memoria:**  
   Captura exhaustiva de `torch.cuda.memory_stats()` tras el denoising para discriminar segmentos activos, segmentos inactivos, bloques pequeños (<1 MB) y bloques grandes (>1 MB).
4. **P4 — Comportamiento de Streaming / Adaptive:**  
   Medición de la huella de memoria antes y después de transferencias de bloques en el paso inicial para verificar si hay coexistencia de capas en VRAM.
5. **P5 — Correlación Cadencia vs Presión de Memoria:**  
   Registro individual de duraciones $t_1, t_2, t_3, t_4, t_5$ contrastadas con el nivel de VRAM físico instantáneo para determinar si la degradación es constante o surge tras rebasar cierto umbral.

---

## 4. Prohibición Expresa de Optimizaciones

Durante F7-D1 está **terminantemente prohibido**:
* Alterar el tamaño de tile espacial o temporal del VAE.
* Forzar cuantización INT8 o NF4.
* Alterar la precisión (dtypes) de pesos o activaciones.
* Insertar llamadas a `torch.cuda.empty_cache()` o `gc.collect()` durante la inferencia como técnica paliativa.
* Modificar la lógica de decisión de AdaptiveEngine o prefetch.
* Podar capas o modificar la arquitectura del modelo.

---

## 5. Criterio de Éxito de F7-D1

F7-D1 **NO busca reducir el consumo por debajo de 4.800 MB**.  
Se considerará exitoso si y solo si aporta evidencia concluyente para caracterizar:
1. En qué momento exacto se genera el pico físico.
2. La evolución comparativa de `Allocated` vs `Reserved`.
3. La estructura de fragmentación interna del allocator.
4. La evolución temporal de la cadencia paso a paso.
5. La confirmación o refutación objetiva de las hipótesis H1 a H4.
