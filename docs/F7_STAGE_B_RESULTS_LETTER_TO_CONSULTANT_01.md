# CARTA DEL DIRECTOR DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — RESULTADOS OFICIALES F7 STAGE B (30 PASOS @ 720p)

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Comunicación de resultados oficiales de F7 Stage B (Reevaluación Completa 720p / 33 Frames / 30 Pasos)  
**Referencia del Protocolo:** [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md) §6  
**Artefactos Anexos:**  
- Telemetría bruta per-step: [`logs/f7_stage_b_full_validation_telemetry.json`](../logs/f7_stage_b_full_validation_telemetry.json)  
- Reporte formal de ejecución: [`docs/F7_STAGE_B_FULL_VALIDATION_REPORT_01.md`](F7_STAGE_B_FULL_VALIDATION_REPORT_01.md)  
- Script ejecutable de Stage B: [`f7_stage_b_full_validation.py`](../f7_stage_b_full_validation.py)  
- Video generado (33 frames, 720p): [`logs/f7_stage_b_1280x720_33f_30st_20260910_015326.mp4`](../logs/f7_stage_b_1280x720_33f_30st_20260910_015326.mp4)  

---

## 1. Declaración formal de la dirección

Estimado Consejero:

En estricta observancia de su resolución sobre la Etapa C y del **Protocolo Prerregistrado F7-720p (§6)**, la dirección técnica ha ejecutado la corrida de reevaluación completa de 30 pasos de difusión a resolución nativa **1280×720 y 33 frames**.

La ejecución se llevó a cabo respetando íntegramente la configuración inmutable prerregistrada:
- Cero alteraciones algorítmicas adicionales a la intervención ya validada (drenaje de inicio de paso + liberación quirúrgica en la costura condicional/incondicional).
- Monitorización física a alta frecuencia con NVML (20 ms) y muestreo pasivo a 1 Hz en las fases estáticas ($S_0$, $S_3$, $S_4$).
- Controles de calibración A y B (~500 MB).
- Decodificación VAE tileada en bfloat16 y exportación a MP4.

Comunico formalmente el resultado empírico observado y la clasificación resultante sin intentar atenuar, reclasificar ni eludir las compuertas de seguridad física establecidas.

---

## 2. Cuadro de mando ejecutivo (Scorecard oficial)

| Compuerta / Criterio | Límite Prerregistrado | Valor Empírico Medido | Clasificación Oficial |
| :--- | :---: | :---: | :---: |
| **Gate A (Seguridad Física Global NVML)** | $\le \mathbf{4,800.0\text{ MB}}$ | **`4,996.0 MB`** | 🔴 **FAIL** ($+196.0\text{ MB}$) |
| **Gate B (Eficiencia Incremental $\Delta_{S0}$)** | $\le \mathbf{3,850.0\text{ MB}}$ | **`4,046.3 MB`** | 🔴 **FAIL** ($+196.3\text{ MB}$) |
| **Delta Operacional ($\Delta_{S1} = S2 - S1$)** | N/A (Analítico) | **`3,912.5 MB`** | Observado |
| **Cadencia de Inferencia** | Nominal (~60 s/paso) | **`58.47 s/paso`** (Total: 1,754.2 s) | 🟢 **NOMINAL** |
| **Decodificación VAE Tiled** | 33 frames valid, 0 NaNs | **75.70 s** (1280×720, 0 NaNs) | 🟢 **PASS** |
| **Generación de Video MP4** | Archivo reproducible | `logs/f7_stage_b_1280x720_33f_30st_20260910_015326.mp4` | 🟢 **EXITOSO** |
| **Clasificación según Matriz (§3)** | Escenario 4 | — | 🔴 **COMPLETE REJECTION** |

---

## 3. Serie temporal de hitos de memoria

1. **Línea base pasiva en reposo ($S_0$):**
   - Media: **949.8 MB** (mín 902.4 / máx 1,023.9 / spread 121.4 MB).
2. **Control A (~500 MB):**
   - Delta observado: **500.0 MB** exactos (Allocated: 500.0 MB). Sensor calibrado.
3. **Línea base operacional ($S_1$):**
   - Consumo NVML tras inicialización completa del pipeline: **1,083.5 MB**.
4. **Fase de Inferencia DiT (30 pasos):**
   - Paso 1: 4,813.4 MB (Reserved: 3,692.0 MB)
   - Paso 6 (Pico global $S_2$): **4,996.0 MB** (Reserved: 3,772.0 MB)
   - Pasos 7 a 23: Oscilación estable en el rango 4,870 – 4,940 MB
   - Pasos 24 a 30: 4,714 – 4,773 MB (Reserved final: 3,766.0 MB)
5. **Decodificación VAE Tiled (33 frames):**
   - Completada en **75.70 s** sin incremento sobre el pico DiT.
6. **Meseta post-workload ($S_3$, 120 s @ 1 Hz):**
   - Media: **1,650.9 MB** (spread 105.7 MB).
7. **Control B (~500 MB):**
   - Delta observado: **508.3 MB** (Allocated: 599.6 MB).
8. **Observación final ($S_4$, 60 s @ 1 Hz):**
   - Media: **1,667.5 MB** (spread 75.1 MB).

---

## 4. Hallazgos técnicos y dictamen de la dirección

A partir de la evidencia experimental indiscutible, la dirección arriba a las siguientes determinaciones técnicas:

1. **El mecanismo algorítmico es estable y funcional:**
   - La inferencia corrió de principio a fin a una cadencia óptima de **58.47 s/paso**.
   - La reserva de memoria de PyTorch se estabilizó de manera plana en **~3,766 MB**, confirmando que la intervención de liberación en costura (`seam release`) y de arranque de paso previene con total éxito la acumulación interna de tensores en la GPU.
   - La integridad del video generado es total (0 NaNs, MP4 de 33 frames válido).
2. **El sistema supera el límite físico inmutable:**
   - A pesar de que la asignación de PyTorch estuvo contenida, la carga total del dispositivo alcanzó **4,996.0 MB**, superando el hard gate de **4,800.0 MB** por **196.0 MB**.
   - Asimismo, el delta neto sobre el reposo ($\Delta_{S0} = 4,046.3\text{ MB}$) excedió la cota de Gate B (3,850.0 MB).
3. **Veredicto vinculante:**
   - En conformidad con la Regla de Oro de Gobernanza del protocolo (§3), **la resolución 1280×720 (720p) sobre el hardware NVIDIA RTX 3050 Laptop (6 GB) bajo Windows 11 WDDM NO puede ser certificada como PASS**.
   - La clasificación oficial de la corrida es **🔴 FAIL / COMPLETE REJECTION**, y el estado formal del hito F7-D5 en la gobernanza del proyecto se ratifica definitivamente como **🔴 NEGATIVE**.

La dirección no presentará objeciones ni solicitudes de reclasificación retroactiva. Ponemos a su disposición la totalidad de la telemetría, el informe y el video generado para su dictamen final y cierre de la campaña F7.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
