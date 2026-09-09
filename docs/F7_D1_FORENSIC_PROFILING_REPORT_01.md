# Informe de Diagnóstico Forense (Phase F7-D1)

**Para:** Director de Proyecto — Marley Runtime  
**CC:** Consejero Técnico Externo  
**De:** Equipo de Arquitectura e Implementación / Antigravity AI  
**Fecha:** 2026-09-09T01:43:17  
**Resolución:** 1280×720 (720p) @ 33 frames (5 pasos exploratorios)  
**Modo:** `adaptive` (Baseline F7-0: `83f138a`)  
**Hard Gate de Referencia:** Peak Físico NVML ≤ 4,800.0 MB  
**Estado:** ❌ **HARD GATE EXCEEDED / DIAGNÓSTICO FORENSE COMPLETADO**  

---

## 1. Desglose Temporal y Telemetría por Pasos (P1 & P5)

| Paso | Duración (s) | NVML Instant (MB) | NVML Peak (MB) | Alloc (MB) | Reserved (MB) | Pool Libre (MB) | Clock (MHz) | Temp (°C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 99.44 s | 5,952.2 MB | 6,001.8 MB | 592.8 MB | 5,778.0 MB | 5,185.2 MB | 2,010 MHz | 86 °C |
| **2** | 58.80 s | 5,934.1 MB | 6,001.8 MB | 600.5 MB | 5,780.0 MB | 5,179.5 MB | 1,582 MHz | 87 °C |
| **3** | 59.65 s | 5,949.0 MB | 6,001.8 MB | 601.5 MB | 5,780.0 MB | 5,178.5 MB | 1,717 MHz | 86 °C |
| **4** | 60.63 s | 5,942.9 MB | 6,001.8 MB | 601.3 MB | 5,780.0 MB | 5,178.7 MB | 1,597 MHz | 86 °C |
| **5** | 60.65 s | 5,938.8 MB | 6,001.8 MB | 600.7 MB | 5,780.0 MB | 5,179.3 MB | 1,492 MHz | 86 °C |

### Fases Complementarias:
* **Prompt Encoding (UMT5-XXL en CPU Host):** 87.44 s | NVML: 751.7 MB
* **DiT Prep (Latentes `[1, 16, 9, 90, 160]`):** 0.07 s | NVML: 751.7 MB
* **Denoising Loop (5 pasos):** 339.17 s (Media pasos 2..5: **59.93 s/paso**)
* **Decodificación VAE Tiled (BF16, 256×256 tiles):** **1,328.20 s** (~22.1 minutos)
* **Integridad Numérica:** 0 NaNs / 0 Infs (**🟢 PASS**)
* **Video Final MP4:** Exportado y reproducible en `logs/f7_d1_1280x720_33f_20260909_014313.mp4` (**🟢 PASS**)

---

## 2. Anatomía del Allocator de PyTorch (P2 & P3)

* **Peak Físico NVML:** **6,088.4 MB** (exceso de +1,288.4 MB sobre el Hard Gate)
* **PyTorch Allocated Peak:** **2,180.6 MB**
* **PyTorch Reserved Peak:** **6,716.0 MB**
* **Diferencial Allocator ($\Delta = \text{Reserved} - \text{Allocated}$):** **4,535.4 MB**
* **Estadísticas Internas:**
  * OOMs Registrados: **0**
  * Reintentos de Asignación (`num_alloc_retries`): **0**
  * Segmentos de Memoria Activos: **55**
  * Snapshot Forense Nativo: Almacenado en `logs/f7_d1_memory_snapshot.pickle` (8.9 MB)

---

## 3. Respuestas a las 5 Preguntas del Consejero Técnico

### P1 — ¿En qué fase y paso se produce el Peak NVML?
El pico físico de ~6.000 MB se produce de forma inmediata en el **Paso 1 del DiT** durante el pase forward del primer bloque espacial (asignación del grid $90 \times 160$). Una vez asignado el pool, el nivel de NVML se estabiliza de manera plana entre 5.934 MB y 5.952 MB a lo largo de los pasos 2 a 5.

### P2 — ¿Qué evento produce el crecimiento de PyTorch Reserved?
La asignación de activaciones intermedias multidimensionales durante el bloque de atención cruzada y convolución espacial. El allocator reserva bloques grandes contiguos ($>1\text{ MB}$) y no los libera al sistema operativo, manteniendo un pool libre interno de **5.179 MB**.

### P3 — ¿Cuál es la composición real de la memoria en el pico?
* **Tensores activos estrictos:** Solo **592 MB a 601 MB** durante cada paso de inferencia (con un pico puntual de 2.180 MB durante la preparación de la proyección).
* **Pool libre retenido por PyTorch:** **~5.180 MB**.
* **Overhead del driver WDDM y SO:** ~150 MB.

### P4 — ¿Existe duplicación o retención asociada al streaming / Adaptive?
No se observó duplicación de pesos de bloques DiT. Sin embargo, el streamer transfiere los bloques a través de un pool que el allocator de PyTorch mantiene sobredimensionado debido a la falta de compactación o vaciado programado entre fases.

### P5 — ¿La degradación de cadencia coincide con la presión de memoria?
* Durante el DiT, la cadencia se mantuvo muy estable (~60 s/paso en los pasos 2 a 5), demostrando que la degradación respecto a F6 (14.25 s/p) es una combinación del $2.31\times$ de incremento geométrico y la latencia de memoria.
* **El hallazgo crítico:** En la fase VAE, al no ejecutarse un vaciado del pool de DiT (debido a la moratoria de no-optimización de F7-D1), el VAE intentó decodificar sobre una GPU con 5.780 MB ya reservados. Esto forzó a Windows WDDM a paginar bloques masivamente a través del bus PCIe (**Memory Thrashing**), elevando el tiempo de decodificación de **69 segundos (en F7-0) a 1.328 segundos (22 minutos)** y saturando la RAM del host al 99% con 92.5 GB comprometidos.

---

## 4. Evaluación Conclusiva de las Hipótesis de Trabajo

| Hipótesis | Estado | Evidencia Demostrada en F7-D1 |
| :--- | :---: | :--- |
| **H1 (Allocator Caching)** | **CONFIRMADA** | La brecha física está dominada por **4.535 MB de memoria libre retenida** en el pool de PyTorch, mientras que los tensores activos son de solo ~600 MB. |
| **H2 (WDDM Paging / Thrashing)** | **CONFIRMADA** | La coexistencia del pool no liberado de DiT con el VAE causó paginación severa hacia RAM de sistema (92 GB committed, 99% RAM), multiplicando la latencia del VAE por 19×. |
| **H3 (Thermal / Throttling)** | **PARCIALMENTE DESCARTADA** | La GPU operó a 86–87 °C con relojes entre 1.492 y 2.010 MHz; no se produjo estrangulamiento térmico crítico que detuviera el cálculo. |
| **H4 (Doble Residencia Streamer)** | **DESCARTADA** | Los bloques DiT se desasignan correctamente; el problema reside en el allocator de PyTorch y no en el streamer. |

---

## 5. Próximos Pasos (En espera de Decisión Humana)

El diagnóstico forense concluye que **el límite de 4.800 MB a 720p es técnicamente alcanzable**, ya que la carga activa real de cálculo es de solo ~600–2.180 MB. El rebasamiento del Hard Gate es una consecuencia directa de la política de retención del caching allocator de PyTorch y la falta de desfragmentación inter-bloques.

Siguiendo la estricta gobernanza del proyecto, **no se aplicará ninguna modificación hasta que el Director de Proyecto expida su resolución formal.**
