# Documento de Diseño e Implementación: Phase F7-1 (Attention Sequence Chunking)

**Documento Oficial:** `docs/F7_1_ATTENTION_CHUNKING_PLAN_01.md`  
**Autorización:** Resolución del Director de Proyecto (Redacción autorizada; Implementación pendiente)  
**Evaluador:** Consejero Técnico Externo  
**Ejecutor:** Equipo de Arquitectura e Implementación / Antigravity AI  
**Fecha:** 9 de septiembre de 2026  
**Estado:** 🟡 **PLAN REDACTADO — EN ESPERA DE AUTORIZACIÓN PARA EJECUCIÓN**  
**Workload Objetivo:** 1280×720 (720p), 33 frames, 5 pasos exploratorios  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (WDDM 3.1)  
**Hard Gate:** Peak Físico NVML ≤ 4,800.0 MB  

---

## 1. Motivación y Evidencia Empírica de F7-D1

En el diagnóstico forense F7-D1 (`docs/F7_D1_FORENSIC_PROFILING_REPORT_01.md`), se confirmó de manera objetiva:
* **Tensores activos estacionarios:** ~600 MB durante la inferencia DiT.
* **Pool libre retenido por PyTorch:** 5,179 MB (Reservado: 5,780 MB a 6,716 MB).
* **Peak Físico NVML:** 6,088.4 MB (rebasamiento de +1,288.4 MB).
* **Origen temporal:** Ocurre puntualmente en el **Paso 1 del DiT** al computar la atención espacial multidimensional sobre la secuencia de $14,400$ tokens ($90 \times 160$).

La técnica **Attention Sequence Chunking** aborda este fenómeno de manera preventiva: reduce el working set en el instante mismo del cálculo, evitando que el caching allocator solicite bloques masivos a Windows WDDM.

---

## 2. Restricciones de Gobernanza y Aislamiento Monovariable

Bajo las órdenes directivas del Director de Proyecto:
1. **Un solo grado de libertad:** Únicamente se modifica el forward de atención para computar en bloques de queries.
2. **Prohibiciones expresas:**
   * Cero modificaciones al VAE.
   * Cero cuantización adicional (INT8/NF4).
   * Cero alteraciones a `AdaptiveEngine` ni prefetch.
   * Cero dynamic chunking variable por ahora.
   * Cero hooks manuales al allocator.
3. **Inmutabilidad de Baselines:** F6 y F7-0 permanecen congelados como referencias absolutas.

---

## 3. Formulación Matemática y Selección del Tamaño de Chunk ($C = 2.048$)

### 3.1 Algoritmo Block-wise SDPA
Dada la secuencia de entrada con $S = 14,400$ tokens y dimensión $D$:
$$Q \in \mathbb{R}^{B \times H \times S \times D}, \quad K, V \in \mathbb{R}^{B \times H \times S \times D}$$

Se particiona $Q$ en $N_c = \lceil 14.400 / 2.048 \rceil = 8$ bloques:
$$Q = [Q_1, Q_2, \dots, Q_8]$$
donde $Q_1 \dots Q_7$ contienen 2,048 tokens y $Q_8$ contiene los 64 tokens residuales.

Para cada bloque $i \in \{1, \dots, 8\}$:
$$\text{Out}_i = \text{SDPA}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{D}}\right) V$$
$$\text{Out} = \text{concat}([\text{Out}_1, \dots, \text{Out}_8], \text{dim}=\text{seq})$$

### 3.2 Justificación del Tamaño $C = 2.048$
1. **Reducción Teórica:** El working set de activaciones intermedias por pase de atención se reduce en un **$85,8\%$** ($1 - 2.048 / 14.400$).
2. **Eficiencia en Tensor Cores:** 2,048 es múltiplo de 256, garantizando máxima ocupación en los SMs de la arquitectura Ampere.
3. **Memoria Temporal Estimada:** El buffer temporal desciende de ~2.180 MB a **$< 350\text{ MB}$**, permitiendo que toda la operación de atención opere dentro de un presupuesto seguro inferior a 3.000 MB.

---

## 4. Matriz de Auditoría en 12 Dimensiones

| Dimensión | Métrica / Instrumento | F7-0 (Baseline) | Meta de Aceptación F7-1 |
| :--- | :--- | :---: | :---: |
| **1. Peak Físico NVML** | Muestreo a 25 ms vía `pynvml` | 6.058,5 MB | **$\le 4.800,0\text{ MB}$ (Hard Gate PASS)** |
| **2. PyTorch Allocated** | `torch.cuda.memory_allocated()` | 2.187,5 MB | $\le 1.200,0\text{ MB}$ en pico |
| **3. PyTorch Reserved** | `torch.cuda.memory_reserved()` | 5.894,0 MB | $\le 4.000,0\text{ MB}$ |
| **4. Host RSS** | `psutil` memoria del proceso | ~2.150 MB | $< 4.000\text{ MB}$ sin thrashing |
| **5. Tiempo Total** | Cronómetro wall-clock | 521,93 s | $< 600\text{ s}$ |
| **6. Cadencia DiT** | Media de pasos $t_2 \dots t_5$ | 68,75 s/paso | $\le 75,0\text{ s/paso}$ |
| **7. Sobrecosto Atención** | Latencia específica por bloque DiT | Línea base SDPA | Sobrecosto $< 10\%$ |
| **8. Integridad Numérica** | Tensores libres de NaN/Inf | 0 NaNs | **0 NaNs / 0 Infs (100%)** |
| **9. Integridad del Video** | Validador de contenedor MP4 | MP4 válido | **MP4 Decodificable y reproducible** |
| **10. Similitud Latente** | Cosine similarity en Paso 1 vs F7-0 | 1.0 (Identidad) | $\ge 0.9990$ |
| **11. Calidad Visual** | Nitidez y ausencia de artefactos | Video F7-0 | Preservada, sin banding |
| **12. Coherencia Temporal**| Estabilidad visual cuadro a cuadro | 33 frames | Sin parpadeos ni discontinuidades |

---

## 5. Diseño de Componentes Experimentales

* **Módulo Operacional Aislado:** [`marley/ops/chunked_attention.py`](file:///d:/Gemini_Admin_Tool/marley_720p/marley/ops/chunked_attention.py)
* **Runner Experimental de Prueba:** [`f7_1_attention_chunking_probe.py`](file:///d:/Gemini_Admin_Tool/marley_720p/f7_1_attention_chunking_probe.py)
* **Principio de No Invasión:** El código de producción y los pipelines existentes no se modifican. El runner aplicará un monkey-patching acotado y reversible exclusivamente durante la ejecución del benchmark F7-1.

---

## 6. Estado y Solicitud

El presente plan ha sido redactado conforme a la autorización expresa del Director de Proyecto.

> **Gobernanza:** No se ha escrito código de ejecución ni alterado el runtime. Se somete el plan a la revisión de la Dirección y el Consejero Técnico para decidir si se autoriza su posterior implementación y ejecución experimental.
