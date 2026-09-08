# Alerta Técnica F6-01: Sobrecarga de Memoria Host RSS en F6-C (Marley Async INT8)

**Para:** Director de Proyecto & Equipo de Arquitectura — Marley Runtime  
**Fecha:** 8 de septiembre de 2026  
**Severidad:** 🟡 **MODERADA (Monitoreo & Mitigación de Arquitectura)**  
**Componente Afectado:** `marley/ops/async_stream_int8.py` (`INT8BudgetedStreamer`)  
**Telemetría de Referencia:** [`logs/f6_condition_c_async_int8.json`](../logs/f6_condition_c_async_int8.json)  

---

## 1. Detección y Evidencia Empírica

Durante la ejecución comparativa de la tríada canónica de Phase F6 (480p, 33 frames, 30 pasos de difusión), se detectó una divergencia anómala en el consumo de memoria RAM de sistema (**Process Host RSS**) y en el pico físico de VRAM durante la **Condición C**:

| Benchmark Condition | Streaming Mode | Process Host RAM (RSS) | Delta vs Sync | Peak NVML VRAM | Delta VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F6-A (Baseline)** | Sync FP16 | **744.9 MB** | Base (0.0 MB) | 2,624.3 MB | Base |
| **F6-B (Async FP16)**| Async FP16 | **1,516.6 MB** | +771.7 MB | 2,698.0 MB | +73.7 MB |
| **F6-C (Async INT8)**| **Async INT8** | **3,805.5 MB** | ⚠️ **+3,060.6 MB** | **2,962.5 MB** | ⚠️ **+338.2 MB** |

> **Hallazgo Crítico:** Mientras que F6-B solo incrementó la RAM en ~770 MB por concepto de buffers de prefetch paginados (`pin_memory`), F6-C disparó el RSS del proceso a **3,805.5 MB** (+2.3 GB respecto a FP16), y la VRAM física subió a **2,962.5 MB** a pesar de que el payload transportado por bloque es la mitad (45.9 MB vs 88.6 MB).

---

## 2. Diagnóstico Causal Forense

La inspección profunda del código en [`marley/ops/async_stream_int8.py`](../marley/ops/async_stream_int8.py) revela dos causas directas:

### Causa 1: Duplicación de Tensores en RAM Host (Doble Residencia FP16 + INT8)
En `INT8BudgetedStreamer._prepare_host_weights()`:
```python
for block_idx, block in enumerate(self.blocks):
    for name, param in block.named_parameters():
        cpu_t = param.detach().to("cpu", dtype=self.dtype)
        original[name] = param.data                         # <--- RETIENE FP16 ORIGINAL (2.66 GB)
        ...
        wq_int8 = wq.to(torch.int8).contiguous()
        pinned_t = wq_int8.pin_memory()                     # <--- CREA COPIA PINNED INT8 (1.38 GB)
        scale_t = scale.half().contiguous().pin_memory()
```
1. **Retención de pesos originales:** Los 30 bloques DiT (`self.blocks`) permanecen instanciados en memoria RAM en precisión FP16 (**~2,658 MB**).
2. **Creación de réplica cuantizada:** Adicionalmente, el streamer crea un diccionario completo de tensores `pinned INT8` (**~1,377 MB**).
3. **Suma matemática de pesos en RAM:** $2,658\text{ MB (FP16)} + 1,377\text{ MB (INT8)} \approx \mathbf{4,035\text{ MB}}$.
4. Al no destruir ni vaciar los tensores FP16 originales de `block.parameters()`, el proceso coexiste con dos representaciones simultáneas de los mismos 30 bloques en la memoria host.

### Causa 2: Page-Locked Memory (`pin_memory()`) en Windows
* Los tensores INT8 y sus escalas se marcan explícitamente con `.pin_memory()`.
* En Windows (WDDM), la memoria pinned queda bloqueada en el *working set* del kernel y el administrador de memoria de Windows **no puede paginarla a disco**.
* Esto fuerza a que los ~1.4 GB de buffers INT8 residan estrictamente en RAM física no paginable.

### Causa 3: Overhead de Descuantización al Vuelo en GPU
* En GPU, `INT8BudgetedStreamer` reserva los slots de cómputo en FP16 (`self.gpu_slots`), los buffers de escala (`self._scale_gpu`) y ejecuta `_apply_dequant()` multiplicando el tensor entero por el vector de escala en cada paso.
* Los kernels de descuantización al vuelo y la presencia de buffers auxiliares de escala explican los **+264.5 MB adicionales de pico físico de VRAM** (2,962.5 MB en F6-C vs 2,698.0 MB en F6-B).

---

## 3. Matriz de Riesgo Operativo

| Entorno de Usuario | Impacto del RSS de 3.8 GB | Nivel de Riesgo |
| :--- | :--- | :---: |
| **Laptops de 24 GB / 32 GB RAM** | Absorbido holgadamente por el sistema operativo (quedan >18 GB libres). No hay contención ni degradación. | 🟢 **Bajo** |
| **Laptops de 16 GB RAM (Presupuesto Ajustado)** | Si el usuario tiene Chrome/IDE abiertos (8–10 GB en uso), una subida de 3.8 GB puede provocar **working set trimming** y forzar a Windows a hacer *paging* de otros procesos a disco, aumentando la latencia de transferencia PCIe. | 🟡 **Medio / Alerta** |

---

## 4. Plan de Remediación y Mitigación de Arquitectura

Para eliminar esta duplicación y optimizar la huella de F6-C / Phase F7:

### Mitigación 1: Mutación In-Place de Tensores en Host (Quick-Win)
Una vez cuantizados los bloques a INT8, liberar inmediatamente los tensores FP16 originales reemplazando `param.data` por tensores vacíos (`torch.empty(0)`) y forzar recolección de basura:
```python
# Liberar FP16 host original tras cuantizar
for name, param in block.named_parameters():
    param.data = torch.empty(0, dtype=self.dtype)
gc.collect()
```
* **Ahorro inmediato:** **-2,658 MB** de memoria host.
* **RSS resultante estimado:** Reducción de 3,805 MB a **~1,200–1,400 MB**.

### Mitigación 2: Pre-Cuantización Serializada Offline (Arquitectura Final)
En lugar de cuantizar al vuelo en CPU durante la inicialización (`__init__`), almacenar los pesos de los bloques DiT directamente en formato INT8 cuantizado (`wan2.1_dit_int8.safetensors`).
* Elimina el tiempo de cuantización inicial (~15 s).
* Elimina por completo los picos de asignación temporal en CPU.

---

## 5. Conclusión y Veredicto de la Alerta

* **Veredicto:** 🟡 **ALERTA REGISTRADA SIN BLOQUEO DE F6.**
* **Justificación:** F6-C cumplió todos sus gates funcionales y de seguridad (0 NaNs, MP4 generado correctamente, VRAM física de 2,962.5 MB muy por debajo del límite de 4,800 MB).
* **Acción:** Incorporar la **Mitigación 1 (In-Place Host Tensor Eviction)** como tarea de optimización en la transición hacia Phase F7.
