# Test G — Model-Level CPU Offload (`model_cpu_offload`)

**Fase:** F0.5 — Sprint 1 (Experimentos de Costo Mínimo)  
**Fecha:** 2026-09-08T06:49:12Z  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (VRAM física total: 6,144 MB · Gate: 4,800 MB)  
**Modelo:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`  
**Resolución / Frames:** 832×480 (480p) · 17 frames  
**Precisión:** DiT: FP16 (`torch.float16`) · VAE: FP32 (`torch.float32`)  
**Estrategia de Offload:** `enable_model_cpu_offload()`  
**Video resultante:** [`logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910.mp4`](file:///d:/Gemini_Admin_Tool/marley_720p/logs/f0_480p_17f_fp16_vae_fp32_model_cpu_20260908_034910.mp4)  

---

## 1. Resumen Ejecutivo y Veredicto

| Criterio | Meta / Gate | Resultado Test G | Veredicto |
| :--- | :---: | :---: | :---: |
| **Peak NVML físico durante inferencia** | ≤ 4,800 MB | **5,861 MiB** (muestreado en vivo vía `nvidia-smi`) | ❌ **FAIL (Exceso +1,061 MB)** |
| **Pico Asignado por PyTorch (`max_memory_allocated`)** | < 6,144 MB | **13,036.9 MB (~13.0 GB virtual)** | ❌ **Paginación WDDM masiva** |
| **Tiempo Denoising DiT (30 pasos)** | — | **212s (3m 32s · ~6.3 s/step)** | 🟢 **Excelente (-28.6% vs F0)** |
| **Tiempo VAE Decode (17 frames)** | < 60s | **484s (8m 04s)** | ❌ **Degradación severa (+50% vs F0)** |
| **Tiempo Total Wall-Clock** | ≤ 600s | **696.3s (11m 36s)** | ❌ (+76.5s vs F0) |
| **Estabilidad / Crash** | Cero OOM crashes | 0 OOM crashes (WDDM absorbió) | 🟢 PASS estructural |

**Veredicto Final Test G:** **NO SUPERA EL GATE.**  
`model_cpu_offload` acelera notablemente el DiT (de 9.93s a 6.3s/paso), pero **agrava el cuello de botella del VAE** si este se mantiene en `float32`.

---

## 2. Telemetría Comparativa: Baseline F0 vs. Test G

| Checkpoint / Métrica | Baseline F0 (`cpu` - sequential) | Test G (`model_cpu`) | Delta / Observación |
| :--- | :---: | :---: | :--- |
| **Offload granularity** | Layer-by-layer | Submodel-by-submodel | — |
| **Denoising DiT (30 steps)** | 297s (9.93 s/step) | **212s (6.33 s/step)** | **-85s (-28.6% tiempo DiT)** 🟢 |
| **VAE decode (17 frames)** | 322s | **484s** | **+162s (+50.3% más lento)** ❌ |
| **Pico virtual PyTorch** | 7,914 MB (reserved) | **13,036.9 MB (allocated)** | **+5,122 MB de activaciones** ❌ |
| **Pico NVML físico real** | 5,451 MB | **5,861 MB** | **+410 MB sobre F0** (95.4% VRAM física) |
| **NVML post-inferencia** | 5,451 MB (retenido) | 902.4 MB (liberado) | Hook liberó el submodelo al terminar |
| **Video generado** | 16 frames @ 480p | 17 frames @ 480p | Íntegro, sin artefactos |

---

## 3. Hallazgos Críticos de Ingeniería

### 3.1 La trampa de la decodificación VAE completa en GPU con FP32
En el baseline F0 (`sequential_cpu_offload`), las capas del VAE se descargaban y subían una a una por PCIe. Aunque esto era lento (322s), limitaba la concurrencia de tensores de trabajo.  
En Test G (`model_cpu_offload`), el submodelo completo VAE reside en GPU. Al decodificar el tensor latente (17 frames) en **`float32`**, las 33 capas convolucionales 3D causales generaron una cascada de activaciones que demandó **13.036 MB de memoria de trabajo virtual**.

### 3.2 Saturación de PCIe por Paginado WDDM
Al solicitar 13.0 GB en una GPU con 6.1 GB físicos:
1. El subsistema WDDM de Windows paginó **más de 7 GB de memoria a la RAM del sistema**.
2. Cada operación convolucional 3D tuvo que mover tensores gigantescos a través del bus PCIe de la laptop.
3. Esto provocó el colapso del rendimiento: el VAE tardó **8 minutos**, empeorando la marca de F0.

### 3.3 Nota metodológica sobre el sampling de NVML
Al usar `model_cpu_offload`, el hook de Diffusers transfiere el VAE de regreso a la CPU en el instante milimétrico en que finaliza `pipe(...)`.  
Por esta razón, un snapshot posterior a la inferencia (`post-inference`) muestra un engañoso `902 MB`. Sin embargo, el muestreo en vivo con `nvidia-smi` capturó el pico real de **5,861 MiB** y PyTorch registró `peak_torch_allocated_mb: 13036.9 MB`.

---

## 4. Conclusión e Implicaciones para el Siguiente Test

1. **Se descarta `model_cpu_offload` con VAE en FP32:** No es una solución viable para el gate.
2. **Confirmación definitiva del diagnóstico:** El origen del problema no es la gestión de offload del DiT, sino **la precisión FP32 del VAE**.
3. **Paso Mandatorio Inmediato: Test H (`bfloat16` VAE):**  
   Al ejecutar el VAE en `bfloat16`:
   * Las activaciones de 13 GB se reducen teóricamente a la mitad (~6.5 GB).
   * Combinado con `sequential_cpu_offload` (o `model_cpu_offload`), se evitará la paginación masiva de WDDM.
   * La RTX 3050 procesará el VAE con núcleos Tensor nativos.
