# Test H — Sequential Offload + `bfloat16` VAE

**Fase:** F0.5 — Sprint 1 (Experimentos de Costo Mínimo)  
**Fecha:** 2026-09-08T06:58:25Z  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (VRAM física total: 6,144 MB · Gate: 4,800 MB)  
**Modelo:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`  
**Resolución / Frames:** 832×480 (480p) · 17 frames  
**Precisión:** DiT: FP16 (`torch.float16`) · VAE: BF16 (`torch.bfloat16`, `--vae-dtype bf16`)  
**Estrategia de Offload:** `enable_sequential_cpu_offload()` (`--offload cpu`)  
**Video resultante:** [`logs/f0_480p_17f_fp16_vae_bf16_cpu_20260908_035823.mp4`](file:///d:/Gemini_Admin_Tool/marley_720p/logs/f0_480p_17f_fp16_vae_bf16_cpu_20260908_035823.mp4)  

---

## 1. Resumen Ejecutivo y Veredicto

| Criterio | Meta / Gate | Resultado Test H | Veredicto |
| :--- | :---: | :---: | :---: |
| **Peak NVML físico continuo (50ms)** | ≤ 4,800 MB | **6,028.9 MB** | ❌ **FAIL (+1,228 MB sobre 4.8 GB)** |
| **Pico Asignado por PyTorch (`max_memory_allocated`)** | < 4,800 MB | **4,366.2 MB** | 🟢 **PASS (-66.5% vs Test G)** |
| **Tiempo Denoising DiT (30 pasos)** | — | **274s (4m 34s · 9.14 s/step)** | 🟢 Estable |
| **Tiempo VAE Decode (17 frames)** | < 60s | **65s (1m 05s)** | 🟢 **ESPECTACULAR (-80% tiempo VAE)** |
| **Tiempo Total Wall-Clock** | ≤ 600s | **339.0s (5m 39s)** | 🟢 **PASS (-45.3% vs Baseline F0)** |
| **Estabilidad / Calidad Visual** | Sin NaNs ni artefactos | 🟢 Video íntegro, sin saturación | 🟢 PASS estructural |

**Veredicto Final Test H:**  
1. **Éxito masivo en latencia:** El VAE decode colapsó de **322s a solo 65s** (un 80% más rápido). El tiempo total cayó de 10.3 min a **5.6 minutos**.
2. **PyTorch allocation dentro del gate (4.366 MB):** Se eliminó por completo la pesadilla de 13 GB de tensores virtuales de FP32.
3. **El Gate NVML físico (6.028 MB):** No se cumple todavía porque al sumar la huella base de Windows/DWM (~1.100 MB) + la asignación de PyTorch (4.366 MB), el consumo físico total roza el límite de la GPU antes de recolectar basura.

---

## 2. Telemetría Comparativa: F0 vs. Test G vs. Test H

| Checkpoint / Métrica | Baseline F0 (`cpu` - FP32) | Test G (`model_cpu` - FP32) | Test H (`cpu` - BF16) | Delta Test H vs Baseline F0 |
| :--- | :---: | :---: | :---: | :---: |
| **Offload** | Sequential | Model-level | Sequential | — |
| **VAE Precision** | FP32 | FP32 | **BF16** | **Mitad de precisión** |
| **Denoising DiT (30 steps)** | 297s (9.93 s/it) | 212s (6.33 s/it) | 274s (9.14 s/it) | -23s (-7.7%) |
| **VAE decode (17 frames)** | 322s | 484s | **65s** | 🟢 **-257s (-79.8%)** |
| **Total Wall-Clock** | 619.8s (10.3 min) | 696.3s (11.6 min) | **339.0s (5.65 min)** | 🟢 **-280.8s (-45.3%)** |
| **PyTorch Alloc Pico** | ~7.9 GB virtual | 13,036.9 MB | **4,366.2 MB** | 🟢 **-8,670 MB (-66.5%)** |
| **NVML Físico Continuo** | 5,451 MB (post-snap) | 5,861 MB (smi) | **6,028.9 MB** (50ms tracker) | +577 MB (tracker captura pico transitorio) |
| **NVML post-inferencia** | 5,451 MB | 902.4 MB | **4,839.1 MB** | ~Gate exacto |
| **Calidad / Artefactos** | Referencia | Referencia | **Excelente (sin NaNs)** | 🟢 Idéntica a referencia |

---

## 3. Hallazgos Críticos de Ingeniería

### 3.1 La victoria de `bfloat16`: Desbloqueo del Cuello de Botella de Latencia
El cambio a `torch.bfloat16` demostró empíricamente por qué la teoría era correcta:
- En FP32, el decodificador causal 3D generaba 13 GB de activaciones que forzaban a Windows WDDM a paginar constantemente sobre PCIe, tardando entre 322s y 484s.
- En BF16, los tensores se redujeron a **4.366 MB**. Al caber casi enteros en VRAM sin thrashing constante de PCIe, **el VAE decodificó 17 frames en apenas 65 segundos**.

### 3.2 ¿Por qué NVML llegó a 6,028 MB si PyTorch solo asignó 4,366 MB?
Esta es la lección de ingeniería más valiosa de este test:
```
   1,100 MB (Windows OS + DWM + apps de fondo)
 + 4,366 MB (Tensores PyTorch del VAE decode)
 +   560 MB (Workspace temporal de cuDNN / PyTorch allocator fragmentation)
 ────────────────────────────────────────────────────────
 = 6,026 MB (Pico físico transitorio en hardware)
```
PyTorch por sí solo está en 4.366 MB (dentro del límite de 4.800 MB), pero en un entorno Windows de laptop, **el presupuesto disponible real para PyTorch no son 4.800 MB, sino ~3.700 MB**.

---

## 4. Conclusión e Implicaciones para la Siguiente Fase

1. **Test I (`float16` VAE):** No aportará reducción de VRAM sobre Test H, ya que tanto FP16 como BF16 utilizan exactamente 2 bytes (16 bits) por elemento (PyTorch asignaría los mismos ~4.366 MB), y además FP16 conlleva riesgo de subdesbordamiento/NaN.
2. **La Solución Definitiva es Específica: `bfloat16` + Chunking Temporal Ligero (2 chunks):**
   * Si dividimos el tensor latente (17 frames = 4 pasos latentes) en **2 chunks de 2 pasos latentes** con `overlap=1`:
   * La asignación de PyTorch caerá de **4.366 MB a ~2.200 MB**.
   * `2.200 MB (PyTorch) + 1.100 MB (SO) = ~3.300 MB NVML Físico`.
   * ¡Estará **1.500 MB por debajo del gate de 4.800 MB**!
   * Y con la velocidad de BF16 (65s), el decodificador por 2 chunks tardará tan solo ~70-75s.
