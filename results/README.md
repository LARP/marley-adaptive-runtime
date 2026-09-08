# Marley Runtime — Registro Central de Pruebas y Benchmarks

Carpeta dedicada a la documentación, telemetría y análisis de resultados de todos los tests del runtime (Fase F0.5 en adelante).

---

## 1. Matriz General de Experimentos (Fase F0.5 — VAE & Offload Optimization)

**Objetivo del Gate F0.5:** Reducir el pico de VRAM física (NVML) a **≤ 4,800 MB** en RTX 3050 6GB Laptop.  
**Baseline F0:** Peak NVML = **5,451 MB** (+651 MB exceso) · Tiempo VAE = **322s** (Sequential Offload, FP32 VAE).

| Test ID | Descripción / Estrategia | Configuración | Peak NVML | Tiempo Denoise | Tiempo VAE | Tiempo Total | Gate F0.5 | Reporte |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0 (Ref)** | Sequential CPU Offload + FP32 VAE | `res=480p, 16f, DiT=fp16, VAE=fp32, offload=cpu` | 5,451 MB | 297s | 322s | 619.8s | ❌ +651 MB | [Baseline F0](../logs/f0_480p_16f_fp16_cpu_20260908_030437_telemetry.json) |
| **Test G** | Model CPU Offload (submodel level) | `res=480p, 17f, DiT=fp16, VAE=fp32, offload=model_cpu` | **5,861 MB** (live peak) | **212s** (6.3s/it) | **484s** | **696.3s** | ❌ +1,061 MB (Paginado WDDM 13GB) | [TEST_G_model_cpu_offload.md](./TEST_G_model_cpu_offload.md) |
| **Test H** | Sequential Offload + bfloat16 VAE | `res=480p, 17f, DiT=fp16, VAE=bf16, offload=cpu` | **6,028 MB** (peak) / 4,839 post | **274s** (9.1s/it) | **65s** 🟢 | **339.0s** 🟢 | ⚠️ Alloc 4,366 MB (VAE -80% tiempo) | [TEST_H_bfloat16_vae.md](./TEST_H_bfloat16_vae.md) |
| **Test I** | Sequential Offload + float16 VAE | `res=480p, 17f, DiT=fp16, VAE=fp16, offload=cpu` | *Opcional* (mismo consumo que BF16) | *TBD* | *TBD* | *TBD* | *Riesgo inestabilidad* | Pendiente |
| **Test J** | Chunked VAE (chunk=2, overlap=1) | Condicional a Sprint 2 | — | — | — | — | *Standby* | Standby |
| **Test K** | Chunked VAE (chunk=1, overlap=1) | Condicional a Sprint 2 | — | — | — | — | *Standby* | Standby |
| **Test L** | Chunked VAE + GPU Residency | Condicional a Sprint 3 | — | — | — | — | *Standby* | Standby |

---

## 2. Estructura de Documentación de Resultados

Cada test cuenta con su archivo de informe técnico individual en esta carpeta:
- `TEST_<ID>_<estrategia>.md`: Documento con hipótesis, comandos reproducibles, telemetría en los 3 niveles de memoria, análisis de cuellos de botella y veredicto del gate.
- Los videos `.mp4` se almacenan localmente y están excluidos del repositorio Git por política de tamaño.
