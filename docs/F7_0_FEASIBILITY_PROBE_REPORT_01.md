# Informe de Viabilidad de Memoria a 720p (Phase F7-0 Probe)

**Para:** Director de Proyecto & Consejero Técnico Externo  
**De:** Equipo de Arquitectura e Implementación / Marley Runtime  
**Fecha:** 2026-09-09  
**Resolución Evaluada:** 1280×720 (720p) @ 33 frames (5 pasos exploratorios)  
**Estado del Probe:** ❌ **HARD GATE VIOLATION / ABORTO SEGURO REGISTRADO**

---

## 1. Métricas Observadas en el Probe F7-0

| Métrica de Auditoría | Valor Observado (720p Probe) | Baseline F6 (480p Media) | Gate / Umbral | Estado |
| :--- | :---: | :---: | :---: | :---: |
| **Peak Físico NVML** | **6058.5 MB** | 2,768.3 MB (FP16) / 4,047.8 MB (Adapt) | **≤ 4,800.0 MB** | ❌ FAIL |
| **PyTorch Allocated** | 2187.5 MB | ~1,204.2 MB | Telemetría interna | Registrado |
| **PyTorch Reserved** | 5894.0 MB | ~1,656.0 MB | Telemetría interna | Registrado |
| **Host Process RSS** | 2151.8 MB | ~3,489.6 MB | Memoria RAM host | Registrado |
| **Cadencia Denoising** | 68.75 s/paso | 14.25 s/paso | Métrica observacional | Registrado |
| **Decodificación VAE Tiled** | 69.13 s | ~28.3 s | Decodificación 33f en 720p | Registrado |
| **Integridad Numérica** | 0 NaNs / Infs 🟢 | 0 NaNs | 100% libre de NaNs | 🟢 PASS |
| **Archivo de Salida** | MP4 Decodificable 🟢 | MP4 Válido | Video reproducible | 🟢 PASS |

---

## 2. Diagnóstico Metodológico y Próximos Pasos

- **Margen Físico Observado:** -1258.5 MB de holgura frente al límite de 4.800 MB.
- **Condiciones de Hardware:** GPU Temp=80 °C, Clock=547 MHz.
- **Gobernanza 'Primero medir, después optimizar':** El probe no superó los criterios. Conforme a la regla de gobernanza, no se aplicarán optimizaciones automáticas. Se formulará la hipótesis basada en el contexto forense registrado para someterla a decisión humana.
