# Acta de Cierre Formal y Aceptación de Dictamen Técnico — Fase F5

**Para:** Consejero Técnico Externo (SD / ComfyUI / Generative AI Runtimes) · Equipo de Marley Runtime  
**De:** Director de Proyecto — Marley Runtime  
**Fecha:** 8 de septiembre de 2026  
**Asunto:** Cierre formal de Fase F5 (`RETIRED BY EVIDENCE`), Adopción de Regla de No Regresión y Directiva de Avance a Fase F6  
**Estado:** 🟢 **CERRADO Y RATIFICADO — DIRECTIVA VINCULANTE**

---

## 1. Aceptación del Dictamen Técnico

He recibido y analizado el informe final del Consejero Técnico Externo respecto a los resultados del probe **F5-A**.

Acepto en su totalidad su evaluación profesional y dicto formalmente:

> **🟢 FASE F5 CERRADA COMO RETIRED BY EVIDENCE — RESOLVED BY EXISTING TILED VAE.**  
> **🟢 SE AUTORIZA EL AVANCE DIRECTO A FASE F6 (Pipeline End-to-End en 480p / 33 frames).**

Coincido plenamente en su reflexión: el valor de esta fase reside en demostrar empíricamente **qué NO construir**. En un entorno de hardware con 6 GB de VRAM compartida con el sistema operativo (WDDM), evitar la sobre-ingeniería y prescindir de capas innecesarias de sincronización, particionado y blending temporal es una de las decisiones más sólidas que hemos tomado en el proyecto.

---

## 2. Incorporación de la Regla de No Regresión del VAE

Adoptando la recomendación del punto §9 del Consejero, se establece con carácter vinculante la siguiente **Regla de No Regresión**:

> [!IMPORTANT]
> **Regla de No Regresión del VAE (Baseline F5-A):**  
> Ninguna optimización, cambio de flujo o integración en Fase F6 podrá degradar el baseline canónico alcanzado en F5-A:
> - **Peak VRAM Físico (NVML):** $\le 2.109\text{ MB}$ (tolerancia máxima de ruido $\le 2.300\text{ MB}$).
> - **Latencia de VAE Decode (33f):** $\approx 27\text{ s}$ (techo operativo $\le 35\text{ s}$).  
> Cualquier desviación anómala exigirá justificación formal o la reversión inmediata a la ruta nativa `bfloat16` + `enable_tiling()` ($256 \times 256$).

---

## 3. Reubicación del Foco de Riesgo hacia F6

Con el VAE temporal definitivamente resuelto a 33 frames (consumiendo solo el **44% del presupuesto** de 4,8 GB y menos de 30 segundos), el cuello de botella del proyecto se traslada en un 100% al **DiT Denoising Loop** (30 pasos de inferencia).

El mandato para **Fase F6** queda definido bajo los siguientes pilares:
1. **Presupuesto Global:** Ejecutar el pipeline completo (Text Encoding + 30 Denoising Steps con Marley Streaming + VAE Decode) bajo el Hard Gate de $\le 4.800\text{ MB}$ (Target $\le 4.000\text{ MB}$).
2. **Milestone B (Wall-Clock):** Demostrar una latencia total end-to-end $< 10\text{ minutos}$ en 480p / 33 frames.
3. **Reutilización de Primitivas Congeladas:** No introducir nuevas optimizaciones en el VAE; concentrar el esfuerzo en evaluar la orquestación entre `BudgetedAsyncStreamer` (FP16), `INT8BudgetedStreamer` y el motor adaptativo de F4 sobre una generación real de 33 frames.

---

## 4. Registro y Cierre Oficial

Con este documento queda concluido el ciclo de revisión de F5:
- **`f5_vae_probe_33f.py`** se preserva como prueba de regresión canónica.
- **`marley/ops/vae_stitch.py`** queda oficialmente **cancelado y retirado**.
- Se ordena el inicio de la planificación técnica de la **Fase F6**.

**Director de Proyecto — Marley Runtime**  
*En memoria de Marley 🐾*
