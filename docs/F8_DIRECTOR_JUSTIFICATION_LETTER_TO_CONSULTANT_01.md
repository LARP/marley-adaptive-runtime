# CARTA DE LA DIRECCIÓN DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — RATIFICACIÓN DE CONDICIONES Y JUSTIFICACIÓN METODOLÓGICA PARA LA EJECUCIÓN DEL F8-SCREENING (A/B DE 10 PASOS)

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Aceptación integral de las correcciones metodológicas, justificación de rigor de procesos limpios y formalización del GO para F8-Screening  
**Referencias:** Dictamen del Consejero [`docs/F8_CONSULTANT_VERDICT_SCREENING_01.md`](F8_CONSULTANT_VERDICT_SCREENING_01.md); Protocolo [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md)  

---

Estimado Consejero:

La dirección técnica acusa recibo de su dictamen formal de aprobación **🟢 GO PARA F8-SCREENING**.

Agradecemos especialmente la agudeza de su corrección metodológica sobre la herencia de estado en memoria. Sometemos a su ratificación definitiva las justificaciones técnicas que blindan la pureza metrológica del experimento antes de proceder a la ejecución:

---

## 1. Justificación de la Arquitectura de Procesos Limpios Desacoplados (Aislados por SO)

Aceptamos e implementamos su corrección como norma obligatoria:

1. **Cero Contaminación Inter-Condición:**
   - La condición **Control A (FP16 Baseline F7)** se ejecutará en un subproceso de Python completamente nuevo, registrará su línea base pasiva basal $S_0$, ejecutará los 10 pasos exactos y terminará forzando la destrucción completa de su contexto CUDA y handles mediante `ExitProcess` / `os._exit(0)`.
   - Entre la finalización de A y el arranque de B, se estipula una **ventana obligatoria de reposo de 60 segundos** para permitir que el subsistema gráfico de Windows (WDDM/DWM) complete la recolección pasiva de páginas y retorne al nivel basal estable del sistema.
   - La condición **Tratamiento B (Cuantización DiT)** arrancará desde un subproceso nuevo y limpio, inicializando su propio contexto CUDA desde cero.
2. **Validez Metrológica:**
   - Con este desacoplamiento total a nivel de kernel, queda matemáticamente garantizado que el Tratamiento B no heredará bloques pre-alocados, pools de arenas de 20/512 MB, fragmentación previa ni estado caliente del allocator de PyTorch.

---

## 2. Fidelidad Estricta al Baseline F7-D5

Garantizamos que la condición Control A reproducirá de manera idéntica las condiciones de la Campaña F7:
- **Workload:** 1280×720, 33 frames, Seed 42, Guidance 5.0, 10 pasos DiT adaptativos.
- **Pipeline:** Liberación sincrónica en costura `cond → uncond` y drenaje al inicio de paso (`empty_cache`).
- **Precisión:** FP16 nativo (`torch.float16`) en bloques DiT, sin ninguna alteración adicional.
- **Instrumentación:** Muestreador NVML continuo a 20 ms de intervalo.

El valor contra el cual se contrastará la reducción no será una estimación teórica, sino el pico físico empírico real medido en el Control A (que en F7-D5 alcanzó su ápice en el Paso 6 con 4,996.0 MB).

---

## 3. Salvaguardas de Implementación en la Condición B (Cuantización sin Copias Duplicadas)

En atención a su advertencia de que la cuantización no genere buffers redundantes que neutralicen el ahorro:
1. **Mapeo de Pesos en el Dispositivo:** Se empleará una implementación de de-cuantización / ejecución compacta por bloque (`in-place dequant` o kernels de precisión reducida) que no mantenga simultáneamente en la GPU el tensor cuantizado y el tensor des-cuantizado.
2. **Telemetría Multidimensional:** Se registrará paso a paso la terna `(Allocated, Reserved, Peak NVML)` para verificar que la reducción física en NVML guarde coherencia causal con el comportamiento de los tensores vivos.

---

## 4. Adopción de la Precisión sobre Fidelidad Numérica

Adoptamos formalmente su formulación:
> *«La similitud coseno en el paso 10 se evaluará como verificación de que la intervención no introduce una desviación numérica significativa respecto del control, reservando cualquier juicio perceptual para evaluaciones sobre imágenes o videos finales.»*

---

## 5. Ratificación Final del GO

Con la incorporación de estas salvaguardas metodológicas:
- La hipótesis es **nítida, falsable y unívoca**.
- Los cuatro umbrales de decisión ($\ge 196\text{ MB}$, $100\text{--}195\text{ MB}$, $< 50\text{ MB}$, $\approx 0\text{ MB}$) quedan **completamente congelados antes de la prueba**.
- El límite de cadencia ($\le 65\text{ s/paso}$) protege la viabilidad computacional.

La dirección técnica considera que el diseño experimental cuenta con la máxima solidez y rigor posibles. Solicitamos su visto bueno final para proceder con el congelamiento del protocolo prerregistrado y la ejecución de la batería A/B.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
