# DICTAMEN TÉCNICO DEL CONSEJERO EXTERNO — F7 STAGE B (RATIFICACIÓN F7-D5 Y CIERRE DE CAMPAÑA)

**De:** Consejero Técnico Externo  
**Para:** Director de Proyecto — Marley Runtime  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Dictamen técnico sobre F7 Stage B — Resultado F7-D5, ratificación de cierre de campaña y precisión metodológica  
**Referencia:** Protocolo `docs/F7_720P_PREREGISTERED_PROTOCOL_01.md` §6  

---

Estimado Director:

He revisado el resultado oficial de **F7 Stage B — Full Validation, 1280×720, 33 frames, 30 pasos**, incluyendo los valores de memoria, cadencia, comportamiento temporal del consumo, ejecución VAE y resultado final del pipeline.

### 1. Ratificación del veredicto

Ratifico que:

**F7-D5 = 🔴 FAIL / COMPLETE REJECTION**

La clasificación debe permanecer inalterada.

El pico físico observado fue:

**4,996.0 MB**

frente al límite prerregistrado:

**4,800.0 MB**

con un exceso de:

**+196.0 MB**.

Asimismo, Gate B permanece en FAIL con un exceso de **196.3 MB**.

No recomiendo ninguna reclasificación retroactiva. La gobernanza del protocolo ha funcionado correctamente.

---

## 2. Ratificación de los resultados positivos

También ratifico que el FAIL del Gate A no debe interpretarse como un fallo general del mecanismo.

La ejecución demostró:

* 30 pasos completos a 1280×720 / 33 frames.
* Sin OOM.
* Sin activación del kill-switch.
* Cadencia de **58.47 s/paso**.
* Reserva PyTorch estabilizada alrededor de **3.77 GB**.
* Sin crecimiento monotónico de residencia física.
* VAE completado correctamente.
* 0 NaNs.
* MP4 válido.
* El VAE no produjo un nuevo máximo de memoria.

Por tanto, la formulación correcta sigue siendo:

> **El mecanismo demostró estabilidad operacional durante una ejecución completa de 30 pasos, pero no proporcionó margen físico suficiente para cumplir el hard gate de 4.8 GB.**

Esto constituye un resultado experimental negativo, pero altamente informativo.

---

## 3. Corrección metodológica necesaria sobre el Paso 6

Los tres mecanismos propuestos —allocator de PyTorch, comportamiento WDDM/DXGK y dinámica de las operaciones de difusión— deben clasificarse actualmente como **hipótesis explicativas**, no como causas causalmente demostradas.

En particular, sustituir cualquier lenguaje causal definitivo por:

> **“El comportamiento observado es compatible con una intervención o redistribución de residencia física dentro del stack WDDM, pero la atribución causal específica a DWM/DXGK u otro componente permanece abierta.”**

Asimismo, se retira cualquier referencia a "gradientes", dado que el workload corresponde a inferencia estricta.

---

## 4. El verdadero hallazgo del Paso 6 y pregunta abierta

El dato más importante permanece intacto:

> **El pico máximo de 4,996 MB ocurrió en el Paso 6 y no al final de los 30 pasos.**

Posteriormente:
* pasos 7–23: aproximadamente 4,870–4,940 MB;
* pasos 24–30: aproximadamente 4,714–4,773 MB.

Esto hace que la hipótesis de una fuga acumulativa simple sea poco compatible con la evidencia observada.

El expediente cierra F7 con una pregunta abierta precisa:

> **¿Qué evento o conjunto de eventos produce el sobrepico de residencia física de aproximadamente 196 MB alrededor del Paso 6, y por qué dicho componente deja de requerirse o deja de permanecer físicamente residente en los pasos posteriores?**

---

## 5. Dictamen final de cierre

Mi resolución final es:

### **F7-D5 — 🔴 FAIL CONFIRMADO**
- **Mecanismo de seam release:** Operacionalmente validado durante 30 pasos.
- **Estabilidad:** Demostrada.
- **Acumulación monotónica:** No observada.
- **VAE:** PASS funcional, sin nuevo pico global.
- **Déficit físico:** 196.0 MB frente al Gate A.
- **Fenómeno Paso 6:** Observado y cuantificado; causa específica aún no demostrada.
- **Resolución 480p:** Certificada bajo F6 para producción.
- **Resolución 720p:** Confinada como capacidad experimental.

Cierro formalmente la campaña F7 bajo el protocolo vigente.

Atentamente,

**Consejero Técnico Externo**  
**Marley Runtime**
