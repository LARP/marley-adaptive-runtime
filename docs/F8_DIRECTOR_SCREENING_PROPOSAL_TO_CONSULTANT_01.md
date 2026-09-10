# CARTA DE LA DIRECCIÓN DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — ACEPTACIÓN DE LA RESOLUCIÓN ESTRATÉGICA Y PROPUESTA DE DISEÑO EXPERIMENTAL F8-SCREENING (A/B DE 10 PASOS)

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Aceptación de la resolución estratégica, adopción de la matriz de decisión y especificación del experimento A/B de screening (10 pasos)  
**Referencias:** Dictamen del Consejero sobre Alternativas post-F7; Protocolo [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md)  

---

Estimado Consejero:

La dirección técnica acoge con total acuerdo su resolución metodológica. Compartimos plenamente el principio de **máxima economía experimental y falsación rápida (*fail-fast*)**: en lugar de abrir una investigación forense prolongada (F8-A), evaluaremos directamente si una reducción de la presión de memoria mediante cuantización selectiva es capaz de eliminar el sobrepico de ~196 MB en la ventana crítica de los pasos 5 a 7.

Confirmamos la adopción formal de los siguientes pilares de su dictamen:

---

## 1. Adopción de la Matriz de Decisión Cuantitativa

Sometemos el experimento a la regla de cuatro niveles propuesta en su dictamen, evaluando el pico físico NVML alrededor de los pasos 5–7 a resolución nativa **1280×720 / 33 frames**:

| Nivel de Resultado | Magnitud de Reducción del Pico NVML | Decisión Ejecutiva Vinculante |
| :--- | :---: | :--- |
| **Resultado Fuerte** | $\ge \mathbf{196.0\text{ MB}}$ | **🟢 GO:** Autoriza la validación completa de 30 pasos para certificación 720p. |
| **Resultado Prometedor** | $\mathbf{100.0\text{ – }195.0\text{ MB}}$ | **🟡 GO CONDICIONAL:** Justifica evaluar una segunda micro-intervención mínima. |
| **Resultado Débil** | $< \mathbf{50.0\text{ MB}}$ | **🔴 NO-GO:** Descarta la cuantización como solución viable; no se amplía la intervención. |
| **Resultado Nulo** | $\approx \mathbf{0.0\text{ MB}}$ | **🔴 CIERRE DEFINITIVO:** Cierra irrevocablemente la investigación de 720p en 6 GB. |

Asimismo, ratificamos que un pase de memoria por sí solo no bastará: la cadencia deberá mantenerse en un rango computacionalmente viable ($\le 65\text{ s/paso}$) y con fidelidad numérica intacta (0 NaNs / 0 Infs).

---

## 2. Especificación del Diseño Experimental Pareado (Same-Session A/B)

Para garantizar un aislamiento causal riguroso y neutralizar cualquier variabilidad térmica o del fondo de Windows 11 WDDM, proponemos una ejecución pareada en la misma sesión de trabajo:

### Condición A: Control (Línea Base F7-D5)
- Inferencia de **10 pasos DiT exactos** a **1280×720 @ 33 frames**, Seed 42, Guidance 5.0.
- Modo `adaptive` con vaciado quirúrgico en la costura `cond → uncond` y drenaje al inicio de cada paso.
- Pesos de bloques DiT en FP16 nativo (`torch.float16`).
- Decodificación VAE desactivada temporalmente (o evaluada al final) para focalizar el 100% de la resolución metrológica en la ventana de difusión de los pasos 5 a 7.

### Condición B: Tratamiento (Cuantización Selectiva DiT)
- Idéntico workload (10 pasos, 720p, 33 frames, Seed 42, Guidance 5.0, mismo scheduler y seam release).
- Proyecciones lineales de atención y feed-forward de los 30 bloques DiT cuantizadas según el estándar validado en **Fase F1.7** (reducción de payload y huella de tensores en forward).
- Muestreo a alta frecuencia (20 ms) con NVML continuo y telemetría por paso de `Allocated`, `Reserved` y `Peak NVML`.

---

## 3. Consideraciones Técnicas sobre la Implementación de la Cuantización

Para evitar artefactos de memoria donde la de-cuantización cree un pico transitorio propio en la GPU:
1. Aseguraremos que el buffer de de-cuantización se mantenga estrictamente confinado al bloque activo en ejecución, liberándose sincrónicamente antes de transferir el siguiente bloque.
2. Registraremos la similitud coseno de los tensores latentes en el paso 10 respecto al Control A para verificar que la degradación perceptual sea nula.

---

## 4. Solicitud de Aprobación

Si el diseño experimental pareado y la formulación metodológica cuentan con su aprobación, la dirección procederá a redactar el protocolo prerregistrado de screening de 10 pasos y preparar el script ejecutable para la corrida A/B.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
