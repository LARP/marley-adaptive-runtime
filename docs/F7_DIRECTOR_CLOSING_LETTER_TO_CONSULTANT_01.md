# CARTA DE CIERRE DE LA DIRECCIÓN DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — RATIFICACIÓN FINAL Y CONCLUSIÓN DE LA CAMPAÑA F7

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Cierre formal de la Campaña F7-720p, adopción de correcciones metodológicas y ratificación del veredicto final F7-D5  
**Referencia:** Protocolo [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md); Dictámenes F7 Stage B del Consejero  

---

Estimado Consejero:

La dirección técnica acusa recibo de su dictamen de ratificación y suscribe en su totalidad las observaciones y precisiones metodológicas señaladas.

### 1. Aceptación de las precisiones epistemológicas
1. **Corrección sobre tensores en inferencia:** Aceptamos sin reservas la corrección. Al tratarse de un pipeline en modo estricto de inferencia (`torch.set_grad_enabled(False)`), la referencia anterior a "gradientes" fue inapropiada. La formulación correcta corresponde a la variabilidad de magnitudes en las activaciones intermedias, tensores latentes del scheduler Euler y workspaces temporales de convolución/atención.
2. **Tratamiento del comportamiento de WDDM como hipótesis abierta:** Reemplazamos cualquier afirmación causal categórica sobre el desalojo de superficies del DWM. Se consigna formalmente que:
   > *"El comportamiento observado es compatible con una redistribución de residencia física dentro del stack WDDM/sistema operativo, pero la atribución causal específica a componentes individuales permanece como hipótesis abierta para investigación posterior."*

### 2. Cierre y balance oficial de F7-D5
La dirección ratifica el estado definitivo del hito:
- **Clasificación Oficial:** **🔴 FAIL / COMPLETE REJECTION (Escenario 4)** frente al Gate A inmutable de 4,800.0 MB y Gate B de 3,850.0 MB, con un déficit exacto de **196.0 MB**.
- **Resultado Operacional:** Estabilidad del mecanismo durante 30 pasos sin crecimiento monotónico de memoria, cadencia nominal de **58.47 s/paso**, PyTorch Reserved acotado en **~3.77 GB**, decodificación VAE completada con 0 NaNs y exportación de video MP4 íntegro.
- **Fenómeno del Paso 6:** Queda formalmente documentado que el máximo absoluto ocurrió en el Paso 6 (4,996.0 MB) y descendió posteriormente hacia el final de la corrida (4,714 – 4,773 MB en pasos 24–30).

### 3. Protocolización de la pregunta de investigación futura
Queda incorporada en el acervo del proyecto la formulación de investigación sugerida para cualquier eventual indagación futura:
> *«¿Qué evento o conjunto de eventos produce el sobrepico de residencia física de aproximadamente 196 MB alrededor del Paso 6, y por qué dicho componente deja de requerirse o deja de permanecer físicamente residente en los pasos posteriores?»*

### 4. Conclusión y estado del Roadmap
Con la presente comunicación:
1. **La Campaña F7 (720p Extension) se declara oficialmente CERRADA como FAIL LIMPIO.**
2. **La resolución 480p se ratifica y consolida como la resolución de producción oficialmente certificada (Phase F6).**
3. **La resolución 720p permanecerá confinada y documentada en el repositorio exclusivamente como una capacidad experimental no certificada para hardware de 6 GB.**

Agradecemos profundamente el rigor, la honestidad científica y la guía metodológica que brindó a lo largo de toda la campaña F7.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
