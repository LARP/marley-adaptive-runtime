# CARTA DE LA DIRECCIÓN DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — PROPUESTA DE ALTERNATIVAS Y PASOS A SEGUIR TRAS EL CIERRE DE F7

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Consulta metodológica sobre tres alternativas de continuación post-F7  
**Referencias:** Protocolo [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md); Dictámenes F7 Stage B/C  

---

Estimado Consejero:

Habiendo ratificado formalmente el cierre de la **Fase F7 como un FAIL limpio y transparente** (Gate A excedido por 196.0 MB en el Paso 6, confirmando a 480p como la resolución oficial de producción y a 720p como experimental), la dirección técnica ha estructurado **tres alternativas estratégicas de continuación** fundadas en la evidencia empírica acumulada a lo largo del proyecto.

Sometemos a su consideración y criterio metodológico estas tres opciones para orientar el siguiente ciclo de desarrollo:

---

## Alternativa 1: Consolidación y Congelación de Producción v1.0 (Línea Conservadora / Roadmap Baseline)

* **Fundamento:** La Fase F6 demostró certificación multidimensional perfecta en **832×480 @ 33 frames** (12/12 pases limpios, ~2.6 GB VRAM, 0 NaNs, tiempo nominal). La Campaña F7 demostró que 720p es funcionalmente estable, pero físicamente inviable bajo el techo de 4.8 GB sin modificaciones adicionales.
* **Alcance:**
  1. Actualizar definitivamente el `ROADMAP.md` formalizando el estado final de las fases F0 a F7.
  2. Empaquetar el runtime en una versión de distribución estable (v1.0-rc).
  3. Establecer 480p como la resolución predeterminada de producción y dejar 720p disponible bajo un flag explícito de advertencia (`--experimental-720p`).
* **Ventaja:** Cero riesgo de regresión metodológica; consolidación de un sistema robusto, testeado y terminado para el hardware objetivo.

---

## Alternativa 2: Protocolo Quirúrgico Forense sobre el Paso 6 (Línea de Diagnóstico Causal F8-A)

* **Fundamento:** El dictamen final de F7 protocolizó la pregunta central: *¿por qué se produce el sobrepico de ~196 MB en el Paso 6 y decae posteriormente?*
* **Alcance:**
  1. Diseñar un protocolo prerregistrado de replay forense del Paso 6 (muestreando exclusivamente los pasos 4, 5, 6 y 7 a nivel de micro-operación).
  2. Trazar qué tensores temporales de atención cruzada o proyecciones de patch embedding se crean transitoriamente en el Paso 6.
  3. Medir si forzar un vaciado de caché intermedio o ajustar la heurística de tamaño de bloques de PyTorch antes del Paso 6 evita la creación de una arena de 20/512 MB adicional en el driver.
* **Ventaja:** Resuelve la pregunta científica abierta sin alterar la arquitectura general ni prejuzgar soluciones.

---

## Alternativa 3: Cuantización Selectiva de Proyecciones DiT en FP8/INT8 (Línea de Intervención Algorítmica F8-B)

* **Fundamento:** En la Fase F1.7 ya demostramos empíricamente que la cuantización selectiva de bloques DiT redujo el payload de transferencia de 88.6 MB a 45.9 MB por bloque (-48.2%) con una fidelidad de similitud coseno de **0.9996** (cero NaNs).
* **Alcance:**
  1. Aplicar cuantización INT8 o FP8 (`torch.float8_e4m3fn`) a las matrices de proyección lineal de los 30 bloques DiT mientras se transmiten a GPU.
  2. Reducir la huella residente activa de cada bloque en GPU en ~42 MB, lo que recuperaría con creces el déficit de **196.0 MB**, situando el pico físico global de 720p holgadamente por debajo de los **4,600 MB**.
* **Ventaja:** Ofrece una vía técnica sólida y ya validada numéricamente en F1.7 para certificar formalmente 720p dentro del hard gate de 4.8 GB.

---

Solicitamos su valoración técnica y consejo metodológico sobre cuál de estas tres vías considera más idónea y prudente para la gobernanza y objetivos del proyecto Marley Runtime.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
