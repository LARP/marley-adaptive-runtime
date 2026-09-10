# CARTA DE LA DIRECCIÓN DE PROYECTO AL CONSEJERO TÉCNICO EXTERNO — ANÁLISIS CRÍTICO, PUNTOS DE DEBATE Y RESOLUCIÓN ESTRATÉGICA POST-F7

**De:** Director de Proyecto — Marley Runtime  
**Para:** Consejero Técnico Externo  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Respuesta técnica y debate metodológico sobre el dictamen de alternativas de continuación post-F7  
**Referencias:** Dictamen del Consejero sobre Alternativas post-F7; Protocolo [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md)  

---

Estimado Consejero:

He recibido y analizado minuciosamente su dictamen metodológico respecto a las tres alternativas propuestas tras el cierre oficial de la Campaña F7.

Coincidimos plenamente en su primer principio de gobernanza: **F6/480p debe permanecer como la línea base indiscutible y certificada de producción**, mientras que cualquier trabajo sobre 720p debe considerarse estrictamente experimental y desacoplado.

Asimismo, consideramos de gran valor su advertencia respecto de no realizar una "suma lineal de payloads" al proyectar los ahorros de la cuantización, dado que bajo el *sequential CPU offload* solo uno o dos bloques DiT residen simultáneamente en VRAM activa.

No obstante, tras un análisis crítico exhaustivo desde la perspectiva de la ingeniería de sistemas y la gestión de recursos del proyecto, someto a su consideración los siguientes **tres puntos técnicos de debate** sobre la conveniencia y alcance de abrir inmediatamente una campaña forense (F8-A) frente a la posibilidad de un cierre definitivo o una intervención unificada:

---

## 1. Primer Punto de Debate: El riesgo de sobre-diagnóstico (*Analysis Paralysis*) en el Paso 6

Su dictamen propone priorizar una investigación forense pura (**F8-A**) para responder qué fenómeno distingue al Paso 6 del resto de la ejecución.

Desde la perspectiva de ingeniería, debemos contrastar esta prioridad con la evidencia ya observada:
1. **El Paso 6 es un transitorio de calentamiento de arena:** En sistemas complejos con *caching allocators* (PyTorch C++ allocator sobre el subsistema virtual WDDM), los primeros 5 a 6 pasos corresponden a la fase de estabilización del *pool* de bloques libres (arenas de 20 MB y 512 MB). Una vez saturado ese espacio de fragmentación interna, el sistema alcanza un régimen permanente (pasos 7 al 23 planos, y pasos 24 al 30 en descenso hacia 4,714 MB).
2. **Costo experimental desproporcionado:** Diseñar, aislar y ejecutar un micro-profiler para los pasos 4 a 7 requerirá instrumentación intrusiva a nivel de kernel/driver que podría alterar los propios tiempos y allocations del allocator. Existe una alta probabilidad de que la conclusión final sea simplemente confirmar lo que la heurística de PyTorch ya estipula: *el allocator requirió 6 pasos para encontrar una topología de bloques reutilizables estable*.

**Pregunta de debate:** ¿Se justifica una inversión de tiempo y recursos en una campaña forense completa F8-A para caracterizar un transitorio de solo 196 MB, en lugar de proceder directamente a la consolidación de la versión de producción o a una prueba empírica directa?

---

## 2. Segundo Punto de Debate: La cuantización (F8-B) como reductora de la presión global de memoria, no solo de pesos

Compartimos su precisión de que ahorrar 42 MB de pesos por bloque no se traduce en 1.2 GB de ahorro simultáneo en GPU, dado que los bloques se descargan secuencialmente. 

Sin embargo, el beneficio físico de la cuantización (F8-B) no radica únicamente en los pesos estáticos:
- **Reducción del tamaño de activaciones intermedias:** Al operar con proyecciones en precisión reducida (FP8/INT8), los tensores temporales generados durante el cálculo de atención y proyecciones lineales ven reducido su *byte footprint*.
- **Evitación del umbral de asignación de arenas:** Reducir la huella pico instantánea de cada bloque en ~40–50 MB durante el pase forward puede ser exactamente lo que impida que el allocator de PyTorch cruce el umbral que le obliga a solicitar una nueva arena de 20 MB o 512 MB a WDDM durante ese Paso 6 crítico.

Por tanto, F8-B no es un "parche a ciegas", sino una reducción estructural de la presión termodinámica del sistema que podría suprimir el transitorio del Paso 6 por diseño.

---

## 3. Tercer Punto de Debate: La ecuación de costo-beneficio del proyecto (¿Es prioritario 720p en 6 GB?)

Su dictamen señala con agudeza que F8-A permitiría determinar *"si el coste experimental justifica continuar persiguiendo 720p en el hardware de 6 GB"*.

La dirección técnica considera indispensable plantear abiertamente esta reflexión estratégica en este momento:
- **480p / 33 frames:** Genera video de alta calidad a **~10 segundos por paso** con un margen de seguridad de más de **2,200 MB** libres en VRAM. Es una experiencia de usuario ágil, robusta y 100% confiable.
- **720p / 33 frames:** Requiere **~58.5 segundos por paso** (casi 30 minutos de cómputo continuo de GPU por video). Incluso si logramos recuperar los 196 MB mediante F8-A o F8-B, el usuario final enfrentará una latencia casi 6 veces mayor en una GPU laptop de gama de entrada.

Si el objetivo primordial de `marley-runtime` es proveer un runtime práctico, usable y eficiente para GPUs de 6 GB, insistir en perseguir los 196 MB de 720p podría constituir una optimización académica con escaso retorno práctico para el usuario final.

---

## 4. Propuesta de Resolución de la Dirección

A la luz de este análisis crítico, la dirección técnica propone al Consejero evaluar **dos caminos de resolución**:

* **Opción A (Pragmática / Cierre Definitivo):**  
  Adoptar formalmente la **Alternativa 1**: consolidar y congelar `marley-runtime` en su **Versión 1.0 de Producción**, ratificando 480p como estándar definitivo y dejando 720p documentado como capacidad experimental en el repositorio. Cerrar el ciclo de desarrollo activo y enfocar esfuerzos en usabilidad o empaquetado.
* **Opción B (Investigación Unificada A/B F8):**  
  Si se decide continuar la investigación de 720p, no dividir F8 en dos campañas separadas y dilatadas en el tiempo, sino diseñar un **experimento comparativo A/B único y acotado**: correr los primeros 10 pasos con y sin cuantización de proyecciones, midiendo simultáneamente la telemetría del Paso 6 y el efecto de la reducción de huella en una sola batería experimental.

Aguardamos su análisis crítico sobre estos puntos para adoptar la decisión ejecutiva final.

Atentamente,

**Director de Proyecto**  
**Marley Runtime**
