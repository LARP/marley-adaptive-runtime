# DICTAMEN TÉCNICO DEL CONSEJERO EXTERNO — F8 SCREENING A/B (APROBACIÓN Y CORRECCIÓN METODOLÓGICA)

**De:** Consejero Técnico Externo  
**Para:** Director de Proyecto — Marley Runtime  
**Fecha:** 10 de septiembre de 2026  
**Asunto:** Aprobación técnica del F8-Screening A/B y corrección metodológica final  
**Referencia:** Propuesta de Screening [`docs/F8_DIRECTOR_SCREENING_PROPOSAL_TO_CONSULTANT_01.md`](F8_DIRECTOR_SCREENING_PROPOSAL_TO_CONSULTANT_01.md)  

---

Estimado Director:

He revisado la especificación propuesta para el **F8-Screening A/B de 10 pasos**.

Mi dictamen es:

# 🟢 GO — APROBADO, CON UNA CORRECCIÓN METODOLÓGICA OBLIGATORIA

La campaña es suficientemente acotada, falsable y económica como para ejecutarse. No recomiendo volver a abrir en este momento una campaña forense extensa sobre el Paso 6.

La hipótesis que debe probarse es sencilla:

> **¿La cuantización selectiva de las proyecciones DiT produce una reducción suficiente y reproducible del pico físico NVML durante la ejecución 720p como para justificar una validación completa de 30 pasos?**

## 1. Corrección sobre “Same-Session A/B”

Recomiendo no ejecutar A y B dentro del mismo proceso manteniendo el mismo contexto de ejecución.

La razón es que precisamente estamos investigando un fenómeno relacionado con memoria, allocator y residencia física. Si A precede a B dentro del mismo proceso, B podría heredar:

* pools del allocator;
* bloques cacheados;
* contexto CUDA;
* estado de residencia;
* fragmentación;
* y otros efectos de calentamiento.

Por tanto, la comparación podría quedar contaminada.

La solución recomendada es:

**A = proceso limpio.**

**B = proceso limpio.**

Mismo workload y mismas condiciones experimentales, pero sin herencia de estado entre ambas condiciones.

Si resulta operacionalmente necesario ejecutarlas en una misma sesión de Windows, cada condición debe comenzar al menos desde un proceso/runtime limpio y registrarse el estado basal correspondiente.

## 2. No modificar el baseline F7

La Condición A debe reproducir lo más fielmente posible el mecanismo de F7:

* 1280×720;
* 33 frames;
* Seed 42;
* Guidance 5.0;
* 10 pasos;
* scheduler idéntico;
* adaptive;
* seam release idéntico;
* FP16;
* misma configuración de offload;
* misma instrumentación NVML.

No debe introducirse ninguna optimización adicional.

Esto es esencial porque el valor de referencia continúa siendo:

**F7-D5 = 4.996 MB de pico físico.**

El screening debe explicar cuánto modifica ese valor la cuantización, no crear un nuevo baseline.

## 3. Condición B

Apruebo la intervención propuesta con una precisión:

La implementación deberá evitar que la cuantización genere una copia completa adicional del bloque que termine neutralizando el beneficio esperado.

El dato decisivo no será:

> “los pesos pesan menos”.

Será:

> **“el sistema completo presenta un pico NVML menor”.**

Por ello, la telemetría debe observar simultáneamente:

* Peak NVML;
* Allocated;
* Reserved;
* transfer volume;
* tiempo por paso;
* y cualquier buffer temporal relevante que pueda identificarse sin instrumentación invasiva.

## 4. Sobre la comparación numérica

Apruebo la comprobación de similitud coseno, pero recomiendo no describirla como prueba de “degradación perceptual nula”.

La similitud coseno es una medida de fidelidad numérica entre representaciones.

Por tanto, debe formularse como:

> **verificación de que la intervención no introduce una desviación numérica significativa respecto del control.**

La validación perceptual, si se desea posteriormente, deberá realizarse mediante una métrica apropiada sobre el resultado generado.

Para este screening, no es necesario convertir la prueba en una evaluación perceptual completa.

## 5. Criterio de éxito del screening

Mantengo la matriz previamente aprobada:

### 🟢 ≥196 MB

La intervención ha recuperado al menos el déficit observado en F7.

**GO → ejecutar validación completa de 30 pasos.**

### 🟡 100–195 MB

Existe una señal suficientemente fuerte como para considerar una segunda micro-intervención.

**GO CONDICIONAL.**

### 🔴 <50 MB

La intervención no proporciona evidencia suficiente para justificar ampliarla.

**NO-GO para continuar por esta vía.**

### ~0 MB

La hipótesis queda esencialmente falsada para este mecanismo.

**Cierre de esta línea experimental.**

## 6. Una observación adicional sobre los 65 s/paso

Apruebo mantener el límite de **≤65 s/paso** como criterio de viabilidad.

No obstante, el screening de 10 pasos debe separar dos preguntas:

**Pregunta primaria:**

¿reduce la memoria física?

**Pregunta secundaria:**

¿a qué coste computacional?

Una intervención que reduzca 250 MB pero convierta 58 s/paso en 110 s/paso no debería considerarse automáticamente una solución de producción.

## 7. Decisión final

Con la corrección indicada, el protocolo queda aprobado.

La secuencia recomendada es:

**F8-Screening A — proceso limpio**

↓

**F8-Screening B — proceso limpio**

↓

**Comparación objetiva**

↓

**Aplicación de la matriz de decisión**

Por tanto:

# 🟢 GO PARA F8-SCREENING

Atentamente,

**Consejero Técnico Externo**  
**Marley Runtime**
