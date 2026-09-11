# F9-1 — Preregistered Workspace Reduction Protocol & Pre-Flight Verifications

**Date:** 2026-09-11  
**Status:** **FROZEN / READY FOR CONFIRMATORY EXECUTION**  
**Charter:** [`docs/F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md`](F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md)  
**Diagnostic Origin:** [`docs/F9_0_ATTRIBUTION_REPORT_01.md`](F9_0_ATTRIBUTION_REPORT_01.md) (`CLOSED - ATTRIBUTED`)  
**Methodological Review:** [`docs/private/F9_STAGE_0_CONSULTANT_VERDICT_01.md`](private/F9_STAGE_0_CONSULTANT_VERDICT_01.md) (APPROVED)  
**Governance Order:** [`docs/private/F9_DIRECTOR_INSTRUCTIONS_TO_TECH_DIRECTOR_01.md`](private/F9_DIRECTOR_INSTRUCTIONS_TO_TECH_DIRECTOR_01.md)

---

## 1. Workload Científico Congelado

* **Modelo:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
* **Resolución:** 1280 × 720
* **Frames:** 33
* **Pasos DiT:** 30
* **Prompt:** *"A golden retriever dog runs joyfully across a sunlit meadow, cinematic lighting, shallow depth of field, 4K."*
* **Negative prompt:** *"blurry, low quality, watermark, deformed"*
* **Semilla:** 42 · **Guidance:** 5.0 · **Modo base:** `adaptive`
* **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Windows 11 WDDM)
* **Referencia histórica F7-D5:** $5,096.1\text{ MB}$ (Déficit $+296.1\text{ MB}$ vs Gate $4,800\text{ MB}$)

---

## 2. Hipótesis Causal F9-1A

> **F9-0 identificó a la Categoría 3 (Workspace de Kernels) como el componente dominante observado en la carga DiT a 720p. En consecuencia, F9-1A establece la hipótesis causal falsable de que una intervención explícita de control/capping de workspaces en los kernels de cómputo reducirá de manera física y reproducible el pico NVML total respecto del control contemporáneo, sin inducir desplazamientos compensatorios a la Categoría 4 (Allocator) ni violar las condiciones de rendimiento.**

---

## 3. Verificaciones Técnicas Previas al Congelamiento (V1 – V8)

### V1. Datos Primarios de Categoría 4 (F9-0)
Recuperados de [`logs/f9_0_attribution_telemetry.json`](../logs/f9_0_attribution_telemetry.json):
* Run 1: `3,173.9400 MB`
* Run 2: `451.6200 MB`
* Run 3: `451.6100 MB`
* Run 4: `451.6200 MB`
* Run 5: `451.6200 MB`

### V2. Cálculo Definitivo de $\epsilon$
* Mediana ($\text{Cat 4}$): $451.6200\text{ MB}$
* $\text{MAD}(\text{Cat 4})$: $0.0000\text{ MB}$
* Fórmula aplicada: $\epsilon = \max(64\text{ MB}, \text{mediana} + 3\times\text{MAD}) \implies \mathbf{64.0000\text{ MB}}$.

### V3 & V4. Determinism Check Control-Control ($N=3$)
Ejecutado con 3 pasadas independientes de 30 pasos DiT:
* $\text{Max absolute diff } (R_1 \text{ vs } R_2): 0.000000\text{e}+00$
* $\text{Max absolute diff } (R_1 \text{ vs } R_3): 0.000000\text{e}+00$
* $\text{Max absolute diff } (R_2 \text{ vs } R_3): 0.000000\text{e}+00$
* **Max $|l_i - l_j|$ global:** $\mathbf{0.000000\text{e}+00} \le 10^{-3}$ (**PASSED / VALID**)
* **Régimen determinista:** Confirmado bit-exacto. No se requiere alterar algoritmos de PyTorch ni introducir asimetrías de control.

### V5 & V6. Secuencia de Bloques Balanceada y Hash Criptográfico
* **Estructura:** 10 bloques pareados (5 bloques $C \to I$, 5 bloques $I \to C$).
* **Secuencia fija:**
  `B1: C->I, B2: I->C, B3: C->I, B4: I->C, B5: I->C, B6: I->C, B7: C->I, B8: I->C, B9: C->I, B10: C->I`
* **Representación canónica:** `C->I|I->C|C->I|I->C|I->C|I->C|C->I|I->C|C->I|C->I`
* **Hash SHA-256 inmutable:**  
  `e0aac00319dc2444414f63c1577fb80662ee558345a66dbb6166c6117547227e`

### V7. Protocolo de Reset Simétrico entre Corridas
* Proceso limpio desacoplado por SO por cada corrida.
* Ventana de reposo térmico y estabilización WDDM de 60 segundos pre-corrida ($S_0 \text{ spread} \le 150\text{ MB}$).
* Procedimiento de reinicio y sincronización simétrico e idéntico para condiciones $C$ e $I$.

---

## 4. Diseño Experimental y Análisis Bootstrap Pareado

Para cada bloque $k \in \{1 \dots 10\}$:
$$\Delta \text{PeakNVML}_k = \text{PeakNVML}_{I,k} - \text{PeakNVML}_{C,k}$$
$$\Delta \text{Cat3}_k = \text{Cat3}_{I,k} - \text{Cat3}_{C,k}$$
$$\Delta \text{Cat4}_k = \text{Cat4}_{I,k} - \text{Cat4}_{C,k}$$

* **Bootstrap pareado:** 10,000 remuestreos con reemplazo sobre las 10 diferencias pareadas $\Delta_k$.
* **Criterio Causal Primario (A):** $\text{mediana}(\Delta \text{PeakNVML}) < 0$ y $\text{Upper } \text{CI}_{95\%} < 0$.
* **Criterio Operacional (B):** $\ge 8 / 10$ corridas de intervención con $\text{PeakNVML}_I < 4,800\text{ MB}$.
* **Criterio Mecanístico (C):** $\text{mediana}(\Delta \text{Cat3}) < 0$ y $\text{Upper } \text{CI}_{95\%} < 0$.
* **Regla Anti-Desplazamiento (D):**
  $$C_k = \begin{cases} \dfrac{\max(0, \Delta \text{Cat4}_k)}{|\min(\Delta \text{Cat3}_k, 0)|}, & \Delta \text{Cat3}_k < 0 \\ \text{NA}, & \Delta \text{Cat3}_k \ge 0 \end{cases}$$
  No más de 1 bloque ($10\%$) con $C_k \ge 0.5$.

---

## 5. Declaración Formal de Congelamiento

Cumplidas las 8 verificaciones previas conforme a las instrucciones de la Dirección y el dictamen del Experto Externo, el protocolo queda formalmente **CONGELADO**.
