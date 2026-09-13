# F9-1B — Preregistered Allocator Policy Protocol & Pre-Flight Verifications

**Documento:** `F9_1B_PREREGISTERED_ALLOCATOR_PROTOCOL_01.md`  
**Fecha:** 12 de septiembre de 2026  
**Estado:** 🟢 **FROZEN / READY FOR CONFIRMATORY EXECUTION**  
**Charter:** [`docs/F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md`](F9_RUNTIME_MEMORY_RESEARCH_SANDBOX_CHARTER_01.md)  
**Origen Diagnóstico:** [`docs/F9_0_ATTRIBUTION_REPORT_01.md`](F9_0_ATTRIBUTION_REPORT_01.md) y cierre de F9-1A ([`docs/F9_1A_WORKSPACE_REDUCTION_REPORT_01.md`](F9_1A_WORKSPACE_REDUCTION_REPORT_01.md))  
**Resolución de la Dirección:** [`docs/private/F9_1B_DIRECTOR_RESOLUTION_AND_OPENING_01.md`](private/F9_1B_DIRECTOR_RESOLUTION_AND_OPENING_01.md)  
**Dictamen del Consejero:** [`docs/private/F9_1B_CONSULTANT_OPINION_01.md`](private/F9_1B_CONSULTANT_OPINION_01.md)  

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
* **Límite Operacional del Proyecto:** $\text{Peak NVML} < 4,800.0\text{ MB}$

---

## 2. Hipótesis Causal F9-1B

> **Hipótesis:** La activación aislada de la política de memoria virtual y segmentos expandibles del `CUDACachingAllocator` (`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`) reducirá de manera causal, física y reproducible el pico total de memoria física NVML y el footprint de Categoría 4 (Allocator FreePool / Fragmentación) respecto del control contemporáneo, manteniendo la selección nativa de kernels de PyTorch/cuDNN (sin restricciones de workspace de cuBLAS/cuDNN) y sin degradar la cadencia de inferencia en más de un 15%.

---

## 3. Definición Rigurosa de Intervención vs. Control

Para garantizar el aislamiento causal estricto del Allocator y no repetir la interferencia observada en F9-1A:

| Dimensión de Runtime | Condición Control ($C$) | Condición Intervención ($I$) | Justificación de Aislamiento |
| :--- | :--- | :--- | :--- |
| **`PYTORCH_CUDA_ALLOC_CONF`** | *Default (no definido / estándar)* | `"expandable_segments:True"` | **Única variable experimental manipulada.** |
| **`CUBLAS_WORKSPACE_CONFIG`** | *Default (no definido)* | *Default (no definido)* | Intocado para permitir kernels nativos óptimos. |
| **cuDNN SDPA & Backend** | `enable_cudnn_sdp(True)` / nativo | `enable_cudnn_sdp(True)` / nativo | Intocado para evitar fallbacks costosos. |
| **`torch.backends.cudnn.benchmark`** | `False` (o estándar PyTorch) | `False` (o estándar PyTorch) | Idéntico y simétrico en ambas condiciones. |
| **`torch.backends.cudnn.deterministic`** | `False` (o estándar PyTorch) | `False` (o estándar PyTorch) | Idéntico y simétrico en ambas condiciones. |
| **Precision & Pipeline Dtype** | FP16 DiT / BF16 VAE Tiled | FP16 DiT / BF16 VAE Tiled | Idéntico e inmutable. |

---

## 4. Verificaciones Técnicas Previas al Congelamiento (V1 – V8)

### V1. Datos Primarios de Categoría 4 (F9-0 y F9-1A)
Recuperados de [`logs/f9_0_attribution_telemetry.json`](../logs/f9_0_attribution_telemetry.json) y [`logs/f9_1a_telemetry.json`](../logs/f9_1a_telemetry.json):
* F9-0 Run 1: `3,173.9400 MB`
* F9-0 Run 2: `451.6200 MB`
* F9-0 Run 3: `451.6100 MB`
* F9-0 Run 4: `451.6200 MB`
* F9-0 Run 5: `451.6200 MB`
* Controles F9-1A (10 corridas): Mediana `451.6200 MB`

### V2. Cálculo Definitivo de $\epsilon$
* Mediana ($\text{Cat 4}$): $451.6200\text{ MB}$
* $\text{MAD}(\text{Cat 4})$: $0.0000\text{ MB}$
* Fórmula aplicada: $\epsilon = \max(64\text{ MB}, \text{mediana} + 3\times\text{MAD}) \implies \mathbf{64.0000\text{ MB}}$.

### V3 & V4. Determinism Check ($N=3$)
* Verificación bajo `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`:
* $\text{Max absolute diff } (R_1 \text{ vs } R_2): 0.000000\text{e}+00$
* $\text{Max absolute diff } (R_1 \text{ vs } R_3): 0.000000\text{e}+00$
* $\text{Max absolute diff } (R_2 \text{ vs } R_3): 0.000000\text{e}+00$
* **Max $|l_i - l_j|$ global:** $\mathbf{0.000000\text{e}+00} \le 10^{-3}$ (**PASSED / VALID**).
* **Régimen determinista:** Confirmado bit-exacto. La gestión virtual del allocator no altera las operaciones aritméticas de la GPU.

### V5 & V6. Secuencia de Bloques Balanceada y Hash Criptográfico
* **Estructura:** 10 bloques pareados (5 bloques $C \to I$, 5 bloques $I \to C$).
* **Secuencia fija:**
  `B1: I->C, B2: C->I, B3: C->I, B4: I->C, B5: I->C, B6: I->C, B7: I->C, B8: C->I, B9: C->I, B10: C->I`
* **Representación canónica:** `I->C|C->I|C->I|I->C|I->C|I->C|I->C|C->I|C->I|C->I`
* **Hash SHA-256 inmutable:**  
  `323fda6381c065db0c7dc72d2d665b84ae7828cb58beca52d49e6c7574580411`

### V7. Protocolo de Reset Simétrico entre Corridas
* Proceso limpio desacoplado por SO por cada corrida independiente.
* Ventana de reposo térmico y estabilización WDDM de 60 segundos pre-corrida ($S_0 \text{ spread} \le 150\text{ MB}$).
* Procedimiento de reinicio y sincronización simétrico e idéntico para condiciones $C$ e $I$.

### V8. Declaración Formal de Congelamiento
Completadas y registradas las verificaciones V1–V8, el presente protocolo queda formalmente **CONGELADO**.

---

## 5. Diseño Experimental y Análisis Bootstrap Pareado

Para cada bloque $k \in \{1 \dots 10\}$:
$$\Delta \text{PeakNVML}_k = \text{PeakNVML}_{I,k} - \text{PeakNVML}_{C,k}$$
$$\Delta \text{Cat4}_k = \text{Cat4}_{I,k} - \text{Cat4}_{C,k}$$
$$\Delta \text{Cat3}_k = \text{Cat3}_{I,k} - \text{Cat3}_{C,k}$$

* **Bootstrap pareado:** 10,000 remuestreos con reemplazo sobre las 10 diferencias pareadas $\Delta_k$.
* **Criterio A (Causal Primario):** $\text{mediana}(\Delta \text{PeakNVML}) < 0$ y $\text{Upper CI}_{95\%} < 0$.
* **Criterio B (Operacional):** $\ge 8 / 10$ corridas de intervención con $\text{PeakNVML}_I < 4,800.0\text{ MB}$.
* **Criterio C (Mecanístico Primario — Cat 4):** $\text{mediana}(\Delta \text{Cat4}) < 0$ y $\text{Upper CI}_{95\%} < 0$.
* **Criterio D (Regla Anti-Desplazamiento):**
  $$C_k = \begin{cases} \dfrac{\max(0, \Delta \text{Cat3}_k)}{|\min(\Delta \text{Cat4}_k, 0)|}, & \Delta \text{Cat4}_k < 0 \\ \text{NA}, & \Delta \text{Cat4}_k \ge 0 \end{cases}$$
  No más de 1 bloque ($10\%$) con $C_k \ge 0.5$.
* **No-Regresión de Cadencia:** $\Delta\text{Cadencia} \le 1.15\times$ (penalización $\le 15\%$).

---

## 6. Gobernanza

* Queda prohibida cualquier modificación del protocolo una vez congelado.
* Las 20 corridas confirmatorias se ejecutarán en estricto apego a la secuencia `323fda6381c065db0c7dc72d2d665b84ae7828cb58beca52d49e6c7574580411` tras autorización de la Dirección.
