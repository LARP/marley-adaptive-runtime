# F7 Stage C — Inter-Process Causal Intervention Report

**Date:** 2026-09-10T01:28:53.048722  
**Protocol Reference:** [`docs/F7_720P_PREREGISTERED_PROTOCOL_01.md`](F7_720P_PREREGISTERED_PROTOCOL_01.md) §5  
**Worker PID:** 45440 (Exit Code: 0)  
**Outcome:** **H3**  

---

## 1. Causal Intervention Verdict

> **H3 SUPPORTED: PERSISTENT EXTERNAL RESIDENCY. VRAM remained elevated at 1056.7 MB (+118.9 MB vs S0) after 60s post-mortem. Residency survives process destruction.**

### 1.1 Key Milestones

- **S0 pre-launch baseline (Monitor B):** 937.8 MB (spread 22.1 MB)
- **S3 plateau pre-kill (Worker A):** 1860.1 MB
- **Immediate 5s post-kill mean:** 1075.2 MB (delta vs S0: +137.3 MB)
- **Final 60s post-kill VRAM:** 1056.7 MB (delta vs S0: +118.9 MB)
- **Net Evaporated Memory:** **803.4 MB** (87.1% of residual residency)

### 1.2 Evaluation against Preregistered Hypotheses

| Hypothesis | Pre-registered Condition | Observed Result | Verdict |
| :--- | :--- | :---: | :---: |
| **H1: Process-Life-Cycle Dependency** | Drops to $\le S0 + 50$ MB within 5s | Delta: +137.3 MB | ❌ Falsified |
| **H2: Deferred OS Release** | Drops to $\le S0 + 50$ MB between 5-60s | Final Delta: +118.9 MB | ❌ Falsified |
| **H3: Persistent External Residency** | Remains elevated > S0 + 50 MB after 60s | Final Delta: +118.9 MB | ✅ **SUPPORTED** |

## 2. Epistemological and Engineering Implications

Se observa un remanente de residencia GPU externo al proceso (~118.9 MB), cuya implementación causal específica —DWM, superficies compartidas, WDDM u otro componente del stack gráfico— permanece abierta para investigación posterior. Sin embargo, el hecho empírico central es que el 87.1% (803.4 MB) de la meseta residual previa dependía estrictamente del ciclo de vida del proceso y fue recuperada inmediatamente por el sistema operativo al morir el contexto.
