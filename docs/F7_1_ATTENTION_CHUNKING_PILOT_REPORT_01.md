# Informe de Piloto Experimental: Phase F7-1 (Attention Sequence Chunking)

**Fecha:** 2026-09-09T02:35:35.824229  
**Resolucion:** 1280x720 @ 33 frames (5 pasos), modo `adaptive`  
**Chunk Size:** C = 2048  

**Clasificacion de evidencia:** `MEASURED` = instrumentado en esta corrida; `OBSERVED` = telemetria muestreada; `DERIVED` = calculado; `HYPOTHESIS` = expectativa no demostrada.

---

## 1. Hechos Medidos / Observados

| Metrica | Clase | Valor | Baseline F7-0 |
| :--- | :---: | :---: | :---: |
| Peak NVML fisico | OBSERVED | 5968.4 MB | 6058.5 MB |
| PyTorch Allocated | OBSERVED | 2155.6 MB | 2187.5 MB |
| PyTorch Reserved | OBSERVED | 5742.0 MB | 5894.0 MB |
| Host RSS | OBSERVED | 162.2 MB | ~2,150 MB |
| Wall-clock total | OBSERVED | 142.83 s | 521.93 s |
n/a (aborto antes de completar 2 pasos) | 59.93 s/step (F7-D1) |
| NaN/Inf | MEASURED | 0 NaNs / 0 Infs | 0 |
| MP4 valido | OBSERVED | No/No generado | Si |

### Equivalencia numerica chunked vs monolitico (MEASURED, sintetico)

* Max abs diff: 0.000e+00
* Diff relativa: 0.000e+00
* Cosine similarity: 1.000000
* Bit-exacta: True  (NO se afirma equivalencia bit-exacta)
* Config: seq=8192, heads=12, head_dim=128, chunk=2048, dtype=torch.float16

### Atencion interceptada en runtime (MEASURED)

| Tag | Variante | S real | Heads | Head_dim | dtype | Backend cfg | Chunked |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| block0.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block0.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block1.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block1.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block2.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block2.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block3.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block3.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block4.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block4.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block5.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block5.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block6.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block6.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block7.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block7.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block8.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block8.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block9.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block9.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block10.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block10.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block11.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block11.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block12.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block12.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block13.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block13.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block14.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block14.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block15.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block15.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block16.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block16.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block17.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block17.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block18.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block18.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block19.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block19.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block20.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block20.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block21.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block21.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block22.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block22.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block23.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block23.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block24.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block24.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block25.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block25.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block26.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block26.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block27.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block27.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block28.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block28.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |
| block29.attn1 | self | 32400 | 12 | 128 | torch.float16 | None | True |
| block29.attn2 | cross | 32400 | 12 | 128 | torch.float16 | None | True |

---

## 2. Comparaciones (DERIVED)

* Reduccion Peak NVML vs F7-0: **+90.1 MB**
* Resultado clasificado: **C** (sin efecto material sobre el Peak NVML (abortado por salvaguarda operativa cercana al limite fisico de 6 GB))

---

## 3. Inferencias (DERIVED)

La memoria se mantiene en el regimen de F7-0. La hipotesis de que la atencion es el evento dominante del pico debe reconsiderarse.

---

## 4. Hipotesis (NO demostradas)

* Que el chunking ataca el evento temporal que produce la expansion del pool del allocator.
* Que S = grid espacial 14,400 coincide con la secuencia real de self-attention (a verificar por la telemetria de atencion medida).

---

## 5. Recomendacion (para decision del Director)

El piloto NO autoriza por si mismo cambios permanentes al runtime. Se remite a la decision del Director: integrar C=2048, repetir, cambiar chunk, abandonar linea o investigar otra causa.
