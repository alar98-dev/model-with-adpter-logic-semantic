# Treinamento SWA — Gemma3-4B

## 1) Objetivo
Triplet Loss: Min $\|f(a)-f(p)\|$, Max $\|f(a)-f(n)\|$.

## 2) Triplets
- Fonte: `data/*.jsonl` ontologias.
- Ex: Âncora "Sol → calor", Pos "Radiação aquece", Neg "Sol esfria".

## 3) Loop
```python
for batch in loader:
    v_a, v_p, v_n = embed_gemma3(batch, 'models/gemma3-4b')
    fa, fp, fn = adapter(v_a), adapter(v_p), adapter(v_n)
    loss = triplet_loss(fa, fp, fn, m=0.2) + ortho_loss
    loss.backward()
```

## 4) Hiperparâms Gemma3-4B
- `d=3072`, `k_local=32`, `k_macro=32`.
- `batch=64` (RAM), `lr=1e-4`, `lambda_ortho=1e-3`.

## 5) Avaliação
- Dist A-P vs A-N.
- Validação lógica humana.

**Run:** `python swa/train_adapter.py --model-dir models/gemma3-4b`

---
*Gemma3-4B (2024).*

