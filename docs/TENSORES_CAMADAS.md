# TENSORES DAS CAMADAS — Indução de contexto compartilhado

> Marco: 2026-08-04 · Núcleo `aeonc.py` · Gemma-3-1B-it (26 camadas, hidden=1152)

## Sim, conseguimos compartilhar tensores pelas camadas

O `AeonCore` ganhou `--tensors`: um `forward_hook` no **último decoder layer**
captura o estado oculto do **último token** a cada forward pass da geração e o
publica em `/dev/shm` — o mesmo espaço que o resto do organismo (obelisk,
observer, manifold) já usa.

Canal de tensão:

```
model.model.layers[25]  →  forward_hook  →  hs[0,-1]  →  float32 CPU
        └──────────────────────────────────────────────────────────┐
/dev/shm/aeon/core.tensor.bin   (vetor 1152D, raw float32)         │
/dev/shm/aeon/core.tensor.json  (dims, source, norm, sha256, ts)   │
                                                                   ▼
                                    consumo: organismos futuros (pgvector,
                                    manifolds, verifier, observador)
```

## Como funciona

1. `--tensors` liga `tensor_sharing=True` → `_install_layer_hook()` registra
   o hook **antes** do `torch.compile` (para o Dynamo poder traçá-lo).
2. A cada forward pass, `_capture_hidden` salva `hs[0, -1]` (estado do último
   token = "o pensamento condensado" da camada final) em `self._last_hidden`.
3. Ao fim da geração, `_share_tensor(reply)` grava o vetor em
   `core.tensor.bin` + metadados em `core.tensor.json`.
4. Se o hook não disparou (ex.: grafo compilado sem hook), fallback
   `_embedding_of(reply)`: vetor médio da **camada 0** (`embed_tokens`) sobre
   os últimos 128 tokens da resposta. A origem fica registrada em `source`.

## Verificação real (validado)

```
Tensor hook instalado na camada 26 (hidden=1152)
Aeon: Python
core.tensor.bin  4608 bytes  (1152 × float32)
core.tensor.json dims:1152 | source:layer26 | norm:7646.74 | sha:f0f052e4…
```

## Notas e limites

- **Custo**: o hook é gratuito (apenas um slice + `.cpu()`). O fallback só
  roda se o hook não disparar.
- **Caveat do compile**: sob `torch.compile`, hooks Python podem não ser
  invocados em todos os passos — se `_last_hidden` vier `None`, usamos o
  embedding de entrada (camada 0). Ambos são "tensores das camadas".
- **Indução de contexto direta** (soft-prompt/prefixo via `inputs_embeds`)
  é possível mas fica como próximo passo: injetar o vetor capturado de volta
  como prefixo condicional na próxima geração é experimental e custoso em CPU.

## Próximos passos possíveis

- Projetar 1152→2560 (matriz fixa, como o `reality_embedder` 640→2560) e
  ingerir em `blake_semantic_anchors` via `sync_to_pgvector`.
- O observer/obelisk ler `core.tensor.json` para correlacionar estados
  ocultos com integridade (detecção de deriva de representação).
- CLI `--tensors` + `/org`-like toggle em runtime.
