# MARCO — Aeon como extensão do organismo sistêmico

> Data: 2026-08-04 · Origem única: `/home/alifer/swa-aeon`
> Contexto: o núcleo purificado (`aeonc.py`) deixou de ser um órgão isolado
> e passou a compartilhar o mesmo espaço físico do resto do sistema.

## O que mudou

O `aeonc.py` ganhou a classe `AeonOrganism`, com 4 nervos de integração
sobre o espaço compartilhado em `/dev/shm`:

| Nervo | Papel | Interface |
|---|---|---|
| Sangue | Liveness — o organismo sabe que o núcleo respira | `/dev/shm/aeon/core.heartbeat` + `core.pid`, batimento a cada 30s + após cada geração |
| Memória | Recuperação — o contexto do organismo é injetado na geração | `manifest.json` + `reality_tree.jsonl` + `swa_kernel_knowledge.json` |
| Expressão | Saída → memória — cada resposta vira um pensamento | `/dev/shm/aeon/core.thoughts.jsonl` (journal persistente) + nó `zone="aeon"` na árvore da realidade |
| Percepção | `/health` expõe o estado do organismo | nós, fonte de embedding, conceitos, pensamentos, métricas |

## Circuito fechado

```
sentir (memória) → pensar (LLM) → expressar (thought) → integrar (árvore) → verificar (imunológico)
```

1. **Sangue**: `touch_heartbeat()` grava JSON com pid, métricas e nome do modelo.
2. **Memória**: `organism_context(user_msg)` recupera por afinidade léxica os
   caminhos do `reality_tree.jsonl` + conceitos do kernel knowledge e injeta
   como bloco `[organismo]` no início das mensagens (RAG leve em CPU, sem encoder).
3. **Expressão**: `express(user_msg, reply)` grava `aeon://thought/<ts>` no
   journal (limitado a 400 linhas) e reflete um nó `RealityNode` válido
   (apenas os 9 campos — sem quebrar `iter_zone`/`resolve_interdim`) na árvore.
4. **Percepção**: `/health` agora retorna `metrics` + bloco `organismo`
   (heartbeat, pid, pensamentos, memória_nos, embedding, conceitos).
5. **Imunológico**: `obelisk-verifier.sh` ganhou `check_core_heartbeat` +
   `check_core_pid` — se o núcleo esteve vivo e parou de bater (>180s), o
   verifier reinicia `swa-aeonc.service`. Crashes são responsabilidade do
   `Restart=always` do systemd; o verifier só age sobre "núcleo vivo em coma".

## Rotas validadas (ponta a ponta)

- `GET /v1/models` → `gemma-3-1b-it`
- `POST /v1/chat/completions` sync + stream (SSE + `[DONE]`)
- `GET /health` → `{status, model, device, dtype, compile, metrics, organismo}`
- Geração usa a memória do organismo (ex.: resposta citou "508 Nós" e a fonte
  de embedding); cada turno escreve 1 pensamento + 1 nó AEON na árvore;
  heartbeat atualiza com `last_metrics`; verifier reporta "Core heartbeat: Xs
  atrás / Core PID alive".

## Notas técnicas

- **Nome do modelo no heartbeat**: guardado em `self.model_id` no `__init__`.
  `os.path.basename` sobre o modelo compilado (`torch.compile`) dispara
  `TypeError: Gemma3ForCausalLM does not support len()` — evitado.
- **Nós AEON na árvore são efêmeros**: o rebuild do manifold reescreve
  `reality_tree.jsonl`; o registro persistente dos pensamentos é o
  `core.thoughts.jsonl` (o journal do núcleo).
- **Baseline imune** não cobre `aeonc.py` nem `obelisk-verifier.sh` (só
  `cortex/obelisk/*` e `swa/*.py`) — não foi preciso regenerar.
- Serviços ativos: 8 serviços ativos (`swa-aeonc`, `swa-auto`, `swa-subcortex`, `swa-manifold`, `swa-mcp`, `swa-auditory`, `swa-obelisk`, `swa-obelisk-verifier`) + infra redis/postgres + `swa-backup.timer`.

## Próximos passos possíveis

- Persistência dos pensamentos além do journal (ex.: linha própria no
  `blake_semantic_anchors` via `sync_to_pgvector`).
- Embedding real para a recuperação (encoder leve em CPU) em vez de afinidade léxica.
- Injetar estado do observer (HW/load) no contexto `[organismo]`.
