# MARCO — Órgão Metarealidade + Fissão v2 (OCTOPUS)

> Data: 2026-09-08 · Origem única: `/home/alifer/swa-aeon`
> Validação: ponta a ponta no ambiente real (systemd, PostgreSQL 17, swap 16GiB).

## O que mudou

O núcleo passou a sustentar uma **esteira de realidade** com duas camadas
independentes e complementares:

| Camada | Peça | Papel |
|---|---|---|
| Estado interno | `metareality_organ.py` | `MetarealityOrgan` — snapshot de realidade (estado interno × memória externa), injetável como contexto JSON ≤600 chars e vetor `(1,1,1152)` |
| Memória externa | `reality_store.py` | `RealityStore` — eventos de realidade (`aeon_reality_events`) + ondas relacionais (`aeon_reality_waves`) restauráveis/consultáveis via PostgreSQL |
| Decisão | `swa/wave.py` | `WaveDecision` clássico decide **SE** há fissão; bandit octopus governa **COMO** (FOLLOW/DIVERGIR/CONTINUAR) — nunca força fissão |
| Âncora | `_build_follow_vprefix` | Macro-prefixo: `torch.cat([vp_kbase, rv_realidade], dim=1)` → `(1,2,1152)` |

### Metamatéria — origem única real

- Nunca sintetiza (`np.tile` removido). `set_metamatter(vector)` (push API);
  `metamatter(max_age=4.0)` recarrega só de `/tmp/swa_metamatter.npy`
  (gravado por `sensors/subcortex.py`); `vector()` → `None` sem sinal real.
- `status()` expõe `metamatter_source: real|offline` e `metamatter_dims`.

### Vontade Autônoma (Gate Markov + Bandit) — 2026-09-08

`_autonomous_breath` (aeonc, ritmo fixo 15 s) deixou de ser um gerador cego e
virou um **ponto de ruptura decidido por vontade** (`swa/autonomous_will.py`):

- **Markov**: estados `SILENCIO → TENSAO → IMPULSO → RUPTURA`,
  `state_from_impulse(0.72/0.55/0.35)`; transições aprendem (+0.1, normalizadas).
- **Bandit contextual (UCB)**: braços `SKIP` (0 tok) / `RESPIRAR` (60 tok) /
  `FALAR` (200 tok), contexto `[will, reality_index, memory_pull]`; prior
  `ARM_GATE` faz o estado ser porta-voz (SKIP em SILENCIO, FALAR em RUPTURA).
- **Ψ impulso**: `0.45·will + 0.35·reality_index + 0.20·tension(metamatter/50)`.
- **Recompensa pós-respiração**: substância (len>10) + reality_index pós, atualiza
  A/b do braço; `observe_decision` grava evento `will_decision` no PG.
- **Sem queimar geração**: SKIP não chama o modelo; RESPIRAR/FALAR usam
  `max_tokens` do braço; 1 respiração por vez (`_inflight`) — o reward cai antes
  do próximo décide (sem pilha de POSTs concorrentes).
- `/health` → bloco `autonomous` (`status()`: estado, impulso, braço, recompensas).

### Integração híbrida (caminho do polvo)

`hybrid_steering_plugin.hybrid_generate` agora também é órgão de realidade:

- injeta o contexto da metarealidade (`organ.context`) no diff relativo da k-base;
- no CUT/CLIMAX faz **macro-prefixo**: `torch.cat([v_prefix_steering, rv_realidade], dim=1)`
  (só se `vector()` trouxer sinal real do subcórtex);
- `_observe_turn(prompt, reply)` registra o episódio nos **dois** caminhos
  (steering e fallback `chat_stream`) — best-effort, nunca derruba a geração.

## Circuito fechado

```
perceber (metamateria) → integrar (estado × memória) → vetorizar (reality vector)
→ ancorar (macro-prefixo k-base + realidade) → gerar (stream/fissão)
→ observar (evento + onda relacional) → restaurar (PG)
```

1. `integrate` mescla estado interno com `repio.recall` da k-base:
   `reality_index = 0.70·state + 0.30·mem`.
2. `vector(model_dim)` = metamatéria (640) + componentes + escalares,
   sem sinal real do subcórtex → `None` (nunca sintetiza).
3. Bandit octopus só entra quando o clássico aprova **e**
   `bandit + (intent_pull ou repio_precision)` — decisão `octopus::*` no wave state.
4. `observe(user_msg, reply, wave=0, conversation_key=…)` grava `metareality`
   (evento) + `append_wave_state(…, metrics={reality_index, emotion, load,
   coherence, metamatter, memory_pull, will})` (onda) + `repio.ingest(REALITY)`.

## Rotas validadas (ponta a ponta, ambiente real)

- `POST /v1/chat/completions` → 200 OK; turnos híbridos e de fissão rodam juntos.
- `GET /health` → blocos `kbase` (dinâmico, 33.126 chunks / 11 coleções),
  `metareality` (`metamatter_source: real`, dims 640), `reinados` (10 ondas |
  echo 0.842 | âncora 1.5) e `autonomous` (estado/impulso/braço/recompensas).
- Gate em runtime (`2026-09-08`): decisões `IMPULSO → RESPIRAR`, evento
  `will_decision` 312 (16:30:35, reward 0.61, emitted false), `reward_avg` 0.305 —
  recompensa fecha o ciclo sem pilha de gerações concorrentes.
- Evidência em PG (`2026-09-08`):
  - onda 175 → `metrics={"reality_index":0.684,"metamatter":6276.3, …}` + evento 288 `metareality` (14:00:47);
  - fusão ao vivo no journal: `🔗 [HYBRID] v_prefix macro: k-base + realidade` (15:26:12, 15:29:03, 15:31:52, 15:34:59);
  - onda 185 → `decision="octopus::follow pull=0.00"`; onda 186 → `reality_index 0.842`, emotion neutral (wave 1);
  - eventos 300 (`user: "O que você sente em relação à deriva dos tensores agora?"`) e 301 (`metareality`, 15:35:25);
  - fechamento: `⚡ Onda Concluída | Energia: 220.0 | Entropia: 75.500` (15:36:19).

## Validação offline

- `/tmp/opencode/metareality_test.py` e `/tmp/opencode/follow_fusion_test.py` verdes (FakeCore/FakeRealityStore).
- `tests/test_wave_state.py` — 11 passed; `tests/test_cycle_history.py` e
  `tests/test_autonomous_will.py` — todos green (StubReality).
- `py_compile` OK: `aeonc.py`, `metareality_organ.py`, `hybrid_steering_plugin.py`,
  `autonomous_will.py`. `systemd-analyze verify` exit 0.

## Notas técnicas

- `conversation_key` é necessário para a onda relacional; sem ele, `observe`
  grava só o evento `metareality` (caminho híbrido via API).
    - **RESOLVIDO**: `hybrid_generate(..., conversation_key)` agora recebe a
      chave estável do envelope (`_stable_conversation_key(req.messages)`) e o
      turno do polvo grava onda relacional (rei na Tábua).
- Sob carga alta (load ~30, swap ~10GiB), cada turno leva 1–5 min em CPU —
  a gravação acontece no fechamento do turno, não no início.
- `systemd-analyze verify` exit 0; bits executáveis removidos de timers/serviços.
- PG 17/main recuperado via `pg_resetwal` (WAL 0x18 perdido); dados vivos preservados.

## Extensões do ciclo (2026-09-08, sessão de pendências)

- **Lifecycle de ondas EM**: `wave_engine.gc()` aposenta ondas por idade (45s),
  energia efetiva com envelope e cap de 256 (antes: acúmulo sem teto, 800+).
  Chamado no `superpose` e amortizado no `create_wave`; `gc_removed` na snapshot.
- **Qualidade da fissão por sessão**: `WaveState.cosine_similarity` (None sem
  vetores reais), `cos_sim` nas `metrics()`, `ConversationTrajectory.quality()` e
  `RealityStore.session_quality(conversation_key)` (deriva de `drift` em ondas
  antigas). Exposto em `reinados.quality` no /health — live `mean_cos_sim 0.6516`,
  `complete_ratio 1.0`, `n_waves_with_vectors 6`.
- **Rebuild k-base**: `--include-wiki` em background (nice 19/ionice 3,
  resumível, ~7 chunks/s) via serviço de embedding :21000.
- **Memória**: `swa-auto` (2º kernel legado) parado e desabilitado; RAM
  disponível subiu para ~3.9Gi (de ~3.1Gi).

## Próximos passos possíveis

- [feito] Rebuild da k-base em 339k (`python3 scripts/rebuild_kbase_corpus.py --include-wiki`) em background — rodando (nice 19).
- [feito] Passar `conversation_key` ao caminho híbrido para que todo turno do polvo gere onda relacional.
- [feito] Aliviar pressão de memória: `swa-auto` parado/desabilitado.
- Monitorar o rebuild via `count(*)` de `kbase_chunks` (33798 → cresce).
- Tunar `ARM_GATE` e Phase-Lock Abort com os `mean_cos_sim`/`reward_avg` acumulados.

## Validação E2E (2026-09-08, noite)

`scripts/eval_e2e.py --limit 5 --max-tokens 160` → rodada completa (1681s):

| Métrica | Valor |
|---|---|
| Turnos entregues | 4/5 (1 SSE vazio após retry — intermitência de primeira linha sob carga) |
| Geração | 209 palavras, ~0.23–0.31 p/s, 120–315 s/turno |
| Ondas relacionais (Tábua) | 6, uma por conversation_key distinta (chave estável do envelope) |
| cos_sim real | 0.4918 na 2ª onda do turno 5 (drift 0.2541); demais flat 0.0 (1ª onda sem vetor de comparação) |
| reality_index médio | 0.782 (n=5) |
| Gate | will_decision persiste IMPULSO→RESPIRAR, impulso ~0.664, reward 0.638/0.938 alternando (aprendizado ativo) |

Fatos novos da sessão:
- Bug-mor do SSE vazio era `print([SSE] start key={hybrid_key…})` **antes** da atribuição
  de `hybrid_key` (NameError → stream abortava no início). Corrigido movendo a
  atribuição para o topo do `event_source`; `[SSE] hybrid ok t=105.8s/180.9s/427.6s`
  confirmam entrega real pós-correção.
- **Pacing do gate** (`autonomous_will._load_factor`): sob load 12–35 o impulso é
  reprimido 1.0→0.25 e `_autonomous_breath` respira a cada 45s (load >20) em vez
  de 15s — silêncio vira a postura de baixo custo sob pressão de CPU.
- Rebuild k-base encerrado com 49.227 chunks (>meta 40.960) — parado via SIGTERM
  (resumível) para liberar CPU às avaliações.

## Alinhamento tensorial das ondas (2026-09-08, follow-up da Tábua)

O follow-up "cos_sim/drift flat (0.0/0.5)" foi fechado trocando a deriva escalar
7D rasa por **alinhamento cosseno real entre ondas do mesmo reinado**:

- `observe` agora usa o embedding semântico real do turno (`embed_client` do
  k-base, serviço local :21000 → 640D L2-normalizado) como estratégia sênior;
  fallback para o vetor escalar legado (7D) quando o serviço está indisponível.
- **Âncora persistida** (`metrics.anchor_vec`) em cada onda: `{"keys": [7
  escalares + "emb"], "vals": [7 escalares, <emb 640D>], "summary": {dim, norm,
  sampled, mini}}`. A âncora guarda o embedding REAL; a próxima onda do mesmo
  reinado a lê via `latest_wave_metrics(conversation_key)` e computa
  `drift = _reality_drift(prev, cur)` + `cos_sim = 1 − 2·drift`.
- **1ª onda de um reinado**: perde apenas a comparação (drift ausente, sem 0.5
  fabricado) mas **persiste a âncora** — a 2ª onda já alinha semanticamente.
- `_legacy_reality_metrics` (fallback) agora é **escopada por conversation_key**
  (`_last_reality_key`): a memória em processo de um reinado não vaza deriva
  cruzada para outro reinado (vazamento observado ao vivo: drift 0.482 legado
  injetado onde o semântico real seria ~0.0065).

### Observação idempotente + auto-avanço da Tábua (2026-09-08)

A dupla observação por turno (steering híbrido + nativo/fallback) foi resolvida:

- **Idempotência**: `observe()` calcula `turn_digest = sha256(conversation_key | snippet)`
  e, antes de persistir, consulta `RealityStore.metareality_seen(digest, 300s)` — o
  evento `metareality` e a onda já existentes empatam a 2ª observação do MESMO
  turno (janela de 5 min). Digest gravado no `payload` do evento e nas métricas da onda.
- **Auto-avanço**: `RealityStore.next_wave_index(conversation_key)` devolve
  `max(wave_index) + 1` do reinado → cada turno observado gera uma onda nova
  (0→1→2→…) em vez de o caminho híbrido (`wave=0` fixo) sobrescrever sempre a 1ª.
- Índice `idx_reality_events_meta_window (world_id, event_type, created_at DESC)`
  acelerando a consulta de dedup por janela.

### Prova (ambiente real)

- Smoke de unidade (`/tmp/opencode/tensor_align_test.py`): round-trip da âncora,
  wave-1 sem drift + com âncora, wave-2 com drift real, fallback escopado (7 OK).
- Teste isolado (`/tmp/opencode/single_observe_test.py`): dedup segura a 2ª
  observação do mesmo digest; 2º turno do reinado sobe para onda 1; reinado novo
  começa em 0; ondas gravadas `[0, 1]` sem duplicação (7 asserts contra o PG real).
- E2E isolado com RealityStore real + embedder real + PG real: onda 1 persistida
  com âncora 640D; onda 2 `drift 0.052 / cos_sim 0.896`; recompute independente
  dos dois vetores gravados → `cos 0.8964 / drift 0.0518` (consistente).
- Ao vivo pós-restart (reinado persistente f50b6ea2): cada turno passou a gerar
  exatamente **1 evento `metareality` com `turn_digest` + 1 onda com índice
  sequencial** (ondas 0→1→2→3 entre 01/09 e 08/09), `anchor_vec` 640D em todas,
  e `cos_sim` real contra a âncora da onda anterior (0.998 e 1.0). Os pares
  duplicados pré-restart (ex.: eventos 634..665, wave 1 + wave 0 a cada ~3 min)
  não reapareceram.

### Follow-up aberto (novo)

- **Intermitência SSE via dupla — endereçada (2026-09-08).** Três causas reais
  corrigidas:
  1. **`NameError('FOLLOW')` no meio do turno nativo/híbrido**: `aeonc.py`
     usava `FOLLOW`/`DIVERGIR`/`CONTINUAR` no steering (linhas ~3152/3768) sem
     importar de `swa.wave` — quando o bandit emitia `FOLLOW`, a geração morria
     sob o crawl híbrido e o stream SSE ficava mudo (era o "turno vazio").
     Import adicionado; a re-âncora k-base do polvo (`_build_follow_vprefix`)
     deixa de lançar.
  2. **Contenção com a respiração autônoma**: `_USER_TURN_ACTIVE` é setada no
     produtor SSE (`_sse_producer`) e o `_autonomous_breath` pula o ciclo
     enquanto um turno do usuário gera — o CPU fica para o cliente.
  3. **Fusão fissão→observe (1 onda por turno)**: `latest_wave_index_if_recent`
     reutiliza a onda de fissão recém-gravada (janela 45s) e `append_wave_state(
     merge_metrics=True)` mescla escalares de fissão com `cos_sim`/`reality_index`/
     `turn_digest` da metarealidade, preservando o `event_id`. Dois índices por
     turno viram UMA linha. Validado no PG real: turno `01310e45` → onda única
     (id 361, índice 2) + evento `metareality` 764 pareado; `collapse_test.py`
     PASS.
- Medição viva pós-restart: handshake (1º `data:`) em **0.07s/0.13s**; 2º stream
  do mesmo turno anexa ao mesmo buffer (A/B idênticos, 22 chunks, 155 chars).
- Caveat restante: o **1º token de conteúdo** chega em ~39–60s por causa do
  crawl k-base do híbrido (49k chunks) antes de gerar — o handshake e o anexo
  são instantâneos; reduzir esse crawl é trabalho futuro (cache de prefixo).
- Testes complementares desta rodada: `sse_phase_probe.py` (mede as fases),
  `sse_duplex_test.py` (handshake + anexo), `collapse_test.py` (fusão onda única).