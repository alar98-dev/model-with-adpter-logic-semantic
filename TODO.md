# SWA Aeon — TODO Geral

## Pendências Lógicas e Multimodais
- [x] Manter/ajustar a materialização para evitar parâmetros em meta e garantir que embeddings estejam materializadas (CPU/bf16).
- [x] Executar validação e confirmar que MISSING/"Some parameters are on the meta device" não aparecem.
- [x] Unificação e implementação da Onda Dissipativa Contínua (Hamiltoniano de Estado + Vontade Semântica Ψ).
- [x] Fissão Termodinâmica adaptativa guiada a nível-token (<end_of_turn>/<eos>).
- [x] Purificação total de diretrizes de texto e transição para Vácuo Sintático puro.
- [x] Acoplamento de 8 BladeTools no Córtex Motor com realimentação de microgravidade em malha fechada OODA.

## Integração Fissão ↔ EM Core (2026-08-25)
- [x] Gate condicional EM no motor de fissão (shield_throttle, matrix_stability, emotion_load)
- [x] Ψ efetiva modulada pela coerência do campo
- [x] Amplitude com acoplamento de energia real do retículo
- [x] Microgravidade com retenção emocional e decay (×0.85 + ×0.90)
- [x] Tensor de Direção: momentum (hidden_end - hidden_start) substituindo média estática
- [x] Phase-Lock Loop: âncora de intenção + correção vetorial anti-divagação
- [x] Válvula Phase-Lock Abort: cos < 0.1 aborta fissão (anti-degeneração)
- [x] Restaurar base_fission_demand com reached_token_ceiling

## Próximos Passos
- [x] Lifecycle de ondas EM: implementar garbage collection para ondas antigas (evitar 800+ ondas ativas) — `swa/em/wave_engine.gc()` (idade 45s, energia efetiva com envelope, cap 256), chamado no superpose e amortizado no create_wave, exposto em `get_state_snapshot.gc_removed`
- [ ] Formalizar equações da fissão termodinâmica para paper/whitepaper acadêmico
- [x] Implementar métricas de qualidade da fissão (ratio cos_sim médio por sessão) — `WaveState.cosine_similarity` + `ConversationTrajectory.quality()` + `RealityStore.session_quality` / `MetarealityOrgan.session_quality` expostos em `reinados.quality` no /health (live: mean_cos_sim 0.6516)
- [ ] Tunar limiar do Phase-Lock Abort (0.1 pode ser ajustado empiricamente)
- [ ] Explorar cos_sim como feature para classificação automática de completude de resposta
- [ ] Documentar a propriocepção sintética como capacidade emergente formal

## Metarealidade / Vontade (2026-09-08)
- [x] conversation_key no caminho híbrido → onda relacional (Tábua): `hybrid_generate(..., conversation_key)` + `event_source` passa `core._stable_conversation_key(req.messages)`
- [x] Rebuild k-base 339k em background (nice 19 / ionice 3, resumível, ~7 chunks/s)
- [x] `swa-auto` (2º kernel legado, 3.8G) parado e desabilitado — liberou RAM disponível p/ 3.9Gi
- [x] Pipeline E2E validado ao vivo (5 turnos): 209 palavras, 6 ondas relacionais por conversation_key, cos_sim real 0.4918 (2ª onda), drift 0.2541, ri médio 0.782, geração ~0.23–0.31 p/s, will_decision regrando (IMPULSO→RESPIRAR, reward 0.638/0.938)
- [x] Pacing do gate sob carga: `_load_factor()` (1.0→0.25 entre load 12–35) reprime o impulso e `_autonomous_breath` reduz o ritmo a 45s quando `/proc/loadavg` > 20 — o silêncio vira a postura de baixo custo sob pressão
- [x] **Intermitência SSE via dupla** atacada em 3 frentes (2026-09-08): (1) `FOLLOW`/`CONTINUAR`/`DIVERGIR` importados de `swa.wave` no `aeonc.py` — antes o nome `FOLLOW` lançava `NameError` no meio do turno nativo/híbrido (tela congelada com só parte do texto); (2) coordenador de prioridade `core._USER_TURN_ACTIVE`: a respiração autônoma cede o CPU enquanto um turno do usuário está gerando (fim da contenção que atrasava a 1ª linha); (3) handshake/lease single-flight já entregam 1º `data:` em <0.15s e o 2º stream do mesmo turno anexa ao mesmo buffer (A/B idênticos). Caveat conhecido: o 1º *token de conteúdo* leva ~39–60s (crawl k-base do híbrido / cold-boot).
- [x] Flats de `cos_sim`/`drift` na 1ª onda resolvidos com **alinhamento tensorial**: `observe` agora compara o embedding semântico real (via `:21000`, 640D) do turno com a âncora (`anchor_vec`) da última onda persistida do mesmo reinado; 1ª onda perde apenas a comparação (sem 0.5 fabricado) e persiste a âncora, a 2ª+ alinha de verdade. Provado E2E no PG real: drift 0.052 ↔ recompute cos 0.8964 (consistente).
- [x] **Dupla observação por turno** resolvida com observação idempotente: `observe()` agora deduplica por digest do turno (`sha256(conversation_key|snippet)`, `turn_digest` persistido no evento e na métrica, janela 300s via `RealityStore.metareality_seen`) e faz **auto-avanço da Tábua** (`RealityStore.next_wave_index`: cada turno do reinado sobe a onda em vez de o caminho híbrido sobrescrever sempre `wave=0`). Índice `idx_reality_events_meta_window` adicionado. Validado no PG real: pós-restart cada turno = 1 evento com digest + 1 onda com índice sequencial e cos real (reinado f50b6ea2 → ondas 0→1→2→3, cos 0.998/1.0 quanto à âncora da onda anterior); os pares duplicados pré-restart (634–665) não reapareceram. Teste isolado `single_observe_test.py` coberto (7 asserts).
- [x] **Onda dupla por turno colapsada** (2026-09-08): a fissão do `stream_messages` (`_persist_wave_state`) passava a gravar onda própria e o `observe` abria outra (`max+1`) — agora `RealityStore.latest_wave_index_if_recent(ck, 45s)` faz o `observe` reutilizar a onda de fissão recém-escrita e `append_wave_state(merge_metrics=True)` mescla os escalares de fissão com os semânticos da metarealidade (`cos_sim`/`reality_index`/`turn_digest`), preservando `event_id`. Validado no PG real: turno `01310e45` → onda única (id 361, índice 2) com métricas mescladas + evento `metareality` 764 pareado; `collapse_test.py` cobre o fluxo fissão→observe (PASS).

## Infraestrutura e Monitoramento
- [x] Correção de colisão de portas: Mover orquestrador OODA da porta 20000 para 20001.
- [x] Auditoria de toda a documentação markdown contra o ecossistema de 8 serviços.
- [x] Sistema de backups seguro em disco com assinaturas SHA256.
- [x] Monitorar o primeiro disparo automático diário do swa-backup.timer.
- [x] Monitorar a estabilidade da assinatura tensorial blake3 em ciclo de geração ativo real (1152d validado).
- [x] Validar a concorrência e o consumo de swap/RAM do host sob execução simultânea do core + loops OODA.
