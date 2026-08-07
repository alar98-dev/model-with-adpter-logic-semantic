# TODO — Sistema Imune Lógico (Obelisco + Blake Gravity)

## 📋 Legenda

- 🔴 Crítica — bloqueia segurança do sistema
- 🟡 Importante — melhora funcionalidade
- 🟢 Desejável — expande capacidade
- ⚪ Experimental — pesquisa futura

---

## FASE 0 — BUGS CONHECIDOS

- [🔴] **10 torch.compile workers** — main.py lança 10 workers do inductor
      usando ~1.5GB RAM extra. Adicionar variável de ambiente:
      `TORCH_COMPILE_WORKERS=1` ou `--disable-compile` no model.py
- [🟡] **Loop bloqueante** — `time.sleep(30)` em main.py impede resposta
      a sinais. Migrar para `select` ou `asyncio` com timeout
- [🟡] **LoRA adapter sobrescrito** — train.py salva adapter no mesmo
      diretório sem backup. Adicionar timestamp no nome do adapter

---

## FASE 1 — CORREÇÕES CRÍTICAS (IMUNIDADE BÁSICA)

### 1.1 Self Baseline

- [🔴] Criar `scripts/generate_baseline.sh` — gera BLAKE3 hash tree de
      todos os arquivos monitorados no deploy
- [🔴] Modificar `collector.py` — comparar hashes com baseline, não com
      string vazia. `hash_entropy = mismatch_count / total`
- [🔴] Baseline versionada — se código mudou legitimamente (git pull),
      re-gera baseline automaticamente
- [🟡] `self_baseline.json` — armazenar em `/etc/swa/obelisk_baseline.json`
      com permissão read-only para o processo

### 1.2 Proteção do Próprio Obelisco (Watchdog do Watchdog)

- [🔴] Criar `scripts/obelisk-verifier.sh` — script bash puro (sem Python)
      que verifica:
      - Hashes BLAKE3 dos 10 arquivos do obelisco vs baseline
      - Heartbeat atualizado nos últimos 60s
      - PID do processo corresponde ao PID file
- [🔴] Criar `deploy/swa-obelisk-verifier.service` — systemd que executa
      o verificador, com `Restart=always` e `RestartSec=5`
- [🔴] Se verificar detectar mutação → `systemctl restart swa-obelisk`
- [🟡] Se verificar detectar falha do obelisco 2 ciclos seguidos →
      restaurar de backup + `systemctl restart swa-obelisk`
- [🟡] Verificador também monitora a si mesmo (hash do próprio script)

### 1.3 Self Tolerance (Prevenção de Autoimunidade)

- [🟡] Validação cruzada 5-fold no dataset antes de treinar — se F1 < 0.9
      em alguma classe, NÃO atualiza o adapter (anergy)
- [🟡] Threshold de confiança mínimo para ação de rollback: exigir
      probabilidade > 0.7 (se LLM der logits com softmax)
- [🟡] Log de falsos positivos: se humano reverter um rollback automático,
      registrar como falso positivo e excluir do próximo treino

---

## FASE 2 — SINALIZAÇÃO E MEMÓRIA (IMUNIDADE ADAPTATIVA)

### 2.1 DAMP Broadcast

- [🔴] Novo tipo de evento IPC: `DANGER_SIGNAL` (além de `OBELISK_HEARTBEAT`)
- [🟡] Payload: `source`, `tensor`, `severity` (0.0-1.0), `label`, `timestamp`
- [🟡] Broadcast para todos os serviços SWA que escutam o IPC socket
- [🟢] swa-dream: ao receber danger, entra em modo reparo (dream repara)
- [🟢] swa-subcortex: ao receber danger, isola módulo afetado

### 2.2 Costimulation (Dupla Verificação)

- [🔴] Rollback só executa se AMBOS concordarem:
      Verificador A: `BlakeGravityTensor.classify()` onboard
      Verificador B: `ObeliskModel.classify()` via LLM
- [🟡] Se discordarem → log + emit `DANGER_SIGNAL` com severity=0.5
      (alerta, não ação) + incrementa contador de falso positivo
- [🟡] Se discordarem 3× consecutivas → `swa-obelisk-train.service`
      executa treino emergencial com dados atuais

### 2.3 Memória Imunológica

- [🟡] Criar `/var/lib/obelisk/immune_memory/` — diretório persistente
- [🟡] Cada ciclo de classificação salva:
      `{timestamp}_{label}.json` contendo tensor, telemetria, decisão
- [🟡] `dataset.py` incluir ultimas 500 amostras reais no próximo treino
      (concatena templates sintéticos + logs reais + memória imunológica)
- [🟢] `scripts/immune_query.py` — busca por similaridade de tensor:
      dado tensor atual, encontra top-5 patógenos históricos mais similares
- [🟢] Detecção de surtos: se 5+ tensores similares em 1h → emitir alarme

### 2.4 Pool de Anticorpos (LoRA Multi-Adapter)

- [🟡] Criar 6 adapters LoRA rank=8, um por classe:
      `models/obelisk-lora-adapter/nominal/`
      `models/obelisk-lora-adapter/lethal_mutation/`
      ...
- [🟡] Router leve (RandomForest ou regras) que seleciona adapter:
      ```python
      def select_adapter(tensor):
          if tensor[0] > 0.4 and tensor[3] > 0.3: return "lethal_mutation"
          if tensor[2] > 0.5: return "thermal_spike"
          ...
      ```
- [🟢] Treino independente para cada adapter (sem competição por rank)
- [🟢] Se adapter de classe rara (`regression_to_rlhf`) tem F1 < 0.8,
      gerar templates sintéticos extras para balancear

---

## FASE 3 — EXPANSÃO (IMUNIDADE DE REBANHO + OBSERVABILIDADE)

### 3.1 Imunidade de Rebanho

- [🟢] Cada serviço SWA (dream, subcortex, MCP) roda mini-obelisco
      leve (só fallback onboard, sem LLM)
- [🟢] Tensores compartilhados via MCP/SSE — todos os mini-obeliscos
      consultam uns aos outros
- [🟢] `ConsensusClassifier` — decisão final por votação ponderada:
      peso maior para obeliscos com mais tempo de vida sem erro
- [⚪] Se 51%+ dos mini-obeliscos detectam anomalia → rollback mesmo
      sem confirmação do obelisco principal

### 3.2 Observabilidade

- [🟢] Tensor stream via WebSocket em tempo real
- [🟢] Gráfico radar 6-eixos (atual vs histórico) para dashboard
- [🟢] Timeline de mutações: `timestamp → label → ação → resultado`
- [🟢] Alerta via WebSocket se severity > 0.7

### 3.3 Bug Bounty Interno (Chaos Testing)

- [🟢] Script `scripts/chaos_injector.py` que simula mutações:
      - Corrompe arquivo aleatório (zera hash)
      - Remove nó do manifold (/dev/shm)
      - Simula HW pressure (remove /dev/sde)
      - Injeta dados monotípicos no structural_origins
- [🟢] Verificar detecção em < 30s (1 ciclo de heartbeat)
- [🟢] Verificar que falso positivo < 1% (sem mutação → nominal)

### 3.4 Artigo Científico

- [🟢] Executar coleta de dados reais por 30 dias para seção experimental
- [🟢] Ablação: comparar F1 de fallback onboard vs LLM 4-bit vs MLP
- [🟢] Exportar `docs/POC_OBELISK_BLAKE_GRAVITY.md` para LaTeX
- [⚪] Submissão ICML 2027 / NeurIPS 2027 / ISCA 2027

---

## PRIORIDADE GERAL

```
FASE 1 ───────────────────────────────────────►  FAZER AGORA
  ├─ 1.1 Self baseline                          🔴 2h
  ├─ 1.2 Verificador (watchdog do watchdog)      🔴 2h
  └─ 1.3 Self tolerance                          🟡 1h

FASE 2 ───────────────────────────────────────►  PRÓXIMA
  ├─ 2.1 DAMP broadcast                          🟡 2h
  ├─ 2.2 Costimulation                           🟡 2h
  ├─ 2.3 Memória imunológica                    🟡 3h
  └─ 2.4 LoRA multi-adapter                     🟡 4h

FASE 3 ───────────────────────────────────────►  FUTURO
  ├─ 3.1 Imunidade de rebanho                   🟢 1 semana
  ├─ 3.2 Observabilidade                        🟢 3 dias
  ├─ 3.3 Chaos testing                          🟢 2 dias
  └─ 3.4 Artigo                                 🟢 1 mês
```

## RESUMO POR ARQUIVO

| Arquivo | Ação |
|---------|------|
| `cortex/obelisk/blake/collector.py` | self baseline (comparar com referência) |
| `cortex/obelisk/blake/tensor.py` | sem mudanças (já correto) |
| `cortex/obelisk/blake/classifier.py` | costimulation (dois verificadores) |
| `cortex/obelisk/model.py` | disable-compile flag, confidence threshold |
| `cortex/obelisk/main.py` | DAMP broadcast, costimulation, self tolerance |
| `cortex/obelisk/dataset.py` | immune memory loader (últimas 500) |
| `cortex/obelisk/train.py` | multi-adapter (6 LoRAs), timestamped backups |
| — novo — | `scripts/generate_baseline.sh` |
| — novo — | `scripts/obelisk-verifier.sh` |
| — novo — | `deploy/swa-obelisk-verifier.service` |
| — novo — | `cortex/obelisk/immune/immune_memory.py` |
| — novo — | `cortex/obelisk/immune/router.py` |
| — novo — | `scripts/chaos_injector.py` |
