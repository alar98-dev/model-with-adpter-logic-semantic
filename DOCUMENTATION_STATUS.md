# Status documental do AEON

## Fonte canônica atual

O índice normativo do runtime atual é
`/services/aeon-gateway-rs/CANONICAL_ARCHITECTURE.md`. O repositório
`/home/alifer/swa-aeon` permanece workspace histórico/auxiliar e não é uma
origem alternativa do Gateway Rust.

Use estes documentos para operar e integrar o sistema:

1. `/services/aeon-gateway-rs/CANONICAL_ARCHITECTURE.md`
2. `/services/aeon-gateway-rs/PREFIX_CONTRACT_V1.md`
3. `/services/aeon-gateway-rs/ARCHITECTURE.md`
4. `/services/aeon-gateway-rs/PHASE_TRANSITION_RUNBOOK.md`
5. `/services/agent-dev/README.md`
6. `/services/agent-dev/DATASET_PIPELINE.md`
7. `/services/swa-narrative-orchestrator/docs/ARCHITECTURE_BASELINE.md`

## Runtime vigente

| Função | Serviço/endereço |
| --- | --- |
| Gateway soberano | `aeon-gateway-rs.service`, `127.0.0.1:8050` |
| Córtex | Gemma 3 1B, `127.0.0.1:20003` |
| Subcórtex | Gemma 3 270M, `127.0.0.1:20002` |
| OpenCode chat | `http://127.0.0.1:8050/v1` |
| OpenCode observabilidade | MCP local `aeon-observability` |
| Persistência operacional | Redis + PostgreSQL |
| Arquivo de ondas | sidecar append-only para auditoria/replay explícito |

## Documentos históricos

Arquivos em `docs/`, `deployments`, changelogs e relatórios anteriores que
mencionam `:20000`, `aeonc.py`, Qwen, FastAPI como gateway, SHM direto ou
Antigravity como runtime são históricos. Eles não devem ser usados para
iniciar o sistema atual. A arquitetura atual não remove esses relatórios
porque eles preservam decisões e evidências de evolução; este índice evita
que sejam confundidos com instruções operacionais.

## Treinamento

O dataset `aeon.tensor_training.v1` está em preparação. Não há treinamento,
LoRA, alteração de pesos nem Gemma 4 no runtime atual.
