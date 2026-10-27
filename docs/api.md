# API vigente do AEON

Base: `http://127.0.0.1:8050`

## Chat e integração OpenCode

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /api/v1/integrations/opencode/context`

O primeiro grupo é OpenAI-compatible e serve ao OpenCode. O segundo registra
contexto de sessão com BLAKE3 no ledger do AEON. Nenhum desses contratos
permite acesso direto a Redis, PostgreSQL, SHM ou execução arbitrária.

## Operação e auditoria

- `GET /health`
- `GET /api/v1/chat/state`
- `GET /api/v1/chat/events/:turn_id`
- `GET /api/v1/chat/training/:turn_id`
- `GET /api/v1/chat/archive/:turn_id/:wave_index`
- `POST /api/v1/chat/archive/replay/:turn_id`
- `POST /api/v1/chat/cancel`

O stream detalhado do ciclo OODA é `POST /api/v1/chat/stream`. O endpoint
OpenAI-compatible com `stream=true` é uma ponte de compatibilidade e devolve
um delta final único; ele não substitui a telemetria SSE nativa.

## Modelos locais

| Papel | Endereço | Modelo |
| --- | --- | --- |
| Gateway | `:8050` | AEON OODA |
| Córtex | `:20003` | Gemma 3 1B |
| Subcórtex | `:20002` | Gemma 3 270M / embeddings 640D |

Os documentos que mencionam `:20000` pertencem ao runtime histórico e estão
classificados em `DOCUMENTATION_STATUS.md`.
