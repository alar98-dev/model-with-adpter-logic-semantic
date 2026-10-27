# Integração OpenCode ↔ AEON

## Estado implementado

O OpenCode usa o gateway Rust do AEON como provedor OpenAI-compatible:

```text
OpenCode : /v1/chat/completions
        └──> AEON Gateway :8050
              ├── OODA/fissão e memória semântica
              ├── Córtex Gemma 3 1B :20003
              └── Subcórtex Gemma 3 270M :20002
```

O endereço histórico `127.0.0.1:20000/v1` não é mais usado. O OpenCode não
deve apontar diretamente para o Córtex, porque isso contornaria memória,
fissão, grounding, persistência e cancelamento do AEON.

## Contratos

### OpenCode → AEON: geração

- `GET /v1/models`
- `POST /v1/chat/completions`

O gateway converte `messages` para o contrato interno `ChatRequest`, preserva
system/history/última mensagem do usuário e registra a origem `opencode` no
estado do turno. A resposta retorna formato OpenAI e o header
`x-aeon-turn-id`.

Quando o cliente envia `stream=true`, a compatibilidade atual devolve um SSE
OpenAI válido em um delta final único. O streaming OODA detalhado, com eventos
de fissão e diagnóstico, continua em `POST /api/v1/chat/stream`.

### OpenCode → AEON: contexto da sessão

O organ opcional em
`/home/alifer/swa-aeon/cortex/organs/opencode_organ.py` lê a base local do
OpenCode em modo somente leitura e envia atualizações para:

```text
POST /api/v1/integrations/opencode/context
```

O gateway calcula `payload_blake3` e publica o evento no ledger
`aeon:opencode:events`. Esse caminho não executa comandos, não injeta texto
diretamente no KV cache e não concede ao OpenCode acesso a Redis, PostgreSQL,
SHM ou à árvore `/`.

### AEON → OpenCode: observabilidade

O servidor local
`/home/alifer/swa-aeon/opencode_aeon_mcp.py` expõe somente consultas
read-only ao OpenCode:

- `aeon_health`
- `aeon_chat_state`
- `aeon_chat_events(turn_id)`
- `aeon_training_export(turn_id)`
- `aeon_wave_archive(turn_id, wave_index)`

As ferramentas consultam apenas a API HTTP do gateway. O MCP não é um canal
de execução arbitrária.

## Configuração

As duas configurações abaixo foram alinhadas:

- `/home/alifer/.config/opencode/opencode.jsonc`
- `/home/alifer/swa-aeon/opencode.json`

Modelo configurado: `aeon-gemma3-1b`, base URL
`http://127.0.0.1:8050/v1`, API key local simbólica `sk-aeon-local`.

## Verificação

```bash
curl -s http://127.0.0.1:8050/v1/models | jq

curl -s http://127.0.0.1:8050/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer sk-aeon-local' \
  -d '{"model":"aeon-gemma3-1b","messages":[{"role":"user","content":"Responda apenas: AEON integrado."}],"stream":false}' | jq
```

O campo `aeon.turn_id` e o header `x-aeon-turn-id` permitem correlacionar a
resposta com `/api/v1/chat/events/:turn_id`, o arquivo de ondas e a futura
exportação `aeon.tensor_training.v1`.

## Verificação realizada

Em 16/10/2026, o comando real do OpenCode foi executado com
`swa-aeon/aeon-gemma3-1b`. O turno `chat-1792173963136-4` foi localizado no
ledger do AEON com evento `fission_synthesis`, `wave_index=0` e cadeia de
integridade BLAKE3. O resultado exibido pelo modelo foi uma resposta de
formatação de ferramenta, portanto a integração foi comprovada, mas a
obediência ao pedido literal deve continuar sendo avaliada como qualidade do
modelo/template, não como prova de transporte.

## O que continua fora do escopo

- Gemma 4 ainda não foi instalado nem substituiu o Gemma 3.
- O prefixo 1152D ainda é uma projeção determinística não treinada.
- A integração não altera pesos e não inicia fine-tuning.
- O dataset é preparação; a curadoria e o treinamento serão fases posteriores.
- Nenhuma tool arbitrária é acionada automaticamente por OpenCode ou AEON.
