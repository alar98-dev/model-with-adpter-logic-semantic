# Guia OpenCode ↔ AEON

Este documento substitui a orientação histórica baseada em IPC Unix e Redis
direto. O contrato de integração atual é HTTP no gateway e MCP read-only para
observabilidade.

## Fluxo de geração

Configure o OpenCode com:

```json
{
  "provider": {
    "aeon": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8050/v1",
        "apiKey": "sk-aeon-local"
      },
      "models": {
        "aeon-gemma3-1b": {"name": "Gemma 3 1B IT — AEON Gateway"}
      }
    }
  }
}
```

O gateway recebe `/v1/chat/completions`, executa o caminho AEON e retorna o
contrato OpenAI. Não aponte o OpenCode para `:20003`: isso bypassa o AEON.

## Observabilidade

Ative o MCP local definido em
`/home/alifer/.config/opencode/opencode.jsonc`:

```json
"aeon-observability": {
  "type": "local",
  "command": ["python3", "/home/alifer/swa-aeon/opencode_aeon_mcp.py"]
}
```

As ferramentas consultam saúde, estado, eventos de turno, ondas duráveis e
exportação de dataset via HTTP. Elas não leem diretamente Redis, PostgreSQL,
`/dev/shm` ou arquivos do sistema.

## Contexto de sessão

O organ opcional `opencode_organ.py` lê o SQLite local do OpenCode em modo
read-only e envia contexto ao endpoint:

```text
POST http://127.0.0.1:8050/api/v1/integrations/opencode/context
```

O AEON calcula BLAKE3 e publica o evento no ledger. Esse é o sentido
OpenCode → AEON; o sentido AEON → OpenCode ocorre pelo resultado do provedor e
pelas consultas MCP.

## Regras de segurança

1. Não usar socket `/dev/shm/aeon` como contrato externo.
2. Não publicar diretamente em Redis.
3. Não acessar PostgreSQL ou arquivos do AEON a partir do OpenCode.
4. Não tratar o prefixo 1152D como K/V nativo ou peso aprendido.
5. Não iniciar tools arbitrárias nem treinamento automaticamente.

Para o mapa completo, consulte
`/home/alifer/swa-aeon/OPENCODE_AEON_INTEGRATION.md`.
