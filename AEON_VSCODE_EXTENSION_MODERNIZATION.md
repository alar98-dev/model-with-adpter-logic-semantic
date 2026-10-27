# Extensão VS Code AEON — contrato moderno

Atualização da extensão `aeon-vscode-extension` para a arquitetura atual do
AEON Gateway Rust.

## Contrato ativo

- Gateway padrão: `http://127.0.0.1:8050`.
- Chat OODA com streaming: `POST /api/v1/chat/stream`.
- Saúde: `GET /health`.
- Estado: `GET /api/v1/chat/state`.
- Compatibilidade de conclusão inline: `POST /v1/chat/completions`.
- Eventos SSE: tokens são lidos de `data.token`; eventos de fissão são
  encaminhados separadamente como diagnóstico.

## O que foi removido da integração de chat

A extensão não usa mais `:20000`, `/api/v1/inline_complete` nem envia o payload
legado `prompt/workspaceFiles` para um serviço inexistente. Também não acessa
Redis, PostgreSQL, SHM ou `llama-server` diretamente.

O contexto do editor é reduzido a workspace, arquivo ativo, linguagem, cursor
e seleção limitada; ele chega ao gateway como `grounding_context`, onde passa
pelas mesmas regras de contexto, memória semântica, BLAKE3 e fissão do chat
AEON.

## Configuração

```json
{
  "aeon.gatewayUrl": "http://127.0.0.1:8050"
}
```

Depois de atualizar a extensão instalada, use `Developer: Reload Window` no
VS Code para recarregar o host da extensão.
