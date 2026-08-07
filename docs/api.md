# Documentação de API — SWA Aeon

A API de desenvolvimento do SWA Aeon expõe endpoints OpenAI-compatíveis para o núcleo central e rotas de monitoramento/diagnóstico.

## Endpoints do Núcleo (Porta 20000)

Exposto pelo serviço `swa-aeonc.service` (`aeonc.py`), executando o Gemma-3-1B-it.

### 1. `/v1/chat/completions` (POST)
Geração de chat compatível com o formato OpenAI. Suporta streaming via Server-Sent Events (SSE).

**Payload de entrada (JSON):**
```json
{
  "model": "gemma-3-1b-it",
  "messages": [
    {"role": "user", "content": "Olá Aeon, como vai?"}
  ],
  "profile": "equilibrado",
  "stream": true
}
```

*Nota:* O campo `profile` é uma extensão do SWA e aceita: `preciso`, `equilibrado`, `explorativo`, ou `duo` (detalhes em `PERFIS_GERACAO.md`).

**Exemplo de requisição (cURL):**
```bash
curl -X POST http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-1b-it", "messages": [{"role": "user", "content": "Explique causalidade."}], "stream": true}'
```

### 2. `/health` (GET)
Retorna o estado de integridade do modelo, métricas de hardware e informações sobre a sincronização com o organismo SWA.

**Exemplo de resposta (JSON):**
```json
{
  "status": "healthy",
  "model": "gemma-3-1b-it",
  "device": "cpu",
  "dtype": "torch.bfloat16",
  "compile": false,
  "metrics": {
    "tokens_generated": 1024,
    "last_tps": 4.5
  },
  "organismo": {
    "heartbeat_ok": true,
    "last_heartbeat_seconds_ago": 12,
    "thoughts_count": 340
  }
}
```

### 3. `/v1/models` (GET)
Lista o modelo ativo carregado no núcleo.
```bash
curl http://localhost:20000/v1/models
```

---

## Endpoints do Orquestrador OODA (Porta 20001)

Exposto pelo serviço `swa-auto.service` (`cortex/run.py`). Contém as rotas clássicas do orquestrador e receptor de ações de controle.

### 1. `/api/v1/chat` (POST)
Processamento cognitivo fechado unificado (OODA Loop).
**Payload de entrada (JSON):**
```json
{
  "prompt": "Explique causalidade em IA."
}
```

---
*Atualizado para SWA-Aeon v2.0 (2026).*
