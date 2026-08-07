# Documentação de API — Semantic Warp Adapter + Gemma3-4B

**Compatível com GitHub Markdown e Swagger (/api/v1/docs).**

## Endpoints Principais

### 1. `/api/v1/chat` (POST) — Chat Stream com Lógica SWA + Gemma3-4B
**Diferença chave:** Processa lógico (SWA ancorado) + geração eloquente via prefixo virtual (v_global cat embeds).

**Payload de entrada (JSON):**
```json
{
  "prompt": "Explique causalidade em IA."
}
```

**Respostas:**
- **200 OK:** Stream SSE `data: {"token": "..."}` + `[DONE]`.
- **400:** `{"error": "Prompt required"}`
- **503:** Gemma não init (`use_gemma=True`).

**Exemplo curl (porta 20000):**
```bash
curl -X POST http://localhost:20000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "O que é causalidade?"}' \
  --no-buffer
```

### 2. `/api/v1/llama_bruto` (POST) — Dual-Model: Llama Bruto + Opc. Gemma3 Norm
**Uso:** Raciocínio bruto (Llama GPU) + normalização (Gemma3 CPU).

**Payload:**
```json
{
  "prompt": "Resolva: 2+2=?",
  "normalize": true  // opc.
}
```

**Respostas:**
- **200:** `{"raw": "...", "normalized": "..."}` (se normalize=true)
- **503:** Llama não carregado (`USE_DUAL_MODEL=true`).

**Exemplo:**
```bash
curl -X POST http://localhost:20000/api/v1/llama_bruto \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Texto bruto.", "normalize": true}'
```

### 3. `/api/v1/llama_stream` (POST) — Stream Llama 3.1-8B
**Payload:** Mesmo que `/llama_bruto`.

**Resposta:** SSE `data: {"token": "..."}`.

### 4. `/api/v1/process` (POST) — Processa Textos → Vetores Ancorados
**Payload:**
```json
{
  "texts": ["evento1", "evento2"],
  "use_fake": true  // dev
}
```

**Resposta 200:**
```json
{
  "results": [
    {"index": 0, "shape": [128], "norm": 1.23}
  ]
}
```

### 5. Utils
- `GET /api/v1/utils/self_check`: Diagnósticos engine.
- `GET /api/v1/docs`: Swagger interativo.

## Códigos de Resposta Comuns
- **200:** Sucesso.
- **400:** Payload inválido.
- **422:** Validação JSON.
- **500:** Erro interno (e.g., OOM).
- **503:** Modelos não carregados.

**Teste completo:**
```bash
docker-compose up &  # aguarde logs
curl http://localhost:20000/api/v1/docs  # Swagger
```

---
*Atualizado para Gemma3-4B (2024). Veja `swa/rest_server.py` fonte.*

