# Semantic Warp Adapter (SWA) com Gemma3-4B — Documentação Técnica

## Resumo do Projeto

**Arquitetura Single-Model Otimizada para RAM:** Gemma3-4B (quantizada float16) + Semantic Warp Adapter (SWA).  
Projetamos vetores latentes da Gemma3-4B em hiperespaço lógico (causalidade, valência, temporalidade) sem alterar pesos base.  
Otimizado para 8-16GB RAM (CPU/Xeon), sem GPU obrigatória. Porta de serviço: **20000**.

**Objetivo:** Melhorar raciocínio lógico de SLMs via adaptação vetorial, com geração guiada por prefixo virtual.

## Diagrama de Fluxo de Dados

```
Texto → Tokenizer(Gemma3-4B)
     ↓
Embedding (mean pool, d=3072) → SWA Local (k=32) → SWA Macro (k=32, prefix_mode)
     ↓
LogicalEngine (rolling buffer + close_context) → v_global
     ↓
Geração (stream_with_prefix: cat(v_global + prompt_embeds) → Autoregressive)
     ↓
Resposta Eloquent (/api/v1/chat)
```

**Modo Dual (opcional):** Llama-3.1-8B (GPU, 4-bit NF4 bruto) + Gemma3-4B (CPU norm). Set `USE_DUAL_MODEL=true`.

## Guia de Configuração

### Pré-requisitos
- Python 3.10+, `pip install -r requirements.txt`
- Modelos: `models/gemma3-4b/` (HF format), opc. `models/Llama-3.1-8B-Instruct/` (dual).
- Docker: Para produção (veja abaixo).

### Docker
```bash
cp .env.example .env  # MODEL_DIR=models/gemma3-4b, GEMMA_DEVICE=cpu
docker-compose up -d
```

**.env chave:**
```
MODEL_DIR=/app/models/gemma3-4b
USE_DUAL_MODEL=false  # true para Llama+Gemmma
GEMMA_DEVICE=cpu
LLAMA_DEVICE=cuda     # se dual e GPU
PORT=20000
```

### Local
```bash
pip install -r requirements.txt
MODEL_DIR=models/gemma3-4b uvicorn swa.rest_server:app --port 20000
```

### Otimização RAM (Gemma3-4B)
- float16: ~8GB RAM.
- low_cpu_mem_usage=True (padrão).
- Teste: `curl -X POST http://localhost:20000/api/v1/utils/self_check`

## Endpoints da API

| Endpoint | Método | Descrição | Exemplo |
|----------|--------|-----------|---------|
| `/api/v1/chat` | POST | Chat stream com SWA + Gemma3 virtual prefix | `{"prompt": "Pergunta lógica?"}` |
| `/api/v1/process` | POST | Processa textos → vetores ancorados | `{"texts": ["um", "dois"]}` |
| `/api/v1/llama_bruto` | POST | Dual: Llama bruto (+opc norm Gemma3) | `{"prompt": "...", "normalize": true}` |
| `/api/v1/docs` | GET | Swagger UI | - |

**Exemplo curl chat:**
```bash
curl -X POST http://localhost:20000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explique causalidade."}' \
  --no-buffer
```

**Estrutura PyTorch:** Veja `docs/api.md`. Testes: `examples/run_quickstart.py`.

---
*Versão: Gemma3-4B RAM-Opt (2024). Contribuições bem-vindas!*

