# Semantic Warp Adapter (SWA) com Gemma-3-1B-it — Documentação Técnica

## Resumo do Projeto

**Arquitetura Purificada baseada em systemd nativo (WSL2):** Gemma-3-1B-it (Núcleo Central, bf16 + `torch.compile`) + Gemma-3-270m (Subcórtex e Obelisk Watchdog).  
Projetamos e expomos o estado latente dos tensores do núcleo (camada 26) via memória compartilhada (`/dev/shm`), permitindo que o orquestrador autônomo OODA (`run.py`) consuma essa gravidade semântica para modular suas ações.

Otimizado para execução em CPU com baixo consumo de recursos (11GB RAM), sem GPU obrigatória.
- **Porta do Núcleo (API OpenAI & Health):** `20000`
- **Porta do Orquestrador OODA (REST/CLI):** `20001`
- **Porta do MCP Gateway (FastAPI/SSE):** `8000`

## Diagrama de Fluxo de Dados

```
Usuário/API ──▶ CLI (aeon) ──▶ aeonc.py (Gemma-3-1B-it, porta 20000)
                                     │
                                     ├─ (grava tensor camada 26 blake3 assinado)
                                     ▼
                      /dev/shm/aeon/core.tensor.bin
                                     ▲
                                     │ (leitura assíncrona do estado latente)
                                     │
Loop OODA (Observe) ───────────────▶ run.py (swa-auto, porta 20001)
```

## Guia de Configuração

### Pré-requisitos
- Python 3.12 (gerenciado via `.pyenv` no ambiente de produção).
- Modelos HF: `models/gemma-3-1b-it` (núcleo) e `models/gemma-3-270m` (subcórtex/obelisk).
- Redis e PostgreSQL rodando localmente no host.

### Execução via Systemd (Produção)
O ecossistema é gerenciado por 8 serviços do systemd. Para habilitar e iniciar todos os serviços:
```bash
sudo systemctl enable --now swa.target
```

Você pode checar o status de todas as unidades rodando:
```bash
systemctl list-units --type=service | grep swa
```

### Execução Local (Desenvolvimento)
Para iniciar o núcleo manualmente (CLI interativa com tensores e compilação ativos):
```bash
python3 aeonc.py --compile --tensors
```

Para iniciar a API do núcleo na porta 20000:
```bash
python3 aeonc.py --api --host 127.0.0.1 --port 20000 --compile --tensors
```

## Estrutura de Portas e Endpoints

| Componente | Porta | Endereço | Principais Endpoints |
|------------|-------|----------|----------------------|
| **Core (aeonc.py)** | 20000 | `http://127.0.0.1:20000` | `/v1/chat/completions` (OpenAI stream), `/health`, `/v1/models` |
| **OODA (run.py)** | 20001 | `http://127.0.0.1:20001` | REST endpoints clássicos do orquestrador |
| **MCP Server** | 8000 | `http://127.0.0.1:8000` | `/sse`, `/messages/` (Starlette) |

---
*Versão: SWA-Aeon v2.0 (2026).*
