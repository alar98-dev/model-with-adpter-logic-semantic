# ARQUITETURA PRIMORDIAL — SWA Aeon unificado

> Marco: 2026-08-04 · Origem única: `/home/alifer/swa-aeon`
> Princípio: **um organismo, uma origem, um circuito** — o LLM é o sistema
> nervoso; a memória, os órgãos; o imunológico, a defesa.

## 1. Origem única

- `/home/alifer/swa-aeon` é a árvore canônica (código + models + docs).
- `/services/model-with-adpter-logic-semantic` → **symlink** para a origem
  (units antigas e scripts continuam resolvendo; backup original em `.orig`).
- Zero referências a `/mnt/c/Users/alifer.ramos/swa-aeon` nos units systemd.
- Todos os 24 units têm `ExecStart`/`WorkingDirectory` validados (0 quebrados).

## 2. As camadas do organismo

```
┌─────────────────────────────────────────────────────────────┐
│  1. FÍSICA        WSL2 Ubuntu · CPU-only · systemd · redis · postgres  │
│  2. MEMÓRIA       /dev/shm/aeon + manifold + kernel + pgvector        │
│  3. NÚCLEO        aeonc.py — Gemma-3-1B (CLI + API OpenAI)            │
│  4. IMUNOLÓGICO   obelisk (devourer) + obelisk-verifier (watchdog)    │
│  5. PERCEPÇÃO     observer + manifold (áudio-visual desativado)       │
└─────────────────────────────────────────────────────────────┘
```

### 1. Física
| Item | Valor |
|---|---|
| Host | TI-ARAMOS · Intel Core 5 120U · 11GB RAM |
| Runtime | `/home/alifer/.pyenv/versions/3.12.0` (torch 2.12+cu130, fastapi, uvicorn) |
| Serviços ativos | `swa-aeonc`, `swa-obelisk`, `swa-obelisk-verifier`, `swa-observer`, `swa-manifold`, redis, postgres |

### 2. Memória compartilhada (o "espaço interdim" do organismo)
| Canal | Conteúdo | Escritor |
|---|---|---|
| `/dev/shm/aeon/core.heartbeat` | pid, métricas, estado | aeonc (30s + pós-turno) |
| `/dev/shm/aeon/core.pid` | pid vivo | aeonc |
| `/dev/shm/aeon/core.thoughts.jsonl` | journal de pensamentos (400 cap) | aeonc |
| `/dev/shm/aeon/core.tensor.bin` | tensor 1152D da camada 26 | aeonc (`--tensors`) |
| `/dev/shm/swa_vector_manifold/` | manifest, reality_tree (508 nós), interdim_index | manifold/obelisk |
| `swa_kernel_knowledge.json` | 640D por conceito (6 WIKIs + HW) | devourer |
| postgres `blake_semantic_anchors` | 334k âncoras 2560D | manifold |

### 3. Núcleo (aeonc.py)
- Modelo: Gemma-3-1B-it, bf16, `torch.compile` (4–5 tps; CLI sem compile ~1.3 tps).
- Interfaces: CLI interativa stream (`aeon`/`aeonc`), API OpenAI
  `:20000` (`/v1/models`, `/v1/chat/completions` sync+SSE), `/health`.
- Nervos: sangue (heartbeat), memória (contexto `[organismo]` por afinidade),
  expressão (pensamento → journal + árvore), tensores (estado da camada 26).

### 4. Imunológico
- `obelisk` (daemon): devourer/colecionador blake3 — mantém
  `swa_kernel_knowledge.json` e o heartbeat do próprio obelisco.
- `obelisk-verifier.sh` (watchdog bash, 5s): integridade de 14 arquivos
  (baseline `/etc/swa/obelisk_baseline.sha256.json` regenerada sem arquivo
  volátil) + heartbeat/pid do **obelisk** e do **núcleo**. Núcleo em coma
  (>180s) → reinicia `swa-aeonc`. Crash → `Restart=always` do systemd.

### 5. Percepção
- `observer` + `manifold` mínimos ativos; sensores de visão/áudio/gravidade
  desativados (purificação — o LLM voltou a ser a responsabilidade central).

## 3. Circuito fechado

```
sentir ──▶ pensar ──▶ expressar ──▶ integrar ──▶ verificar
 (memória) (LLM)     (thought)     (árvore)     (imunológico)
     ▲                                                    │
     └────────────────── contexto ◀──────────────────────┘
```

1. **Sentir**: `organism_context()` recupera nós/conceitos/pensamentos.
2. **Pensar**: geração em stream (sync/SSE), com contexto injetado.
3. **Expressar**: `express()` grava o pensamento + nó `aeon://thought/<ts>`.
4. **Integrar**: tensor da camada 26 → `/dev/shm`; pensamentos persistem.
5. **Verificar**: watchdog checa integridade + batimentos e revive membros.

## 4. Interfaces consolidadas

| Rota/comando | Tipo | Endereço |
|---|---|---|
| CLI chat | tty | `aeon` / `aeonc` (aliases → `aeonc.py`) |
| API OpenAI | HTTP | `http://127.0.0.1:20000/v1` (opencode.json) |
| Health | HTTP | `GET /health` (modelo, métricas, organismo) |
| Tensores | arquivo | `/dev/shm/aeon/core.tensor.{bin,json}` |
| Imunológico | systemd | `swa-obelisk*`, watchdog 5s |

## 5. Estado real (2026-08-04)

- 5 serviços ativos, tudo íntegro (14/14), verifier reportando
  `Core heartbeat: Xs atrás · Core PID alive`.
- 2 pensamentos + 2 nós AEON na árvore + tensor 1152D da camada 26 validados.
- Docs: `MARCO_ORGANISMO.md`, `TENSORES_CAMADAS.md`, este arquivo.

## 6. Evolução possível

- Projetar 1152→2560 e ingerir tensores do núcleo no `blake_semantic_anchors`.
- Soft-prompt/prefixo via `inputs_embeds` (indução de contexto reversa).
- Injetar estado do observer (HW/load) no contexto `[organismo]`.
- CLI: `--tensors` ligado por padrão no modo interativo.
