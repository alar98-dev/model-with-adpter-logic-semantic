# K-Base — Evolução Estrutural e Ganhos Reais

**Data:** 05 de Agosto de 2026
**Sistema:** SWA-Aeon · núcleo `aeonc.py` · banco `swa_matrix`
**Arquiteto:** Alifer Ramos / Aeon

## 1. O Problema Estrutural que Motivou a Evolução

A matriz `blake_semantic_anchors` (334.236 âncoras) usava embeddings de 2560 dimensões
`embedding_2560` **hash-determinísticos** (sha256-seed): textos distintos com mesmo hash
geravam o mesmo vetor. Consequências:

- `nearest` entre textos distintos era **ruído** — não havia espaço semântico real.
- O índice HNSW antigo (`blake_semantic_hnsw_2560`) era **INVÁLIDO**: pgvector limita
  índices HNSW/IVFFlat a **2000 dimensões**. Vetores 2560d nunca puderam ser indexados.
- A recuperação dependia de heurística léxica sobre caminhos/manifold, sem cosseno real.

## 2. A Evolução Estrutural

### 2.1 Embedding real indexável
- Modelo: **Gemma-3-270M** (hidden 640d), mean-pool da última camada + L2-normalização.
- Espaço **640d nativo** é o indexável (pgvector). A projeção 2560d gold
  (`native_to_gold`, seed 137) permanece para compatibilidade/tensor, sem indexar.
- Serviço `swa-embedder` (systemd, `:21000`), dtype **fp32**: nesta CPU (Intel Core 5 120U,
  sem AVX512-bf16) bf16 **emula** e fica ~4x mais lento (164 vs 42 tokens/s).

### 2.2 Motor K-Base (`swa/kbase.py`)
- Schema `kbase_chunks`: `id, layer, source_path, collection, chunk_index, text,
  embedding_640 vector(640), embedding_2560 vector(2560), tokens_tsv tsvector,
  weight, created_at`; UNIQUE `(source_path, layer, chunk_index)`.
- Índices: `kbase_chunks_hnsw_640` (HNSW cosine), GIN `tokens_tsv`, btree `collection/layer`.
- **Multicamada [100, 250, 500, 1000]**: 1 embedding por fonte, reutilizado nas 4
  granularidades (janelas com sobreposição 15%, `MIN_CHUNK=40`).
- **Busca híbrida** `retrieve()`:
  - Denso: cosseno 640d (HNSW) — `_dense_top`.
  - Esparso: `tsvector` + `websearch_to_tsquery` (semântica OR, recall alto) — `_sparse_top`.
  - **Fusão RRF** estática (`k=60`) — `_rrf_fuse`.
  - **"The Great Filter"** — dedup por `source_path` em `Set` O(1), mantendo melhor score.
- Cache LRU (32 queries) com lock; `last_status` expõe estado/elapsed/hits.
- Fallback estrito: embedder fora → `state=sem embedder`, retorno vazio, sem crash.

### 2.3 Integração no núcleo (`aeonc.py`)
- `KBaseEngine` + `EmbedClient` com fallback gracioso (não carrega se `:21000` fora).
- Bloco `[kbase]` (fontes recuperadas com score/camada/snippet) injetado no
  `organism_context` de toda geração.
- Comando CLI `/kbase [on|off|busca]`; bloco `kbase` no `/health`.

## 3. Validação Ponta a Ponta

- `retrieve("azure client secret credential autenticação")` →
  `[0.75/CODE_BLOCK/250] mmap://…/identity-client/auth.py :: Microsoft Graph authentication…`
  — 146ms, fontes distintas, camadas variadas.
- Consultas de código: cosseno 640d 0.72–0.78 (afinidade real, determinística).
- Geração no núcleo (via API `:20000`): o Aeon **citou a fonte do corpus**
  (`/services/identity-client/graph_collector.py`) e foi honesto sobre limites do código —
  sinal de que o contexto recuperado guia a resposta, não apenas decora.
- `/health` reporta `kbase.ready`, `chunks`, `collections`, `layers`, `last_query`.

## 4. Ganhos Reais Medidos

| Métrica | Antes | Depois |
|---|---|---|
| Espaço semântico | hash-determinístico (ruído) | Gemma-3-270M 640d real |
| Índice vetorial | HNSW 2560d INVÁLIDO | HNSW cosine 640d válido |
| Busca híbrida | léxica por caminho | denso 640d + esparso tsvector + RRF |
| Dedup | N/A | Great Filter O(1) por fonte |
| Latência retrieve | — | 143–350ms (pós-embed; embed ~3s, cache LRU) |
| Camadas | única | 4 granularidades [100,250,500,1000] |
| Início (seed) | 0 | 401 fontes / 8.403 chunks / 2 coleções |
| Correções de robustez | — | `plainto_tsquery`(AND)→`websearch_to_tsquery`(OR); dims 640d consistentes no ORDER BY |

## 5. Estado da Execução e Próximos Passos

- Build completo do corpus curado (5.652 fontes) rodando em background
  (`swa-kbase-build`, systemd, Nice=19, **idempotente e resumível**) — ~5,5–8h na carga atual.
- WIKI_KNOWLEDGE (334k âncoras) permanece fora por custo de embedding; ativável via
  `--include-wiki` (curva de dias, não recomendado neste hardware).
- Próximo: ao fim do build, revalidar recall/recobrimento das 4 camadas e evoluir
  o `[kbase]` para seleção de camada por query (semântica vs léxica).

---
*A máquina observa. A máquina indexa. A máquina responde com a memória que ela mesma construiu.*
