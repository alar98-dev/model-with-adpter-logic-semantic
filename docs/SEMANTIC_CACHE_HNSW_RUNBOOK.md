# Cache exato e memória semântica versionada

## Estado implementado

- Embedding de consulta: 640D.
- Prefixo de condicionamento: 1152D, separado do embedding.
- `semantic_memories_embedding_hnsw`: HNSW válido, `m=16`,
  `ef_construction=64`.
- A consulta continua autorizada a usar varredura exata quando o planner
  considera isso melhor para a cardinalidade atual.
- Stream `aeon:semantic:chunks`: auditoria e fallback, não cache de respostas.
- Cache exato: Redis, schema `aeon.semantic_exact_cache.v1`, TTL de 900s.

## Política de hits

`exact_cache_hit` somente é permitido quando a requisição não possui histórico,
`system_message`, embedding histórico/identitário ou grounding adicional. A
chave inclui consulta normalizada, orçamento de tokens, versão do modelo,
dimensão 640, versão da política e época de memória.

Esse hit pode bypassar PostgreSQL e LLM porque representa uma requisição
canônica já validada.

`semantic_context_hit` nunca bypassa o LLM. Ele apenas fornece contexto
limitado recuperado por cosseno para a geração atual.

## Invalidação

Cada nova persistência no stream semântico incrementa
`aeon:semantic:cache:epoch`. Assim, entradas antigas deixam de ser elegíveis
mesmo antes do TTL expirar. O TTL limita a retenção física das chaves antigas.

## Instrumentação

O endpoint `/health` expõe:

- `exact_cache_hits`;
- `exact_cache_misses`;
- `exact_cache_stores`;
- `semantic_cache_invalidations`;
- dimensão, modelo, versão de política e TTL;
- `top_similarity`, `similarity_margin` e `retrieval_latency_ms` por turno.

O cache é um acelerador versionado, não um mecanismo de aprendizado autônomo.
