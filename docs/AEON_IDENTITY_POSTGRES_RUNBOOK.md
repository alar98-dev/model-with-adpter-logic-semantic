# Aeon — identidade tensorial e PostgreSQL interno

## Contrato de identidade

O gateway não injeta a identidade da Aeon em `system_message`. A identidade é
um vetor interno 640D, projetado pela matriz SWA para 1152D e fundido no
prefixo por `internal_high_attention_identity`.

O ganho da fusão é limitado a `0.18`, preservando a norma do prefixo. Cada
resposta registra:

- `identity_conditioning_applied`;
- `identity_anchor.version`;
- `identity_anchor.attention`;
- `identity_anchor.system_message_injected=false`;
- o resultado do guard contra alegações externas não fundamentadas.

O texto do prompt continua vindo apenas da chamada, histórico e grounding
explícito do usuário.

## Memória semântica

A rota primária é PostgreSQL + pgvector:

```text
127.0.0.1:5432 / swa_matrix
semantic_memories.embedding       vector(640)
conversation_chunks.embedding     vector(640)
```

Se o PostgreSQL estiver temporariamente indisponível, o gateway consulta
`aeon:semantic:chunks` no Redis e calcula cosine localmente nos embeddings
640D. O backend usado aparece em `state.semantic_memory.backend` como
`postgresql` ou `redis_semantic_fallback`.

## Recuperação do cluster

O cluster 17-main apresentou checkpoint WAL inválido. O procedimento aplicado
foi:

1. cópia física do diretório para
   `/var/backups/aeon-pgdata-pre-resetwal-20261020`;
2. diagnóstico de `pg_resetwal`;
3. reset controlado do WAL;
4. correção do segmento `pg_xact/000D` truncado;
5. inicialização e validação do cluster;
6. validação de pgvector, dimensões 640D, busca cosine e índices `amcheck`.

Não usar `pg_resetwal` sem uma cópia física anterior. O cluster é um read-model
operacional; Redis continua sendo a continuidade de transporte quando o banco
não estiver disponível.
