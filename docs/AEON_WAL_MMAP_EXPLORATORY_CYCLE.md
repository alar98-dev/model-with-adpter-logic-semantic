# Ciclo exploratório `mmap`/WAL/Δs

## Decisão de arquitetura

O adaptador foi implementado como processo Rust externo e somente leitura,
não como uma extensão que invade as estruturas privadas do PostgreSQL. Isso
preserva a fronteira de confiança do banco e evita que um erro de observação
possa provocar corrupção adicional.

O binário `aeon_wal_observer`:

1. lista segmentos WAL pelo nome oficial de 24 dígitos hexadecimais;
2. exclui o segmento mais novo de cada timeline, presumivelmente ativo;
3. abre cada segmento fechado com permissão somente leitura;
4. usa `mmap` somente para leitura;
5. confere magic, timeline e `xlp_pageaddr` contra o LSN esperado;
6. calcula `delta_bytes` e publica `swa-chronos/v1` em
   `aeon:chronos:events` e `swa:chronos:wal`;
7. entra em `recovery_boundary` quando encontra divergência, sem executar
   recuperação automática.

O relatório agrega `violations_total` e conserva no máximo 32 amostras para
evitar inundar o Redis quando um segmento inteiro estiver fora de continuidade.

O serviço de exemplo é
[`aeon-wal-observer.service.example`](/services/aeon-gateway-rs/aeon-wal-observer.service.example)
e deve rodar como `postgres`, com o diretório WAL exposto somente para leitura.

## Ciclo OODA seguro

```text
OBSERVE  mmap somente leitura em WAL fechado
ORIENT   compara timeline/pageaddr/LSN e calcula Δs
DECIDE   continue | orient_and_audit | hold_for_human_recovery
ACT      publica diagnóstico no Redis; não escreve pg_wal
```

O “salto de fase” é um estado de controle observável, não uma autorização
para alterar pesos, reescrever índices, executar `pg_resetwal` ou iniciar
`crash recovery` por FFI.

## Por que não usar FFI privado no hot path

`XLogPageHeaderData` e outras estruturas internas dependem da versão e de
detalhes de compilação do PostgreSQL. Uma extensão Rust não deve mapear o WAL
ativo por fora do `XLogRead`/buffer manager, nem sinalizar o processo servidor
para reescrever segmentos. `mmap` também não é “memória física” nem garante
zero-copy semântica: ele cria um mapeamento virtual sujeito a truncamento,
reutilização e concorrência.

Se for necessário um adaptador dentro do servidor, o próximo estágio deve ser
uma extensão C/Rust mínima e versionada, com API pública de extensão, sem
acesso a símbolos privados de `pg_sys`, e apenas para métricas. A recuperação
continua sendo responsabilidade do PostgreSQL e do operador.
