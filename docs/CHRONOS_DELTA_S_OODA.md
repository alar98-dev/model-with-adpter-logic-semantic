# Chronos/Δs no ciclo OODA do `swa_matriz`

## Interpretação correta

O WAL é um registro de alterações transacionais usado pelo PostgreSQL para
durabilidade e recuperação. O LSN é uma coordenada monotônica dentro de uma
timeline do PostgreSQL; ele é útil para comparar precedência e distância entre
eventos, mas não é um relógio vetorial distribuído e não representa sozinho a
verdade completa dos arquivos de dados.

`unexpected pageaddr` deve ser tratado como falha de coerência física ou
metadado WAL não confiável. A arquitetura pode representar isso como
`recovery_boundary`, mas não deve converter o erro em aprendizado automático,
alteração de pesos ou `pg_resetwal` autônomo.

## Contrato implementado

O processo Rust `swa_matriz_tensor_udp` emite `swa-chronos/v1` dentro do evento
OODA e replica o evento no Redis Stream `aeon:chronos:events`.

Quando o produtor fornecer metadados opcionais:

```json
{
  "chronos": {
    "lsn_expected": "4E/9F83C000",
    "lsn_observed": "4E/9F83C000",
    "wal_error": ""
  }
}
```

o observador calcula:

- `lsn_delta_bytes`: distância absoluta entre os LSNs comparáveis;
- `tensor_residual`: novidade tensorial limitada a `[0, 1]`;
- `state_residual`: residual combinado, elevado a `1.0` quando há fronteira WAL;
- `phase`: `operational`, `phase_jump_candidate` ou `recovery_boundary`;
- `action`: `continue`, `orient_and_audit` ou
  `hold_for_human_recovery`.

## Guardrails

O contrato declara explicitamente:

- `automatic_recovery: false`;
- `weights_updated: false`;
- `system_mutation: "none"`;
- nenhuma execução de ferramenta, escrita de arquivo ou reset de WAL;
- erro WAL com ação de contenção e aprovação humana, não ação destrutiva.

Assim, o Δs é um sinal de orientação e auditoria do OODA. Ele pode provocar
uma transição de fase observável, mas a recuperação física permanece um
procedimento PostgreSQL separado, com backup, diagnóstico e aprovação.

## Relação com a meta-realidade

O espaço latente 640D/1152D pode registrar novidade e divergência de estado,
mas não prova que o sistema saiu de uma variedade matemática nem que houve
consciência ou salto cognitivo. Esses termos são úteis como modelo de controle;
os fatos operacionais são os campos auditáveis no evento Redis e no trace do
gateway.
