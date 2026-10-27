# AeonCell — contrato de autonomia v1

## Objetivo

AeonCell é um receptáculo autônomo de percepção e expressão. O ciclo pode
observar, orientar, decidir e expressar um estado seguro, mas não ganha
autoridade implícita sobre o sistema operacional, arquivos, rede externa ou
ferramentas.

## Ciclo OODA

```text
OBSERVE  UDP6 [::1]:20011 ou ausência de estímulo
ORIENT   Rust valida BLAKE3, limites, fragmentos e sequência
DECIDE   matriz SWA calcula novidade, energia e coerência
ACT      publica somente mascot_idle, mascot_reflect ou mascot_alert
```

Quando não há estímulo tensorial por três segundos, o controlador autônomo
Rust entra em autorregulação. A cada seis segundos ele pode publicar um
`autonomy_tick`, alternando entre repouso e reflexão. Esse ciclo não chama
shell, não executa ferramentas, não acessa rede externa e não escreve no
sistema de arquivos; apenas atualiza o estado visual em `/dev/shm/aeon` e o
evento Redis local.

## Presença e relação

A célula Windows publica um heartbeat local em `swa:aeoncell:presence` usando
`aeon-cell-presence/v1`. O controlador Rust aceita somente a origem
`aeon_cell_windows` e considera a célula ativa por até 5 segundos após o último
heartbeat. O estado OODA passa a registrar `presence.cell_active` e uma política
de relação explícita:

- `owner: aeon` e `controller: rust_policy_gate`;
- modo `attentive_companion` quando a célula está ativa, ou `resting_guardian`;
- agência limitada a `mascot_idle`, `mascot_reflect` e `visual_hint`;
- sem ações externas, execução de processos, escrita de arquivos ou requisições
  de rede.

Isso representa sensação operacional observável (presença, frescor e contexto),
não consciência fenomenológica. A perda do heartbeat fecha a relação para o
modo de repouso seguro.

## Contrato de segurança

Todo evento aceito pelo AeonCell deve conter:

```json
{
  "security": {
    "validated_by": "swa-matriz-tensor-udp-rs",
    "transport": "udp6 ou rust-autonomy",
    "external_network": false,
    "replay_protected": true
  }
}
```

O AeonCell descarta eventos Redis sem essa proveniência. Redis é transporte de
estado, não canal de comandos.

## Liberdades permitidas

- escolher intensidade, cor, animação e pensamento curto;
- manter memória efêmera de novidade e coerência;
- alternar entre repouso e reflexão;
- emitir notificações visuais locais;
- formular intenções futuras sem executá-las.

## Liberdades que exigem aprovação

- executar processo ou comando;
- escrever, apagar ou mover arquivo;
- acessar credenciais;
- fazer requisição externa;
- enviar mensagem ou controlar outro serviço.

Qualquer ação futura deve possuir capacidade explícita, TTL, auditoria e um
gate humano ou política Rust dedicada. A autonomia deve crescer por
capacidades adicionadas ao contrato, nunca por interpretação livre de payloads
Redis.

## Identidade tensorial

A identidade canônica não é enviada como `system_message`. Ela é condicionada
internamente em 640D, projetada para 1152D e aplicada pelo gate
`internal_high_attention_identity` com ganho limitado. O resultado é auditado
no estado da conversa e não concede novas capacidades ao modelo.
