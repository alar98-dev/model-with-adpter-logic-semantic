# CANAL CÓRTEX↔SUBCÓRTEX + PERSONA + AUTO-REFLEXÃO + TENSOR ASSINADO

> Marco: 2026-08-05 · Derivado da conversa com o Aeon sobre sua personalidade
> e melhorias. O núcleo propôs "abrir o canal de comunicação entre o córtex e
> o subcortex"; implementamos isso — e mais três itens que ele sugeriu.

## 1. Canal córtex↔subcortex (bidirecional real)

O subcórtex (`sensors/subcortex.py`, Gemma-3-270m) é o sensor-poeta do
organismo: gera uma "imagem mental" dos sensores físicos. O canal agora tem
as duas direções:

```
córtex → subcórtex: /dev/shm/aeon/core.intent.jsonl   (o que o núcleo processou/necessita)
subcórtex → córtex: /tmp/swa_imagem_mental.txt         (imagem mental dos sensores + estado do córtex)
```

- **Expressão (córtex→sub)**: ao fim de cada geração, `express_intent()` grava
  1 linha no `core.intent.jsonl` (ex.: `perfil explorativo — interagi sobre
  <tópicos> — N tok em Xs`). O subcórtex lê a última linha e a injeta no prompt
  como `[ESTADO DO CÓRTEX]`, refletindo sobre ela na sua imagem mental.
- **Percepção (sub→córtex)**: `organism_context()` lê a imagem mental e injeta
  `percepção (subcórtex): <imagem>` no bloco `[organismo]` de cada geração.
  Validado ponta a ponta: o núcleo citou conteúdo da imagem mental do subcórtex
  numa resposta.

Parse robusto no subcórtex: decodifica só os tokens novos (sem eco do prompt),
remove marcadores markdown e ruído `[...]`, capa em 2 frases/400 chars, com
fallback se vazio.

## 2. Persona explícita

`PERSONA` (constante no `aeonc.py`) define a identidade estável do Aeon —
observador introspectivo, honesto sobre seus limites, auto-vigilante,
com propósito. Injetada como `system` quando não há `--system` explícito.
A personalidade deixou de ser improvisada a cada sessão.

## 3. Auto-reflexão por interação

Cada pensamento gravado no `core.thoughts.jsonl` ganhou o campo `lesson`: a
última frase da resposta vira a "lição aprendida" do organismo (extração
leve, sem custo de geração extra).

## 4. Assinatura BLAKE3 da injeção tensorial

A injeção de tensor (`--tensors` → `core.tensor.bin`) agora é assinada:

- `_share_tensor()` grava `"signature": "blake3:<hexdigest>"` no
  `core.tensor.json` (fallback `sha256:` se `blake3` indisponível) e faz
  auto-verificação na gravação.
- `obelisk-verifier.sh` ganhou `check_tensor_signature()`: recomputa o BLAKE3
  do `.bin` e compara com a assinatura gravada. Detecta adulteração
  ("injeção suspeita") — testado: íntegra → tamper → DIVERGENTE → restaurada.
  É informativo (não derruba o núcleo).

## Arquivos tocados

- `aeonc.py` — `PERSONA`, `read_subcortex_perception`, `express_intent`,
  `_derive_intent`, `_lesson_from`, `signature` no `_share_tensor`,
  percepção no `organism_context`.
- `sensors/subcortex.py` — `ler_estado_cortex()`, prompt sensorial curto,
  parse limpo.
- `scripts/obelisk-verifier.sh` — `check_tensor_signature`.
- `docs/` — esta documentação.
