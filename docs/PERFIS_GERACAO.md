# PERFIS DE GERAÇÃO — Especialização por responsabilidade

> Marco: 2026-08-04 · Nativo no `aeonc.py` (CLI `aeon`/`aeonc` + API `:20000`)

Separar responsabilidades de geração em **perfis especializados** — sem trocar de
modelo. Cada perfil ajusta amostragem (`temperature`, `top_p`), orçamento de
tokens (`max_new_tokens`), `repetition_penalty` e injeta uma **instrução de
especialização** como mensagem `system` (mesclada com o contexto do organismo).

## Perfis

| Perfil | Objetivo | temp | top_p | max_tok | rep | Instrução |
|---|---|---|---|---|---|---|
| `preciso` | resposta exata, curta, factual | 0.1 | 0.6 | 256 | 1.1 | conciso e objetivo |
| `equilibrado` | padrão — fluência + precisão | 0.7 | 0.95 | 512 | 1.0 | — |
| `explorativo` | **mais texto** — rico, detalhado, diverso | 0.95 | 1.0 | 1024 | 1.15 | desenvolver amplamente |
| `duo` | **ambos** — 1º exato, 2º expansão | 0.7 | 0.95 | 768 | 1.0 | estrutura em 2 partes |

## Uso na CLI (nativo)

```bash
aeon                        # CLI interativa (modo atual: equilibrado)
aeonc --profile explorativo # inicia já em explorativo
aeonc --tensors --profile duo
```

Dentro da sessão:

```
> /modo                     # lista perfis + atual
> /modo preciso             # troca na hora
> /modo explorativo         # estica a geração (mais texto)
> /modo duo                 # resposta exata + expansão
> /temp 0.9                 # override manual persiste até trocar de perfil
```

`/temp`, `/top-p`, `/max` continuam sendo overrides manuais; trocar de perfil
redefine para os valores do perfil. Cada turno reporta o perfil usado nas
métricas (`[N tok · Xs · Y tps · dtype · perfil]`).

## Uso na API

`GET /v1/profiles` — lista perfis + o atual.
`POST /chat` e `POST /v1/chat/completions` aceitam campo `profile`:

```bash
curl -s http://127.0.0.1:20000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-3-1b-it","messages":[{"role":"user","content":"..."}],
       "profile":"explorativo","stream":false}'
```

Se o perfil é informado, ele define os padrões de temp/top-p/max; valores
explícitos no request têm prioridade. O `opencode.json` continua usando o
default (equilibrado).

## Persistência (o organismo não esquece)

O perfil ativo é gravado em `/dev/shm/aeon/core.state.json` a cada troca
(`/modo`, API) e **restaurado no boot** — sobrevive a restarts do serviço.
Sem `--profile` explícito, o núcleo volta ao último perfil usado.

Como o `opencode.json` não injeta `profile` no corpo da requisição (padrão
OpenAI), o **default da API** é o perfil persistido — hoje `explorativo`, para
que o opencode ganhe respostas ricas e detalhadas. Para mudar:

```bash
echo '{"profile": "preciso"}' > /dev/shm/aeon/core.state.json
sudo systemctl restart swa-aeonc
```

ou via CLI: `/modo preciso` (persiste na hora).

## Detalhe técnico

- **Instrução e contexto** são mensagens `system`; o template Gemma exige
  alternância de papéis → `_merge_system_messages()` mescla `system`s
  consecutivos em um só. (Bug latente exposto pelo `duo`, corrigido.)
- `do_sample`/`repetition_penalty` vêm do perfil ativo; por chamada pode-se
  passar um `GenerationProfile` sem tocar no perfil global (seguro para
  concorrência na API).

## Estratégia "esticar conteúdo"

- `explorativo` = mais texto (temp alta + orçamento 1024 + rep 1.15).
- `preciso` = resposta enxuta e factual (temp 0.1 + top_p 0.6 + rep 1.1).
- `duo` = os dois em um único turno.
