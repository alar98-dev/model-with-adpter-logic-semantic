# Evolucao Arquitetural do SWA-Aeon

## Objetivo

Registrar a transicao do projeto de uma base com acoplamentos difusos para um fluxo mais previsivel, com separacao operacional entre Cortex, Subcortex, Manifold e o runtime principal do SWA.

## Linha do Tempo Resumida

### Fase 1 - Runtime monolitico e caminhos implicitos
- O Cortex carregava defaults dispersos.
- Havia fallback antigo para `models/gemma-3-4-b` em pontos sensiveis.
- Os budgets de geracao eram fixos em varios trechos.
- Os services antigos apontavam para paths obsoletos como `/mnt/PROJECTS/...`.

### Fase 2 - Separacao operacional de modelos
- `swa-auto.service` passou a ser o orquestrador do Cortex principal.
- `swa-subcortex.service` ficou responsavel por sinais rapidos e sensoriais com Gemma 270M.
- `swa-manifold.service` passou a reconstruir o manifold/embeddings reais em boot controlado.
- O `cortex/run.py` passou a centralizar `MODEL_DIR`, `VECTORIZER_MODEL_DIR` e `VOICE_MODEL_DIR` (voice desabilitado em 2026-07-20).

### Fase 3 - Fluxo de geracao orientado ao estado
- O `max_new_tokens` deixou de ser constante em rotas criticas.
- O budget passou a responder a entropia, carga e um vetor afetivo/sentimental do runtime.
- `swa/flux_coupling.py` passou a respeitar `dynamic_max_tokens` ao inves de ignorar o parametro.

### Fase 4 - Higienizacao estrutural
- Imports repetidos e ruido do topo de `cortex/run.py` foram reduzidos.
- O service REST legado foi alinhado para `/services/model-with-adpter-logic-semantic`.
- A documentacao de boot passou a refletir a ordem real de inicializacao.

## Estado Atual

O projeto agora opera com tres contratos principais:

1. `swa-auto` controla a experiencia do Cortex principal.
2. `swa-subcortex` fornece sinais sensoriais separados e mais baratos.
3. `swa-manifold` garante que embeddings e estruturas de memoria nao dependam de improviso no boot.

## Principais Pontos de Cuidado

- Evitar reintroduzir defaults escondidos fora de `DEFAULT_MODEL_DIR`.
- Evitar `max_new_tokens` hardcoded em caminhos novos de geracao.
- Evitar services legados apontando para paths Windows antigos.
- Manter a documentacao sincronizada quando a ordem de boot mudar.

## Proximos Passos Recomendados

- Consolidar os scripts de verificacao de boot e runtime em uma pequena bateria de smoke tests.
- Reduzir comentarios narrativos que contradizem o comportamento real.
- Tornar a separacao de sentimento/entropia um contrato publico do runtime.
