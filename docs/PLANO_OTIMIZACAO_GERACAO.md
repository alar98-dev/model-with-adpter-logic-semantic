# Plano de Otimização — Geração de Conteúdo + Inferência Relacional

> Objetivo: **gerar mais tokens** no aeonc (gemma-3-1b, CPU-only) **mantendo a inteligência interna**
> (prefixo/bias SWA, BlakeGravity, camadas de contexto) e ativando **inferência relacional real entre
> os modelos** (draft 270m, sistema imune obelisk, subcórtex em loop fechado, tensor layer26).

---

## 1. Diagnóstico medido (2026-08-05)

### Hardware
- Intel Core 5 120U, **5 núcleos físicos / 10 threads**, 11 GB RAM, AVX2, sem CUDA.
- `torch 2.12.0+cu130` CPU-only, `torch.get_num_threads()=5` (núcleos físicos), `inter_op=5`.
- PyPI acessível. `intel_extension_for_pytorch` instalado porém **incompatível** (exige torch 2.8) → fora.

### Baseline de geração (aeonc)
- `dtype=bfloat16`, `--compile` ligado, thread `TextIteratorStreamer` + `model.generate`.
- Medido sob carga: **0.71 tps** (48 tok em 67s, perfil `preciso`); ~1.3–1.4 tps sem carga.
- `do_sample=True`, `top_p=0.95`, `temperature` por perfil (aeonc.py:83-121, 687-701).

### Contenção de CPU (causa #1 da lentidão)
| processo | CPU | RAM | função | impacto na geração |
|---|---|---|---|---|
| embedder (270m fp32, :21000) | **~200%** | ~2.6GB | embeddings kbase/build | nenhum |
| obelisk (270m bf16) | ~50% | ~1.1GB | sistema imune/classificador | nenhum (não steera tokens) |
| subcórtex (270m bf16) | ~36% | ~0.6GB | consome intent → imagem mental | nenhum |
| aeonc (1b bf16) | ~11% | ~1.3GB | geração | — |
| load average | **15–18** | — | — | satura 10 threads |

### Achados arquiteturais (verificados)
1. **Dois caminhos de decode divergentes**:
   - `aeonc.py` usa HF `model.generate` (loop C++ otimizado, KV cache).
   - `swa/logical_engine.py` + `swa/gemma_integration.py` (GemmaGenerator) usam **loop Python manual**
     por token (`softmax`/`multinomial`), onde vivem o prefixo SWA e o BlakeGravity.
   - **A "inteligência" só roda no caminho lento; o aeonc em produção não a usa na geração.**
2. **LoRA/adapters**: dirs `models/obelisk-lora-adapter/*`, `autolora-adapters/*`, `swa-lora-adapter` estão
   **vazios**; nenhum `load_adapter` no código de inferência. A inteligência real é in-memory
   (SemanticWarpAdapter prefixo/bias + BlakeGravity layer-14).
3. **Inferência relacional atual é unidirecional e desativada**:
   - tensor layer26 (`core.tensor.bin`) **obsoleto** (07:23) — aeonc rodou sem `--tensors`.
   - ContextBus ring buffer existe mas não é consumido pelo aeonc.
   - Obelisk: `DANGER_SIGNAL` aponta para socket sem listener → sistema imune não steera tokens.
   - Subcórtex: escreve `/tmp/swa_imagem_mental.txt` (vivo) mas aeonc não lê.
4. **270m carregado 3× em RAM** (embedder fp32, obelisk bf16, subcórtex bf16) ≈ 3–4 GB e 3 cópias de CPU.

---

## 2. Meta

- 0.7 → **3–6 tps** no aeonc, mantendo prefixo/bias SWA + BlakeGravity ativos **dentro do caminho rápido**.
- Inferência relacional funcional: 270m como draft (speculative), obelisk como immune-steerer,
  subcórtex em loop fechado, tensor layer26 compartilhado.

---

## 3. Fases

### Fase 0 — Cortes imediatos (risco baixo, ~30 min, efeito ~2–3×)
1. **`Nice=19` no embedder + build** → aeonc sempre preempta. Edit unit + daemon-reload.
2. **Obelisk**: intervalo 30s→120s (daemon loop) e `CPUQuota=100→50` → libera ~0.5 core.
3. **Subcórtex**: `CPUQuota=150→100` → libera ~0.5 core.
4. **`torch.set_num_threads(10)` no aeonc** (HT ajuda decode memória-bound) — medir antes/depois.
5. **Protocolo de baseline** (aba) — re-medir com a máquina livre.

### Fase 1 — Throughput do 1B (efeito principal, 1–2 dias)
1. **int8 weight-only (torchao)** no 1B: `pip install torchao`; `quantize_(model, int8_weight_only())`.
   - Esperado **2–3× vs bf16 emulada** em AVX2 (bf16 não tem SIMD nativo nesta CPU; int8 tem).
   - Peso ~1.4 GB; testar com/sem `--compile` (podem conflitar → desligar compile se precisar).
   - Fallback se torchao falhar: bf16 com `torch.set_num_threads(10)` + trim de contexto.
2. **Trim de contexto/prefill**: medir tamanho do prompt (organism_context + kbase + introspect); cortar
   blocos não essenciais para reduzir TTFT e memória de KV.
3. **Reuso de KV entre turns** (só se o trim não bastar): manter `past_key_values` por sessão.

### Fase 2 — Decodificação especulativa (relacional + velocidade, 2–3 dias)
1. Carregar **gemma-3-270m-it como `assistant_model`** no `model.generate` do aeonc.
   - 270m ~4–5× mais barato que 1B; aceitação ~0.6–0.8 → **+1.5–2.5× sobre a Fase 1**.
2. **Consolidar o 270m em UM serviço** (embedder + draft + subcórtex/obelisk consultam o mesmo processo):
   economia de ~2–3 GB RAM e ~1 core. Novo contrato de serviço `:21000` com endpoint `/draft`.
3. Medir `acceptance rate`, `tps` com 2/4/6 tokens de draft.

### Fase 3 — Inferência relacional real (inteligência entre modelos, 3–5 dias)
1. **Migrar a inteligência para o caminho rápido**: injetar `v_prefix` (prefix embeddings SWA) e
   BlakeGravity via `LogitsProcessor`/prefixo no HF generate do aeonc — mesma semântica do
   `logical_engine`, mas em loop C++.
2. **Ativar `--tensors`**: aeonc publica layer26; verificar `core.tensor.bin` fresco.
3. **Obelisk → steering**: reparar o canal (socket real) e aplicar os sinais de risco como
   `LogitsProcessor` (penalizar drift/reescrever) — transforma o sistema imune em controle de geração.
4. **Loop fechado subcórtex→aeonc**: aeonc lê `/tmp/swa_imagem_mental.txt` no organism_context
   (o canal já existe e está vivo).
5. **ContextBus**: consumir o ring buffer no aeonc (flags INJECTED/AEON_MODULATED).

---

## 4. Protocolo de medição (rodar com build pausado / máquina "quieta")

Cenário fixo: 3 prompts (1 código, 1 prosa, 1 relacional), `max_new_tokens=256`, perfil padrão.
Métricas por execução: `tps`, `ttft`, `acceptance rate` (Fase 2), `RAM pico`, `load avg` simultâneo.
Registrar em `docs/KBASE_EVOLUTION.md`-style de benchmark (tabela config → tps → ganho).

Comando de baseline rápido (fora do aeonc, carregando o 1B direto):
```bash
python3 scripts/bench_gen.py --model models/gemma-3-1b-it --dtype bfloat16 --max-new-tokens 256
```
(scripts/bench_gen.py a criar na Fase 1 — mede prefill + decode, com/sem quantização, com/sem compile.)

---

## 5. Riscos e mitigação

| risco | mitigação |
|---|---|
| int8 + `--compile` incompatíveis | testar combinações; fallback: só int8 sem compile |
| quantização degrada resposta (fidelidade SWA) | benchmark de regressão textual (3 prompts fixos) antes de fixar |
| speculative lento em CPU (draft caro) | medir acceptance; se <1.3×, manter só Fase 1 |
| fp32 no 1B inviável (RAM) | descartado — não consta no plano |
| IPEX quebrado no venv | ignorar (ou remover) |
| Fase 3 muda semântica de geração | atrás de flag `--intelligence=fast`; default conservador até validar |

---

## 6. Ordem de execução sugerida

```
Fase 0 (30min, imediato) → medir → Fase 1 int8 (1 dia) → medir
→ Fase 2 speculative (2-3 dias) → medir → Fase 3 relacional (3-5 dias) → medir/validar
```
Cada fase só avança se a métrica melhorar e o teste de regressão passar.
