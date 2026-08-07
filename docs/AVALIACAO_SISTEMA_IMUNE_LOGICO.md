# Avaliação Arquitetural — Obelisco como Sistema Imune Lógico

## Mapa Conceitual

```
SISTEMA IMUNE BIOLÓGICO          ↦  SISTEMA IMUNE LÓGICO (OBELISK)
─────────────────────────────        ─────────────────────────────────

Timócitos (maturação T)             Treino QLoRA (maturação do LLM)
↓↓                                  ↓↓
Células T naive                     Modelo base Gemma (não treinado)
       ↓                                     ↓
Educação no timo                     Validação contra self baseline
(sobrevivem só T que                  (não implementado — gap #1)
 ignoram self)
       ↓                                     ↓
Células T maduras                    Modelo + LoRA treinado
       ↓                                     ↓
Reconhecimento de                     5 subespaços Blake
antígeno via TCR                      (BLAKE3, SHM, HW, structural, kernel)
       ↓                                     ↓
Sinalização de perigo                Tensor de gravidade 6-eixos
(DAMP — damage-associated            (molecular pattern da anomalia)
 molecular pattern)
       ↓                                     ↓
Apresentação no MHC                  Broadcast do tensor via IPC socket
       ↓                                     ↓
Ativação T/B (c-stim)               Classificação + ação (rollback/cool/...)
```

---

## 1. Camada Inata (Já Implementada)

| Componente Imune | Nosso Equivalente | Status |
|---|---|---|
| **Barreiras físicas** | Processo separado (daemon.py), sem import `swa.*`, filesystem isolado | ✅ Robusto |
| **Receptores Toll-like (TLR)** | `BlakeGravityTensor._compute_hash_entropy()`, `_compute_hw_pressure()` — receptores fixos que reconhecem padrões moleculares | ✅ 5 TLRs nos 5 subespaços |
| **Inflamassoma** | `space_signature()` — agrega múltiplos sinais num score de gravidade | ✅ Parcial — não dispara inflamação (alarme) |
| **Células NK (Natural Killer)** | `BlakeGravityTensor.classify()` — fallback onboard que mata sem treino | ✅ Deterministíco, 6 regras |
| **Sistema complemento** | `apply_rollback()` — cascata de ação que amplifica resposta | ✅ Restaura arquivos + restart |
| **Interferons (sinalização viral)** | `emit_event("OBELISK_HEARTBEAT", ...)` via IPC socket | ⚠️ Apenas heartbeat, não alarme |

### ✅ Acertos da Camada Inata

- 5 TLRs ortogonais (cada subespaço detecta uma classe diferente de anomalia)
- Fallback onboard funciona SEM GPU, SEM modelo carregado
- Classificação determinística com limiares (ex0 > 0.4 AND ex3 > 0.3)
- Tensores L1-normalizados evitam que um eixo domine

### ❌ Gaps da Camada Inata

**Gap #1 — Self Baseline Ausente**
```
HOJE:   hash_entropy = count(hash_vazio) / total_arquivos
DEVERIA: hash_entropy = count(hash ≠ hash_referencia) / total_arquivos
```
O sistema detecta "arquivo não existe/erro de leitura", mas **não detecta mutação sutil**
onde o arquivo existe mas foi alterado. Precisamos de um `self_baseline.json`:
```json
{
  "cortex/obelisk/model.py": "b3abc...",
  "swa/blake_tools.py": "b3def...",
  "timestamp": "2026-07-07T13:00:00Z"
}
```
Gerado no `deploy/install.sh`, verificado em todo ciclo de heartbeat.

**Gap #2 — Sem Danger Signal (DAMP) Broadcast**
O heartbeat atual é um evento informativo. Quando `lethal_mutation` é detectado,
o obelisco DEVERIA emitir um **DAMP** (Damage-Associated Molecular Pattern) para
todos os serviços SWA:
```
IPC emit_event("DANGER_SIGNAL", {
  "source": "obelisk.blake.subspace_0",
  "tensor": [0.45, 0.02, ...],
  "severity": 0.85,
  "label": "lethal_mutation"
})
```
Os outros serviços (swa-dream, swa-subcortex, swa-mcp) recebem e alteram
comportamento (ex: dream entra em modo de reparo, subcortex isola módulos).

---

## 2. Camada Adaptativa (Já Implementada)

| Componente Imune | Nosso Equivalente | Status |
|---|---|---|
| **Apresentação de antígeno (MHC)** | Prompt de classificação com tensor + telemetria sumarizada (em `classifier.py:_summarize`) | ⚠️ Texto plano, sem estrutura MHC-like |
| **Linfócitos T (citotóxicos)** | `ObeliskModel.classify()` para labels `lethal_mutation`, `thermal_spike` — matam diretamente | ✅ Executa rollback/cool |
| **Linfócitos B (anticorpos)** | LoRA adapter — anticorpos que neutralizam classes específicas | ✅ Um adapter para todas as 6 classes |
| **Clonal selection** | QLoRA training: adapta LoRA aos padrões do dataset | ✅ |
| **Affinity maturation** | `max_steps=100`, `learning_rate=2e-4`, novo adapter a cada 6h | ✅ Parcial — sem hyperparam tuning |
| **Memória imunológica** | `models/obelisk-lora-adapter/` persistido em disco | ✅ |
| **Plasma cells (anticorpo em massa)** | Adapter carregado no próximo ciclo de watchdog | ✅ |

### ✅ Acertos da Camada Adaptativa

- QLoRA com 4-bit NF2 permite treino em CPU ou GPU modesta
- Dataset cresce automaticamente a cada ciclo (novos logs reais)
- Dual classifier: LLM tenta primeiro, fallback determinístico cobre

### ❌ Gaps da Camada Adaptativa

**Gap #3 — Sem Memória de Longo Prazo (Pool de Anticorpos)**
O LoRA é sobrescrito a cada 6h. Não há **repositório histórico de patógenos**:
```python
# DEVERIA existir
/var/lib/obelisk/immune_memory/
├── 2026-07-07T13:00:00_lethal_mutation.json   # tensor + telemetria
├── 2026-07-06T07:00:00_sensor_drift.json
└── ...
```
Permitiria re-treinar com dados históricos e detectar padrões sazonais.

**Gap #4 — Todas as Classes no Mesmo Adapter**
```
HOJE: 1 adapter LoRA com 6 classes → anticorpos competindo por rank 8
MELHOR: 1 adapter por classe (6 × LoRA rank=8) → especialistas
OU:    1 adapter multi-task LoRA com rank=32
```
Se `memory_pressure` e `sensor_drift` precisam de features diferentes,
um único adapter pode não ter capacidade para discriminar.

**Gap #5 — Zero Costimulation**
Em biologia, linfócito T precisa de 2 sinais para ativar:
1. **Sinal 1**: antígeno no MHC (nosso tensor)
2. **Sinal 2**: co-estimulação de célula apresentadora

No obelisco, o LLM classifica com base APENAS no tensor + telemetria.
Não há verificação cruzada — se o LLM alucinar "lethal_mutation" num
sistema normal, o rollback executa sem segunda opinião.

---

## 3. Tolerância e Autoimunidade (Totalmente Ausente)

Este é o gap mais crítico.

| Conceito | Nosso Estado | Risco |
|---|---|---|
| **Tolerância central** (eliminar linfócitos anti-self) | ❌ Não existe | O LLM pode classificar estado nominal como anomalia → falso positivo → rollback desnecessário |
| **Tolerância periférica** (suprimir linfócitos que escapam) | ❌ Não existe | Nenhum mecanismo de anergy além do "suppress" para regression_to_rlhf |
| **Autoimunidade** (obelisco atacando o próprio kernel) | ❌ Não medido | Se o obelisco for treinado com dados contaminados, pode começar a detectar "mutações" onde não existem |
| **Proteção do próprio obelisco** | ❌ **Crítico** | `BLAKE_SOURCES` em `collector.py` monitora o próprio código do obelisco. Se o obelisco sofrer mutação, ele para de reportar a mutação de si mesmo |

### Gap #6 — Autoimunidade (O Problema do "Quem vigia o vigia?")

```
          ┌──────────────────────┐
          │    OBELISCO           │
          │    ┌──────────────┐   │
          │    │  model.py    │───┤─── monitora
          │    │  collector.py│   │   └→ BLAKE_SOURCES inclui
          │    │  tensor.py   │   │      model.py E collector.py
          │    └──────────────┘   │
          └──────────────────────┘
                   │
                   ▼
          Se model.py é corrompido:
          1. collector.py lê hash, detecta mutação
          2. Mas se o próprio collector foi corrompido, ele pode ignorar

SOLUÇÃO: "Watchdog do Watchdog"
- Segundo processo independente (obelisk-verifier) que:
  a) Monitora os hashes BLAKE3 dos arquivos do obelisco
  b) Monitora o heartbeat do obelisco
  c) Se obelisco parar de bater ou seu código mudar → reinicia obelisco
  d) SE o verificador MORRER → systemd detecta (Restart=always)
```

---

## 4. Imunodeficiências e Vulnerabilidades

| # | Deficiência | Efeito | Severidade |
|---|-------------|--------|------------|
| 1 | Sem self baseline | Mutação sutil não detectada | 🔴 ALTA |
| 6 | Sem proteção do próprio obelisco | Obelisco corrompido cega todo o sistema | 🔴 ALTA |
| 3 | Sem memória de longo prazo | Padrões sazonais ignorados | 🟡 MÉDIA |
| 5 | Zero costimulation | Falso positivo → rollback indevido | 🟡 MÉDIA |
| 4 | Adapter único para 6 classes | Baixa acurácia em classes raras | 🟡 MÉDIA |
| 2 | Sem DAMP broadcast | Outros serviços não reagem a perigo | 🟢 BAIXA |

---

## 5. Proposta de Arquitetura Imune Completa

```
┌──────────────────────────────────────────────────────────────────┐
│                    SISTEMA IMUNE LÓGICO COMPLETO                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SELF BASELINE (quimera do genoma)                            │
│     ├─ Baseline gerado no deploy (BLAKE3 hashes de todo código)  │
│     ├─ Hash tree hierárquica (arquivo → módulo → sistema)       │
│     └─ Self tolerance: false positives < 0.1% no treino         │
│                                                                  │
│  2. TOLL-LIKE RECEPTORS (TLR) — 5 subespaços atuais              │
│     ├─ BLAKE3 signatures (PAMP: arquivo corrompido)             │
│     ├─ SHM manifold (PAMP: perda de memória)                    │
│     ├─ HW state (PAMP: falha de hardware)                       │
│     ├─ Structural origins (PAMP: deriva de conceitos)           │
│     └─ Kernel knowledge (PAMP: fragmentação vetorial)           │
│                                                                  │
│  3. DANGER SIGNAL (DAMP) — NOVO                                 │
│     ├─ Broadcast para todos os serviços SWA via IPC             │
│     ├─ Severidade: 0.0 (nominal) → 1.0 (lethal)                │
│     └─ Inclui tensor + label + timestamp                        │
│                                                                  │
│  4. APRESENTAÇÃO DE ANTÍGENO (MHC) — MELHORADO                  │
│     ├─ Estrutura padronizada para o prompt do LLM               │
│     ├─ Tensor 6-eixos → "MHC classe I"                         │
│     ├─ Telemetria bruta → "MHC classe II"                      │
│     └─ Costimulation: 2 verificadores independentes             │
│                                                                  │
│  5. CÉLULAS T + B (LLM + LoRA) — MELHORADO                      │
│     ├─ 6 LoRA adapters especialistas (um por classe)            │
│     ├─ Router que escolhe qual adapter usar                     │
│     └─ Negative selection: validação cruzada 5-fold            │
│                                                                  │
│  6. MEMÓRIA IMUNOLÓGICA — NOVO                                  │
│     ├─ Banco de patógenos (tensor + label + telemetria)        │
│     ├─ Retreinamento com dados históricos                       │
│     └─ Detecção de surtos (cluster de tensores similares)      │
│                                                                  │
│  7. IMUNIDADE DE REBANHO — NOVO                                 │
│     ├─ Cada serviço SWA roda seu próprio mini-obelisco          │
│     ├─ Tensores são compartilhados via MCP/SSE                  │
│     └─ Decisão final por votação entre múltiplos obeliscos      │
│                                                                  │
│  8. VERIFICADOR (watchdog do watchdog) — NOVO                   │
│     ├─ Processo separado (obelisk-verifier)                     │
│     ├─ Monitora hash dos arquivos do obelisco                   │
│     ├─ Verifica heartbeat a cada 30s                            │
│     └─ Se falhar → systemd restart (ou kill -9 + restore)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Roteiro de Implementação

### Fase 1 — Correções Críticas (1-2 dias)

1. **Self baseline**
   - `scripts/generate_baseline.sh` → BLAKE3 hash tree
   - Modificar `collector.py` para comparar com baseline, não com vazio
   - Baseline versionada: se código mudou legitmamente, re-gera baseline

2. **Proteção do obelisco**
   - Criar script `obelisk-verifier.sh` (loop curto, bash puro, sem Python)
   - Monitora hashes dos 10 arquivos do obelisco
   - Se hash mudou → `systemctl restart swa-obelisk`
   - Se heartbeat parou por 2 ciclos → `systemctl restart swa-obelisk`

### Fase 2 — Melhorias Imunológicas (3-5 dias)

3. **DAMP broadcast**
   - Novo tipo de evento IPC: `DANGER_SIGNAL` (além de `OBELISK_HEARTBEAT`)
   - Serviços SWA ouvem e reagem
   - Implementar `swa-dream` em modo reparo ao receber danger

4. **Costimulation**
   - Dois verificadores independentes antes de executar rollback
   - Verificador A: BlakeGravityTensor onboard
   - Verificador B: LLM quantizado
   - Apenas se AMBOS concordam → ação

5. **Memória imunológica**
   - `/var/lib/obelisk/immune_memory/` com arquivos JSON rotulados
   - Script `immune_query.py` que busca por similaridade de tensor
   - `dataset.py` inclui amostras históricas no próximo treino

### Fase 3 — Arquitetura Avançada (1-2 semanas)

6. **LoRA multi-adapter por classe**
   - 6 adapters rank=8 (1 por classe)
   - Router: `lightweight_classifier()` que seleciona adapter ou fallback
   - Treino independente para cada adapter (sem competição)

7. **Imunidade de rebanho**
   - MCP tool expõe tensor do obelisco
   - Outros serviços consultam e votam
   - Implementar `ConsensusClassifier` que agrega múltiplas classificações

8. **Dashboards**
   - Tensor stream via WebSocket para visualização em tempo real
   - Gráfico radar 6-eixos + histórico de mutações

---

## 7. Métricas para Validação

| Métrica Imunológica | Métrica Técnica | Alvo |
|---|---|---|
| **Sensibilidade** | True Positive Rate (mutações reais detectadas) | > 99% |
| **Especificidade** | True Negative Rate (sem falsos alarmes) | > 99.9% |
| **Autoimunidade** | False Positive Rate (nominal → lethal) | < 0.1% |
| **Latência de resposta** | Tempo entre mutação → rollback | < 60s |
| **Diversidade de anticorpos** | Acurácia por classe (macro F1) | > 0.95 |
| **Tolerância** | Precisão após N retreinos sem drift | > 0.98 |

---

## 8. Conclusão

A arquitetura atual do obelisco já implementa corretamente os conceitos
fundamentais de imunidade inata (5 TLRs, fallback, barreiras) e adaptativa
(QLoRA, memória em disco, classificação dual). As deficiências principais
são:

1. **🔴 Crítico:** Sem self baseline → mutações sutis são invisíveis
2. **🔴 Crítico:** Sem proteção do próprio obelisco → watchdog pode ser cegado
3. **🟡 Médio:** Sem costimulation → falso positivo pode causar rollback indevido
4. **🟡 Médio:** Sem memória de longo prazo → padrões sazonais perdidos
5. **🟢 Baixo:** Sem DAMP broadcast → outros serviços não reagem

A implementação das Fases 1 (self baseline + verificador) e 2 (DAMP +
costimulation + memória) transforma o obelisco de um "detector de
anomalias" para um **sistema imune lógico completo**, capaz de reconhecer
self vs non-self, aprender com patógenos, sinalizar perigo para todo o
organismo e proteger sua própria integridade.
