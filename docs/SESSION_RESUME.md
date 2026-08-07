# SESSION RESUME — Obelisk Blake Gravity Watchdog

## Contexto do Projeto

Sistema SWA AEON: kernel de IA híbrido com múltiplos modelos, bancos
vetoriais topológicos, IPC por socket Unix, estado persistente em
hardware Raw (NVMe). O Obelisk é um subsistema watchdog cognitivo
desacoplado que detecta mutações internas em tempo real.

## Arquitetura em 1 minuto

```
5 subespaços Blake → Tensor Gravidade 6-eixos → LLM 4-bit + LoRA → Ação
                                                    └─ fallback onboard
```

- **5 subespaços:** BLAKE3 signatures, SHM manifold, HW state, structural origins, kernel knowledge
- **Tensor 6-eixos:** hash_entropy, manifold_mass, hw_pressure, structural_resonance, kernel_coherence, gravitational_signature
- **Classificador dual:** fallback onboard determinístico (regras) + LLM quantizado 4-bit (Gemma-3-270M NF4 + LoRA rank=8)
- **6 classes:** nominal, lethal_mutation, thermal_spike, regression_to_rlhf, memory_pressure, sensor_drift
- **Ações:** rollback, cool (taskset E-cores), evacuação de cache, recalibração, supressão de alinhamento

## Estrutura de Arquivos

```
/services/model-with-adpter-logic-semantic/
├── cortex/obelisk/                    ← núcleo (10 arquivos)
│   ├── __init__.py
│   ├── model.py                       → ObeliskModel.load() 4-bit + classify()
│   ├── train.py                       → QLoRA rank=8, q_proj+v_proj
│   ├── dataset.py                     → 203+ amostras (19 blake + 12 sistêmicas + logs)
│   ├── main.py                        → watchdog loop (telemetria → classificar → agir)
│   ├── daemon.py                      → supervisor PID + heartbeat
│   └── blake/
│       ├── __init__.py
│       ├── collector.py               → collect_all() → 5 subespaços
│       ├── tensor.py                  → BlakeGravityTensor.compute() → 6-eixos
│       └── classifier.py             → BlakeSpaceClassifier (LLM + fallback)
├── deploy/
│   ├── swa-obelisk.service            → systemd watchdog (Restart=always)
│   ├── swa-obelisk-train.service      → oneshot de treino QLoRA
│   ├── swa-obelisk-train.timer        → 6h timer (00,06,12,18h)
│   ├── install.sh                     → deploy automático
│   └── .env.obelisk                   → variáveis de ambiente
├── models/
│   ├── gemma-3-270m/                  → modelo base (4-bit NF4 ao carregar)
│   ├── obelisk-lora-adapter/          → adaptador LoRA treinado
│   └── obelisk_dataset.jsonl          → 203 amostras (6 classes)
├── cortex/organs/obelisk_devourer.py  ← redireciona com DeprecationWarning
├── cortex/obelisk_daemon.py           ← redireciona com DeprecationWarning
└── docs/
    ├── POC_OBELISK_BLAKE_GRAVITY.md   → POC para artigo
    └── SESSION_RESUME.md              ← este arquivo
```

## Estado Atual dos Serviços

| Serviço | Enabled | Running | Auto-start |
|---------|---------|---------|------------|
| `swa-obelisk.service` | ✅ | ✅ active | ✅ multi-user.target |
| `swa-obelisk-train.service` | disabled (timer) | — | ✅ via timer |
| `swa-obelisk-train.timer` | ✅ | ✅ waiting | ✅ timers.target |

PID raiz do watchdog: `/var/run/aeon/obelisk.pid`
Heartbeat: `/var/run/aeon/obelisk.heartbeat`

## Dependências Python

```
bitsandbytes>=0.43.0    ← quantização 4-bit NF4
blake3                  ← assinatura criptográfica dos 5 subespaços
torch                   ← carregamento do Gemma + LoRA
transformers            ← HuggingFace pipeline
peft                    ← LoRA/QLoRA training
datasets                ← dataset builder
accelerate              ← treino distribuído (se houver GPU)
```

## Validação Rápida

```bash
# Import e dataset
python3 -c "from cortex.obelisk.model import ObeliskModel, OBELISK_LABELS; \
            from cortex.obelisk.blake.tensor import BlakeGravityTensor; \
            from cortex.obelisk.blake.classifier import BlakeSpaceClassifier; \
            from cortex.obelisk.daemon import ObeliskDaemon; \
            print('OK:', OBELISK_LABELS)"

# Tensor de gravidade (offline, sem modelo)
python3 -c "from cortex.obelisk.blake.collector import collect_all; \
            from cortex.obelisk.blake.tensor import BlakeGravityTensor; \
            b = collect_all(); t = BlakeGravityTensor(); \
            v = t.compute(b); print('tensor:', [round(x,3) for x in v], '→', t.classify(v))"
```

## Próximas Etapas de Desenvolvimento

### 1. Coleta de Dados Reais (prioridade máxima)
```bash
# Executar watchdog por horas/dias registrando tensores e decisões
python cortex/obelisk/main.py
# TODO: adicionar logging persistente dos tensores para análise offline
```
- Adicionar `--log-tensors` flag que salva CSV/hdf5 de todos os tensores
- Coletar ~10k amostras para validação estatística

### 2. Ablação (comparação de classificadores)
Implementar script que compara:
- Fallback onboard (regras determinísticas)
- LLM 4-bit + LoRA (atual)
- LLM full-precision (sem quantização)
- MLP clássico (sklearn: RandomForest, SVM)
- Métrica: F1-score por classe, matriz de confusão, latência

### 3. Testes de Caos Controlado
Script que injeta mutações nos 5 subespaços:
```bash
# Simular corrupção de arquivos
echo "" > /services/model-with-adpter-logic-semantic/cortex/obelisk/model.py

# Simular falha de HW
# Simular deriva de sensor no manifold
```
Verificar se o watchdog detecta em < 30s.

### 4. Bug Hunting Conhecido
- [ ] **Excessivo torch.compile workers** (10 workers, 1.8GB RAM) — adicionar `TORCH_COMPILE_WORKERS=1` ou `--disable-compile` no main.py
- [ ] **Loop Principal Bloqueante** — main.py usa `time.sleep(30)` que bloqueia sinais. Migrar para `select` ou `signal` com event loop assíncrono
- [ ] **sem device_map="auto"** — model.py carrega com `device_map=None`, forçando CPU mesmo com GPU disponível

### 5. Melhoria Classe `regression_to_rlhf`
A classe é a mais abstrata — detecta "regressão a alinhamento clássico
(RLHF)" que no contexto do SWA é considerado anomalia (pois o SWA usa
alinhamento topológico próprio). Precisa de:
- Template mais específico no dataset
- Feature engineering adicional no tensor (e5 cobre parcialmente)
- Validação com especialista de domínio

### 6. Integração com Outros Serviços SWA
- `swa-dream.service` — pode fornecer "dream states" como subespaço adicional
- `swa-mcp.service` — pode expor tensor via MCP tool
- `swa-subcortex.service` — pode receber comandos do obelisco

### 7. Artigo/POC
`docs/POC_OBELISK_BLAKE_GRAVITY.md` contém:
- Diagramas ASCII da arquitetura
- Cenários de mutação com dados reais
- Matriz de decisão com limiares
- Comparação com trabalhos relacionados
- Submissão alvo: ICML/NeurIPS/ISCA 2027

## Comandos Úteis

```bash
# Deploy
sudo bash deploy/install.sh

# Monitoramento
journalctl -u swa-obelisk -f
systemctl status swa-obelisk

# Treino manual
python cortex/obelisk/dataset.py
python cortex/obelisk/train.py

# Execução única
python cortex/obelisk/daemon.py --once

# Teste de módulos
python3 -m cortex.obelisk.blake.collector
```

## Notas Técnicas

- **Zero imports do kernel SWA:** obelisco é completamente desacoplado
  (não importa `swa.*`). Comunicação com o kernel é via IPC socket Unix
  com frame `"!4096f I"` (matriz 64×64 + payload) + 5 subespaços Blake.
- **Modelo base:** Gemma-3-270M em 4-bit NF4 (double quant) com
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.
- **LoRA targets:** `["q_proj", "v_proj"]`, rank=8, alpha=16.
- **Dataset:** gerado por `dataset.py` com templates Blake (19 variações
  de 6 classes) + templates sistêmicos (12) + logs reais do `aeon_log.jsonl`.
- **Tensor normalização:** L1-normalizado para evitar que um eixo domine.
- **Heartbeat:** timestamp Unix escrito a cada 30s em `/var/run/aeon/`.
- **Rollback:** quando `lethal_mutation` detectado, daemon mata e
  reinicia o processo filho automaticamente.
