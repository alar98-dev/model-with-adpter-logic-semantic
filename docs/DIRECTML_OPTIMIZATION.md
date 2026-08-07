# DirectML Optimization Guide — SWA LLM

## Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA DE OTIMIZAÇÃO GPU                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ATUAL (PyTorch XPU):                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Python → PyTorch → Intel Extension → Level-Zero → Intel GPU       │   │
│  │                                                                     │   │
│  │  • 80 EUs (Execution Units)                                         │   │
│  │  • 6.9GB memória compartilhada                                      │   │
│  │  • FP16 nativo                                                      │   │
│  │  • ~20-30 tok/s (estimado)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OTIMIZADO (DirectML):                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Python → ONNX Runtime → DirectML → DxCore → Intel GPU             │   │
│  │                                                                     │   │
│  │  • Otimizações DNN específicas                                      │   │
│  │  • Graph fusion automático                                          │   │
│  │  • Memória otimizada                                                │   │
│  │  • ~40-60 tok/s (estimado)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  VANTAGENS DO DIRECTML:                                                     │
│  • DxCore já disponível: /dev/dxg                                         │
│  • DirectML via WSL2 nativo                                                │
│  • ONNX Runtime otimizado para DNN                                         │
│  • Graph fusion (combina operações)                                        │
│  • Memória eficiente (sem overhead PyTorch)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Math: Por que DirectML é mais rápido?

### 1. Graph Fusion (Fusão de Grafos)

```
ANTES (PyTorch XPU):
  Operação 1: A = matmul(W1, x)    → Kernel 1
  Operação 2: B = add(A, b1)       → Kernel 2
  Operação 3: C = relu(B)          → Kernel 3
  → 3 launches de kernel, 3 acessos à memória

DEPOIS (DirectML com fusion):
  Operação 1: C = fusion(matmul+add+relu)(W1, x, b1)  → Kernel 1
  → 1 launch de kernel, 1 acesso à memória
```

Matematicamente:
$$\text{Fusion}(\text{matmul}, \text{add}, \text{relu}) = \text{relu}(W_1 \cdot x + b_1)$$

O DirectML combina automaticamente:
$$\text{Antes: } O(3k) \text{ kernels} \rightarrow \text{Depois: } O(k) \text{ kernels}$$

### 2. Memory Coalescing (Coalescência de Memória)

```
ANTES (PyTorch XPU):
  Thread 1 lê endereço 0x1000
  Thread 2 lê endereço 0x2000  → Acessos aleatórios
  Thread 3 lê endereço 0x3000

DEPOIS (DirectML):
  Thread 1 lê endereço 0x1000
  Thread 2 lê endereço 0x1004  → Acessos sequenciais
  Thread 3 lê endereço 0x1008
```

### 3. Pipeline Parallelism (Paralelismo de Pipeline)

```
ANTES (PyTorch XPU):
  Tempo: |---Kernel 1---|---Kernel 2---|---Kernel 3---|
  GPU:   [computando ]  [  idle      ]  [computando ]

DEPOIS (DirectML):
  Tempo: |---Kernel 1---|
  GPU:   [computando   ]
         |---Kernel 2---|
         [computando   ]
         |---Kernel 3---|
         [computando   ]
```

## Setup

### 1. Instalar DirectML

```bash
# No WSL2 Debian
pip install onnxruntime-directml

# Verificar disponibilidade
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Deve mostrar: ['DmlExecutionProvider', 'CPUExecutionProvider']
```

### 2. Converter Modelo para ONNX

```bash
# Converter Gemma-270M para ONNX
python scripts/directml_optimizer.py --convert --model models/gemma-3-270m-it
```

### 3. Rodar Benchmark

```bash
# Comparar XPU vs DirectML
python scripts/directml_optimizer.py --benchmark
```

## Uso no PyTest Visual

```python
from swa.llm_pytest.quantized_interpreter import QuantizedTestInterpreter

# Usar DirectML automaticamente
interpreter = QuantizedTestInterpreter(device="auto")
interpreter.load()  # Detecta DirectML automaticamente

# Explicar falha
explanation = interpreter.explain_failure(
    test_name="test_foo",
    error="AssertionError: 1 != 2"
)
```

## Métricas Esperadas

| Métrica | PyTorch XPU | DirectML | Melhoria |
|---------|-------------|----------|----------|
| Tokens/s | 20-30 | 40-60 | 2x |
| Latência | 50ms | 25ms | 2x |
| Memória | 512MB | 256MB | 2x |
| CPU Usage | 80% | 40% | 2x |

## Troubleshooting

### Erro: "DmlExecutionProvider not found"

```bash
# Verificar DirectX no WSL2
ls -la /dev/dxg

# Instalar DirectML
pip install onnxruntime-directml --force-reinstall
```

### Erro: "Out of memory"

```bash
# Reduzir batch size
# Usar quantização INT4 mais agressiva
# Fechar outros processos GPU
```

### Performance ruim

```bash
# Verificar se DxCore está sendo usado
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx', providers=['DmlExecutionProvider'])
print(sess.get_providers())
"
```
