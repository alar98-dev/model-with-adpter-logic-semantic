# Seleção de Modelo — Gemma3-4B Otimizada para RAM

## Modelo Principal: Gemma3-4B

**Por quê Gemma3-4B?**
- ~4B params: Equilíbrio performance/latência em RAM (8-16GB).
- Substitui dual-model (Llama8B + Gemma1B) por single-model eficiente.
- Suporte nativo HF Transformers, chat_template.json pronto.
- Otimizada lógica via SWA (micro/macro adapters).

**Hardware Compatibilidade:**
- **RAM 8GB+ (CPU/Xeon):** float16 (~8GB), low_cpu_mem_usage=True.
- **GPU 6GB+:** device='cuda', opc. 4-bit (bitsandbytes NF4).
- **Fallback:** FakeVectorizer dev/teste.

## Configuração Quantização RAM

Em `gemma_integration.py` (padrão):
```python
self.lm = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # ~8GB vs 16GB float32
    device_map='cpu',
    low_cpu_mem_usage=True
)
```

**Otimizações Avançadas:**
- **4-bit CPU?** bitsandbytes (GPU-only); use ONNX int8 export.
- **Batch segs:** Chunking micro→macro (ver arquitetura).
- **Memória:** shm_size=2g Docker, mem_limit=16g.

## Integração SWA
```
d_model=3072 (Gemma3-4B), k_local=32, k_macro=32 (prefix_mode=True)
```
- Embed: `last_hidden_state.mean(dim=1)`
- Fluxo: Vectorizer → SWA → LogicalEngine → v_prefix → stream_with_prefix.

## Setup Diretórios
```
models/gemma3-4b/
├── config.json
├── tokenizer.json
├── model.safetensors (float16)
└── chat_template.json
```
Clone: `scripts/clone_gemma.sh`

## Alternativas (Futuro)
- **Qwen2-7B:** Se >16GB RAM.
- **Mistral-7B:** GPU-only.

**Teste:**
```python
from swa.gemma_integration import GemmaGenerator
gen = GemmaGenerator('models/gemma3-4b', device='cpu')
print(gen.generate("Teste."))
```

---
*Atualizado: Gemma3-4B RAM-Opt (2024). Removidas refs antigas Llama/Gemma1B.*

