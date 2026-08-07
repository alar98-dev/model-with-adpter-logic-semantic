
- GELU → `W_b`: $k\to3072$.
- LayerNorm saída.
- Prefix_mode macro: Para v_global em geração.

## 3) Fórmula
$$V_{SWA}(v)=\mathrm{LayerNorm}(W_b\cdot\mathrm{GELU}(W_a v))$$

**Ortonormalidade:** $\mathcal{L}_{ortho}=\|W_b^\top W_b - I_k\|_F^2$

## 4) Perda
$\mathcal{L}=\mathcal{L}_{triplet} + \lambda_{ortho}\mathcal{L}_{ortho}$

## 5) Integração Gemma3-4B
```
Text → Gemma3 embed (float16, cpu) → SWA → LogicalEngine → v_prefix
v_prefix → cat(embeds prompt) → stream_generate (autoregressive)
```

## 6) Bias de Camada (Opc.)
$x^{l+1} = \mathrm{LayerNorm}(x^l + a^l + \mathrm{Bias}^l(v_\mathrm{global}))$

## 7) Tool-Aware
Regex tools → process_event → JSON forçado.

**Otimizações:** float16 Gemma3 (~8GB RAM), low_cpu_mem_usage.

---
*Atualizado Gemma3-4B (2024).*

