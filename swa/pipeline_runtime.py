"""Configuração do pipeline interno: NUMA/threads e política de KV-cache no decode Gemma.

Lido de `os.environ` em cada passo de decode (`load_kv_runtime_from_env`) para permitir
ajuste dinâmico sem reiniciar o processo (útil com orchestration / sidecar que exporta env).

NUMA: `optimize_for_xeon` respeita `SWA_CPU_TOPOLOGY` (ver `swa.adapter`).
KV: `SWA_KV_USE_CACHE`, `SWA_MAX_SEQ_LEN`, `SWA_ATTN_IMPLEMENTATION`.

Nota (SWA × modelo bruto): o prefixo virtual SWA (`v_prefix`) concatena-se à matriz de
embeddings do prompt antes do forward autoregressivo; não substitui as camadas KV do
Transformer — apenas condiciona o estado inicial do decode. A política acima regula
`past_key_values` / `use_cache` desse decode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def apply_runtime_bootstrap() -> None:
    """Aplicar afinidade OpenMP/MKL/torch + NUMA antes de carregar modelos."""
    # Compat: alguns branches renomearam a função de otimização.
    from swa import adapter as _adapter

    if hasattr(_adapter, "optimize_for_xeon"):
        _adapter.optimize_for_xeon(0)
    elif hasattr(_adapter, "optimize_for_intel_120u_pcores"):
        _adapter.optimize_for_intel_120u_pcores()
    else:
        # fallback no-op
        pass



@dataclass(frozen=True)
class KVRuntimeConfig:
    use_cache: bool
    max_seq_len: int
    attn_implementation: Optional[str]


def load_kv_runtime_from_env() -> KVRuntimeConfig:
    """Snapshot atual da política de KV / atenção (chamar no início de cada decode)."""
    raw_max = os.environ.get("SWA_MAX_SEQ_LEN", "0").strip() or "0"
    try:
        max_seq = max(0, int(raw_max))
    except ValueError:
        max_seq = 0
    impl = os.environ.get("SWA_ATTN_IMPLEMENTATION", "").strip() or None
    use_cache = os.environ.get("SWA_KV_USE_CACHE", "true").lower() in ("1", "true", "yes", "on")
    return KVRuntimeConfig(
        use_cache=use_cache,
        max_seq_len=max_seq,
        attn_implementation=impl,
    )
