"""Protocolo interno SWA (métricas e inventário de modelos).

Versão estável do contrato JSON: incluir sempre a chave "protocol": "swa.internal.v1"
nas respostas dos endpoints /api/v1/internal/*.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "swa.internal.v1"


def envelope(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"protocol": PROTOCOL_VERSION, "kind": kind, **payload}


def _is_hf_weight_file(name: str) -> bool:
    lower = name.lower()
    if not (name.endswith(".safetensors") or name.endswith(".bin")):
        return False
    return "model" in lower


def verify_hf_model_dir(model_dir: str) -> Dict[str, Any]:
    """Verifica layout HuggingFace local: config, pesos únicos ou índice + shards."""
    path = os.path.abspath(model_dir)
    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.isdir(path),
        "config_present": False,
        "tokenizer_present": False,
        "weights_ready": False,
        "weight_files": [],
        "index_weight_map_unique": [],
        "missing_shards": [],
        "issues": [],
    }
    if not out["exists"]:
        out["issues"].append("directory_missing")
        return out

    cfg = os.path.join(path, "config.json")
    out["config_present"] = os.path.isfile(cfg)
    if not out["config_present"]:
        out["issues"].append("missing_config_json")

    for tok in ("tokenizer.json", "tokenizer.model"):
        if os.path.isfile(os.path.join(path, tok)):
            out["tokenizer_present"] = True
            break
    if not out["tokenizer_present"]:
        out["issues"].append("missing_tokenizer")

    names = os.listdir(path)
    weight_files = sorted(n for n in names if _is_hf_weight_file(n))
    out["weight_files"] = weight_files

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                idx = json.load(f)
            wm = idx.get("weight_map") or {}
            unique_shards = sorted(set(wm.values())) if isinstance(wm, dict) else []
            out["index_weight_map_unique"] = unique_shards
            missing = [s for s in unique_shards if not os.path.isfile(os.path.join(path, s))]
            out["missing_shards"] = missing
            out["weights_ready"] = out["config_present"] and not missing and bool(unique_shards)
            if missing:
                out["issues"].append("incomplete_sharded_weights")
        except (OSError, json.JSONDecodeError) as e:
            out["issues"].append(f"index_read_error:{e}")
            out["weights_ready"] = False
    else:
        out["weights_ready"] = out["config_present"] and bool(weight_files)
        if out["config_present"] and not weight_files:
            out["issues"].append("no_weight_files")

    return out


def scan_models_root(models_root: str) -> Dict[str, Any]:
    """Lista subdiretórios imediatos de models_root com verificação HF resumida."""
    root = os.path.abspath(models_root)
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return {"models_root": root, "exists": False, "entries": []}

    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub) or name.startswith("."):
            continue
        v = verify_hf_model_dir(sub)
        entries.append(
            {
                "name": name,
                "relative": os.path.join(os.path.basename(root), name).replace("\\", "/"),
                "weights_ready": v["weights_ready"],
                "weight_shard_count": len(v["weight_files"]) or len(v["index_weight_map_unique"]),
                "issues": v["issues"],
            }
        )
    return {"models_root": root, "exists": True, "entries": entries}


def latency_probe(
    engine: Any,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Mede o mesmo caminho que /api/v1/chat (run_optimized_stream com prompt simples)."""
    if engine.gemma_gen is None:
        return {"error": "gemma_not_loaded", "use_fake_or_missing_weights": True}

    t0 = time.perf_counter()
    token_count = 0
    t_first: Optional[float] = None
    try:
        for _tok in engine.run_optimized_stream(
            {"prompt": prompt},
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            token_count += 1
    except ValueError as e:
        t_end = time.perf_counter()
        return {
            "error": str(e),
            "turn_total_ms": round((t_end - t0) * 1000.0, 3),
        }
    t_end = time.perf_counter()
    turn_total_ms = (t_end - t0) * 1000.0
    ttft_ms = (t_first - t0) * 1000.0 if t_first is not None else None
    gen_ms = (t_end - t_first) * 1000.0 if t_first is not None else turn_total_ms
    return {
        "prompt_chars": len(prompt),
        "max_new_tokens_cap": max_new_tokens,
        "tokens_yielded": token_count,
        "path": "run_optimized_stream",
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "turn_total_ms": round(turn_total_ms, 3),
        "generation_window_ms": round(gen_ms, 3),
        "tokens_per_s": round(token_count / (gen_ms / 1000.0), 3) if gen_ms > 0 else None,
    }
