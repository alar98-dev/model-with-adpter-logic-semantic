#!/usr/bin/env python3
"""Export runtime compatibility matrix for SWA.

Creates runtime_compatibility_matrix.json next to repo root.

This is used to freeze/observe the HF ecosystem (transformers/tokenizers/sentencepiece)
against local tokenizer schema (tokenizer.json) for Gemma3.

Usage:
  python scripts/export_runtime_compatibility.py --model-dir models/gemma-3-4-b
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional


def _get_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _try_sentencepiece_present() -> bool:
    try:
        import sentencepiece  # noqa: F401

        return True
    except Exception:
        return False


def _read_tokenizer_schema(model_dir: str) -> Dict[str, Any]:
    fp = os.path.join(model_dir, "tokenizer.json")
    out: Dict[str, Any] = {
        "tokenizer_json_exists": os.path.isfile(fp),
        "tokenizer_json_size_bytes": os.path.getsize(fp) if os.path.isfile(fp) else None,
        "declared_model_type": None,
        "schema_version": None,
        "keys_present": {},
        "parse_error": None,
    }
    if not os.path.isfile(fp):
        return out

    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
            out["keys_present"][k] = k in data

        model = data.get("model") or {}
        if isinstance(model, dict):
            out["declared_model_type"] = model.get("type")

        # Some tokenizers include schema_version; may not exist.
        out["schema_version"] = data.get("version") or data.get("schema_version")
    except Exception as e:
        out["parse_error"] = str(e)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--out", default="runtime_compatibility_matrix.json")
    args = ap.parse_args()

    matrix: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "gemma_runtime": {
            # Gemma version is best-effort; in local repos it might not exist.
            "gemma_version": os.path.basename(args.model_dir),
        },
        "runtime": {
            "transformers": _get_version("transformers"),
            "tokenizers": _get_version("tokenizers"),
            "sentencepiece_installed": _try_sentencepiece_present(),
            "torch": _get_version("torch"),
            "bitsandbytes": _get_version("bitsandbytes"),
        },
        "tokenizer": {
            **_read_tokenizer_schema(args.model_dir),
        },
        "capabilities": {
            "fast_supported": None,
            "slow_supported": None,
            "xpu_support": os.environ.get("GEMMA_DEVICE", None),
            "quantization_support": "4bit" if os.environ.get("LOAD_IN_4BIT", "true").lower() in ("1", "true", "yes") else None,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}")
    print(json.dumps(matrix, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

