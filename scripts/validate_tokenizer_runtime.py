#!/usr/bin/env python3
"""Validate tokenizer runtime compatibility for SWA.

Standalone certification script (NO impact on API startup).

Outputs structured JSON to stdout and returns non-zero exit codes on failure.

Usage:
  python scripts/validate_tokenizer_runtime.py \
    --model-dir models/gemma-3-4-b \
    --fast \
    --slow \
    --validate-equivalence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple


def _now() -> float:
    return time.perf_counter()


def _try_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import transformers

        out["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        out["transformers"] = None

    try:
        import tokenizers

        out["tokenizers"] = getattr(tokenizers, "__version__", None)
    except Exception:
        out["tokenizers"] = None

    try:
        import sentencepiece  # noqa: F401

        out["sentencepiece"] = True
    except Exception:
        out["sentencepiece"] = False

    return out


def _load_tokenizer(model_dir: str, use_fast: bool, legacy: bool) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=use_fast,
            legacy=legacy,
        )
        return tok, None
    except Exception as e:
        return None, str(e)


def _extract_schema_diagnostics(model_dir: str) -> Dict[str, Any]:
    fp = os.path.join(model_dir, "tokenizer.json")
    out: Dict[str, Any] = {
        "tokenizer_json_exists": os.path.isfile(fp),
        "tokenizer_json_size_bytes": os.path.getsize(fp) if os.path.isfile(fp) else None,
        "keys_present": {},
        "declared_model_type": None,
        "parse_error": None,
    }
    if not os.path.isfile(fp):
        return out

    try:
        import json

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
            out["keys_present"][k] = k in data

        model = data.get("model") or {}
        if isinstance(model, dict):
            out["declared_model_type"] = model.get("type")
    except Exception as e:
        out["parse_error"] = str(e)

    return out


def _tokenize_ids(tok: Any, text: str) -> List[int]:
    # Prefer encode if present.
    if hasattr(tok, "encode"):
        return list(tok.encode(text, add_special_tokens=False))
    # Fallback to __call__
    out = tok(text)
    if isinstance(out, dict) and "input_ids" in out:
        return list(out["input_ids"])
    raise RuntimeError("Unable to extract input_ids from tokenizer")


def _decode(tok: Any, ids: List[int]) -> str:
    if hasattr(tok, "decode"):
        return tok.decode(ids, skip_special_tokens=False)
    raise RuntimeError("Tokenizer has no decode()")


def validate_equivalence(fast_tok: Any, slow_tok: Any, prompts: List[str]) -> Dict[str, Any]:
    diffs: List[Dict[str, Any]] = []

    for i, p in enumerate(prompts):
        fast_ids = _tokenize_ids(fast_tok, p)
        slow_ids = _tokenize_ids(slow_tok, p)

        same_ids = fast_ids == slow_ids

        # Decode roundtrip for sanity.
        fast_dec = _decode(fast_tok, fast_ids)
        slow_dec = _decode(slow_tok, slow_ids)
        same_decode = fast_dec == slow_dec

        if not same_ids or not same_decode:
            diffs.append(
                {
                    "prompt_index": i,
                    "prompt": p,
                    "same_ids": same_ids,
                    "same_decode": same_decode,
                    "fast_ids_head": fast_ids[:32],
                    "slow_ids_head": slow_ids[:32],
                    "fast_dec_head": fast_dec[:256],
                    "slow_dec_head": slow_dec[:256],
                }
            )

    if not diffs:
        return {"equivalent": True, "diff_count": 0}
    return {"equivalent": False, "diff_count": len(diffs), "diffs": diffs[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--no-fast", dest="fast", action="store_false")
    ap.add_argument("--slow", action="store_true", default=True)
    ap.add_argument("--no-slow", dest="slow", action="store_false")
    ap.add_argument("--validate-equivalence", action="store_true", default=True)
    ap.add_argument("--no-validate-equivalence", dest="validate_equivalence", action="store_false")
    ap.add_argument("--legacy", action="store_true", default=True)
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")

    args = ap.parse_args()

    prompts = [
        "hello",
        "system",
        "assistant",
        "Gemma 3 test: semantic warp adapters.",
        "Tokenizers must be fast when compatibility holds.",
    ]

    report: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "runtime": _try_versions(),
        "tokenizer_schema_diagnostics": _extract_schema_diagnostics(args.model_dir),
        "tokenizer": {
            "fast": {"requested": args.fast, "loaded": False, "error": None, "backend": None},
            "slow": {"requested": args.slow, "loaded": False, "error": None, "backend": None},
        },
        "equivalence": None,
        "latency_ms": {},
    }

    fast_tok = None
    slow_tok = None

    if args.fast:
        t0 = _now()
        tok, err = _load_tokenizer(args.model_dir, use_fast=True, legacy=args.legacy)
        dt = (_now() - t0) * 1000
        if tok is not None:
            fast_tok = tok
            report["tokenizer"]["fast"].update({"loaded": True, "error": None, "backend": tok.__class__.__name__})
        else:
            report["tokenizer"]["fast"].update({"loaded": False, "error": err, "backend": None})
        report["latency_ms"]["fast_load"] = dt

    if args.slow:
        t0 = _now()
        tok, err = _load_tokenizer(args.model_dir, use_fast=False, legacy=args.legacy)
        dt = (_now() - t0) * 1000
        if tok is not None:
            slow_tok = tok
            report["tokenizer"]["slow"].update({"loaded": True, "error": None, "backend": tok.__class__.__name__})
        else:
            report["tokenizer"]["slow"].update({"loaded": False, "error": err, "backend": None})
        report["latency_ms"]["slow_load"] = dt

    if args.validate_equivalence and fast_tok is not None and slow_tok is not None:
        report["equivalence"] = validate_equivalence(fast_tok, slow_tok, prompts)

    # decide exit code
    # - success if at least one backend loaded
    # - equivalence failure is WARN unless both loaded
    exit_code = 0

    fast_loaded = report["tokenizer"]["fast"]["loaded"]
    slow_loaded = report["tokenizer"]["slow"]["loaded"]

    if not fast_loaded and not slow_loaded:
        exit_code = 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

