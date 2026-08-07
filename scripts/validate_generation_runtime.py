#!/usr/bin/env python3
"""Validate generation() runtime for the configured Gemma3 stack.

Standalone script (NO FastAPI startup coupling).

Checks:
- model/tokenizer load (real HF)
- generate() returns output

Usage:
  python scripts/validate_generation_runtime.py \
    --model-dir models/gemma-3-4-b \
    --device cpu \
    --max-new-tokens 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional


def _now() -> float:
    return time.perf_counter()


def _try_load_model(model_dir: str, device: str, max_memory: Optional[dict] = None) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = {
        "device_map": "auto" if device != "cpu" else None,
    }
    # For CPU: prefer explicit to avoid device_map weirdness.
    if device == "cpu":
        kwargs.pop("device_map", None)

    if max_memory:
        kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **{k: v for k, v in kwargs.items() if v is not None},
    )

    if device == "cpu":
        model = model.cpu()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--device", default=os.environ.get("GEMMA_DEVICE", "cpu"))
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--prompt", default="Hello, semantic warp adapters:")
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "device": args.device,
        "runtime": {},
        "generation": {},
    }

    try:
        import transformers
        report["runtime"]["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        report["runtime"]["transformers"] = None

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        t0 = _now()
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        report["generation"]["tokenizer_load_ms"] = (_now() - t0) * 1000
        report["generation"]["tokenizer_class"] = tok.__class__.__name__
    except Exception as e:
        report["generation"]["tokenizer_ok"] = False
        report["generation"]["tokenizer_error"] = str(e)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    # Load model
    try:
        t0 = _now()
        model = _try_load_model(args.model_dir, args.device)
        report["generation"]["model_load_ms"] = (_now() - t0) * 1000
        report["generation"]["model_class"] = model.__class__.__name__
    except Exception as e:
        report["generation"]["model_ok"] = False
        report["generation"]["model_error"] = str(e)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3

    # Generate
    try:
        import torch

        model.eval()
        inputs = tok(args.prompt, return_tensors="pt")
        if args.device != "cpu":
            # Best-effort move.
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = _now()
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        gen_ms = (_now() - t0) * 1000

        text = tok.decode(out_ids[0], skip_special_tokens=True)

        report["generation"].update(
            {
                "generate_ok": True,
                "generate_ms": gen_ms,
                "output_text_head": text[:400],
            }
        )
    except Exception as e:
        report["generation"]["generate_ok"] = False
        report["generation"]["generate_error"] = str(e)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 4

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

