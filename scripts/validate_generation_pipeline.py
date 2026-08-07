#!/usr/bin/env python3
"""Validate generation pipeline end-to-end:

TokenizerManager -> LogicalEngine(tokenizer injected) -> GemmaGenerator -> generate

Outputs a JSON report to stdout.

Usage:
  python scripts/validate_generation_pipeline.py \
    --model-dir models/gemma-3-4-b \
    --device cpu \
    --prompt "Hello" \
    --max-new-tokens 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict


def _now() -> float:
    return time.perf_counter()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--device", default=os.environ.get("GEMMA_DEVICE", "cpu"))
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SWA_MAX_NEW_TOKENS", "8")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("SWA_TEMPERATURE", "0.7")))
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "device": args.device,
        "tokenizer_manager": {},
        "shapes": {},
        "generation": {},
    }

    try:
        import sys
        sys.path.insert(0, os.getcwd())

        from swa.tokenizer_manager import TokenizerManager

        tm = TokenizerManager()

        report["tokenizer_manager"]["validate"] = tm.validate(args.model_dir)
        tok = tm.load(args.model_dir)
        tm.warmup()
        report["tokenizer_manager"]["health"] = tm.health()
        report["tokenizer_manager"]["has_tokenizer_obj"] = tok is not None

        if tok is None:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
    except Exception as e:
        report["tokenizer_manager"]["error"] = str(e)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    try:
        # Build engine (LogicalEngine now accepts tokenizer injection)
        import torch

        from swa.adapter import SemanticWarpAdapter
        from swa.logical_engine import LogicalEngine

        a_local = SemanticWarpAdapter(d=2560, k=32)
        a_macro = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)

        eng = LogicalEngine(
            model_dir=args.model_dir,
            adapter_local=a_local,
            adapter_macro=a_macro,
            device=args.device if args.device else "cpu",
            use_gemma=True,
            tokenizer=tok,
        )

        # Minimal adapter/tensor checks
        v = eng.vectorizer.embed([args.prompt])
        report["shapes"]["embedding_in"] = list(v.shape)
        v_local = eng.adapter_local(v)
        report["shapes"]["adapter_local"] = list(v_local.shape)
        v_global = eng.process_event(args.prompt)
        report["shapes"]["v_global"] = list(v_global.shape)

        # Non-streaming smoke
        t0 = _now()
        text = eng.generate_response(args.prompt)
        t1 = _now()
        report["generation"].update(
            {
                "generate_ok": True,
                "generate_ms": (t1 - t0) * 1000,
                "text_head": text[:400],
            }
        )

        # Streaming smoke
        t0 = _now()
        toks = []
        for token in eng.generate_guided_stream(
            args.prompt,
            v_global,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            toks.append(token)
        t1 = _now()
        report["generation"].update(
            {
                "stream_ok": True,
                "stream_ms": (t1 - t0) * 1000,
                "stream_tokens": len(toks),
                "stream_head": "".join(toks)[:400],
            }
        )

    except Exception as e:
        report["generation"]["error"] = str(e)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

