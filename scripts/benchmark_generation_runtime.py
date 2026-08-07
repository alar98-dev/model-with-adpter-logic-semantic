#!/usr/bin/env python3
"""Benchmark generation runtime:

- Warmup + steady-state
- Streaming token throughput (tokens/sec)

Notes:
- Relies on the same injection pipeline as `validate_generation_pipeline.py`.
- KV cache behavior is controlled via env vars consumed by load_kv_runtime_from_env().

Usage:
  python scripts/benchmark_generation_runtime.py \
    --model-dir models/gemma-3-4-b \
    --device cpu \
    --prompt "Hello" \
    --max-new-tokens 16 \
    --rounds 4 \
    --warmup-rounds 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List


def _now() -> float:
    return time.perf_counter()


def _tokens_from_stream(tokens: List[str]) -> int:
    return len(tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--device", default=os.environ.get("GEMMA_DEVICE", "cpu"))

    ap.add_argument("--prompt", default=os.environ.get("SWA_PROMPT", "Hello"))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SWA_MAX_NEW_TOKENS", "16")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("SWA_TEMPERATURE", "0.7")))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("SWA_BENCH_ROUNDS", "4")))
    ap.add_argument("--warmup-rounds", type=int, default=int(os.environ.get("SWA_BENCH_WARMUP_ROUNDS", "1")))
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "env_kv": {
            "SWA_KV_USE_CACHE": os.environ.get("SWA_KV_USE_CACHE"),
            "SWA_MAX_SEQ_LEN": os.environ.get("SWA_MAX_SEQ_LEN"),
            "SWA_ATTN_IMPLEMENTATION": os.environ.get("SWA_ATTN_IMPLEMENTATION"),
        },
        "tokenizer_manager": {},
        "benchmark": {
            "warmup": [],
            "steady": [],
        },
    }

    from swa.tokenizer_manager import TokenizerManager
    from swa.adapter import SemanticWarpAdapter
    from swa.logical_engine import LogicalEngine

    tm = TokenizerManager()
    tm.validate(args.model_dir)
    tok = tm.load(args.model_dir)
    tm.warmup()
    report["tokenizer_manager"]["health"] = tm.health()
    if tok is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    a_local = SemanticWarpAdapter(d=128, k=32)
    a_macro = SemanticWarpAdapter(d=128, k=32, prefix_mode=True, d_model=3072)

    eng = LogicalEngine(
        model_dir=args.model_dir,
        adapter_local=a_local,
        adapter_macro=a_macro,
        device=args.device,
        use_gemma=True,
        tokenizer=tok,
    )

    # Prepare v_global once per round (process_event updates memory; we keep it deterministic by resetting)
    def run_one(round_id: int) -> Dict[str, Any]:
        eng.memory.clear()
        v_global = eng.process_event(args.prompt)
        toks: List[str] = []
        t0 = _now()
        for token in eng.generate_guided_stream(
            args.prompt,
            v_global,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            toks.append(token)
        t1 = _now()
        ms = (t1 - t0) * 1000
        n_tok = _tokens_from_stream(toks)
        tps = (n_tok / ((t1 - t0) if (t1 - t0) > 0 else 1e-9))
        return {
            "round": round_id,
            "ms": ms,
            "tokens": n_tok,
            "tokens_per_sec": tps,
            "head": "".join(toks)[:120],
        }

    for i in range(args.warmup_rounds):
        out = run_one(i)
        report["benchmark"]["warmup"].append(out)

    for i in range(args.rounds):
        out = run_one(i + args.warmup_rounds)
        report["benchmark"]["steady"].append(out)

    # Aggregate
    steady = report["benchmark"]["steady"]
    if steady:
        avg_ms = sum(x["ms"] for x in steady) / len(steady)
        avg_tps = sum(x["tokens_per_sec"] for x in steady) / len(steady)
        report["benchmark"]["summary"] = {
            "avg_ms": avg_ms,
            "avg_tokens_per_sec": avg_tps,
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

