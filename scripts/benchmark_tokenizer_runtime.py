#!/usr/bin/env python3
"""Benchmark tokenizer runtime performance (FAST vs SLOW).

Standalone benchmarking script (no FastAPI integration).

Measures:
- load latency (ms)
- warmup latency
- tokens/sec
- per-call latency (basic p50/p95)

Usage:
  python scripts/benchmark_tokenizer_runtime.py --model-dir models/gemma-3-4-b --backend fast
  python scripts/benchmark_tokenizer_runtime.py --model-dir models/gemma-3-4-b --backend slow
  python scripts/benchmark_tokenizer_runtime.py --model-dir models/gemma-3-4-b --backend both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple


def _now() -> float:
    return time.perf_counter()


def _load_tokenizer(model_dir: str, use_fast: bool, legacy: bool) -> Tuple[Any, float]:
    from transformers import AutoTokenizer

    t0 = _now()
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast, legacy=legacy)
    dt_ms = (_now() - t0) * 1000
    return tok, dt_ms


def _encode_ids(tok: Any, text: str) -> List[int]:
    if hasattr(tok, "encode"):
        return list(tok.encode(text, add_special_tokens=False))
    out = tok(text)
    if isinstance(out, dict) and "input_ids" in out:
        return list(out["input_ids"])
    raise RuntimeError("tokenizer output does not include input_ids")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[int(f)] * (c - k)
    d1 = xs[int(c)] * (k - f)
    return d0 + d1


def bench_backend(model_dir: str, use_fast: bool, legacy: bool, iterations: int, batch_size: int) -> Dict[str, Any]:
    tok, load_ms = _load_tokenizer(model_dir, use_fast=use_fast, legacy=legacy)

    corpus = [
        "hello world",
        "system: you are a semantic warp adapter",
        "assistant: embeddings geometry preserved",
        "Gemma 3 tokenizer benchmark for FAST vs SLOW",
        "Tokenizers define tokens; adapters operate on hidden-state tensors.",
    ]

    # Build a batch of inputs
    # We keep it variable by cycling through corpus.
    texts = [corpus[i % len(corpus)] for i in range(batch_size)]

    # Warmup
    warmup_n = max(5, min(25, iterations // 20))
    t_w0 = _now()
    for _ in range(warmup_n):
        for t in texts:
            _ = _encode_ids(tok, t)
    warmup_ms = (_now() - t_w0) * 1000

    # Timed runs
    latencies_ms: List[float] = []
    total_tokens = 0

    t_first = None
    first_latency_ms = None

    # We do: each iteration encodes all texts; latency per call is per-text.
    # tokens/sec is based on total tokens / total time.
    t0 = _now()
    for it in range(iterations):
        for j, t in enumerate(texts):
            t_call0 = _now()
            ids = _encode_ids(tok, t)
            t_call1 = _now()

            if t_first is None:
                t_first = it
                first_latency_ms = (t_call1 - t_call0) * 1000

            latencies_ms.append((t_call1 - t_call0) * 1000)
            total_tokens += len(ids)
    total_ms = (_now() - t0) * 1000

    tokens_per_sec = (total_tokens / (total_ms / 1000.0)) if total_ms > 0 else None

    return {
        "backend": "fast" if use_fast else "slow",
        "loaded": True,
        "tokenizer_class": tok.__class__.__name__,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "iterations": iterations,
        "batch_size": batch_size,
        "total_ms": total_ms,
        "tokens_total": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "first_token_latency_ms": first_latency_ms,
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "max": max(latencies_ms) if latencies_ms else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gemma-3-4-b"))
    ap.add_argument("--backend", choices=["fast", "slow", "both"], default="both")
    ap.add_argument("--legacy", action="store_true", default=True)
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)

    args = ap.parse_args()

    report: Dict[str, Any] = {
        "model_dir": args.model_dir,
        "backend": args.backend,
        "results": {},
    }

    try:
        if args.backend in ("fast", "both"):
            report["results"]["fast"] = bench_backend(args.model_dir, use_fast=True, legacy=args.legacy, iterations=args.iterations, batch_size=args.batch_size)
    except Exception as e:
        report["results"]["fast"] = {"loaded": False, "error": str(e)}

    try:
        if args.backend in ("slow", "both"):
            report["results"]["slow"] = bench_backend(args.model_dir, use_fast=False, legacy=args.legacy, iterations=args.iterations, batch_size=args.batch_size)
    except Exception as e:
        report["results"]["slow"] = {"loaded": False, "error": str(e)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

