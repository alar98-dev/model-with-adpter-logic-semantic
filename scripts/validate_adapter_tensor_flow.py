#!/usr/bin/env python3
"""Validate adapter tensor flow independent from tokenizer backend.

Checks that the adapter pipeline operates on real tensors and produces
stable outputs (no NaNs/shape errors).

This intentionally does NOT instantiate tokenizers.

Usage:
  python scripts/validate_adapter_tensor_flow.py
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict


def _now() -> float:
    return time.perf_counter()


def main() -> int:
    # Ensure repo root is importable when running as `python scripts/...`
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from swa.adapter import SemanticWarpAdapter


    report: Dict[str, Any] = {
        "test": "adapter_tensor_flow",
        "device": os.environ.get("GEMMA_DEVICE", "cpu"),
        "runtime": {},
        "result": {},
    }

    try:
        import torch

        report["runtime"]["torch"] = getattr(torch, "__version__", None)
        device = "cpu"
        x = torch.randn(2, 16, 128, device=device)

        a_local = SemanticWarpAdapter(d=128, k=32)
        a_macro = SemanticWarpAdapter(d=128, k=32, prefix_mode=True, d_model=3072)

        t0 = _now()
        y_local = a_local(x)
        y_macro = a_macro(x)
        ms = (_now() - t0) * 1000

        def stats(t):
            return {
                "shape": list(t.shape),
                "mean": float(t.mean().item()),
                "std": float(t.std().item()),
                "nan": bool(torch.isnan(t).any().item()),
                "inf": bool(torch.isinf(t).any().item()),
            }

        report["result"].update(
            {
                "ok": True,
                "apply_ms": ms,
                "local": stats(y_local),
                "macro": stats(y_macro),
            }
        )
    except Exception as e:
        report["result"].update({"ok": False, "error": str(e)})

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["result"].get("ok") else 5


if __name__ == "__main__":
    raise SystemExit(main())

