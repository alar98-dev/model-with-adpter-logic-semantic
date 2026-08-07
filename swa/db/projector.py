"""Vector projection utilities for database operations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from swa.db.base import BaseController


class ProceduralProjector:
    """Projects natural language intents into structured vector operations."""

    _PATTERN_MAP = [
        (r"(?i)(buscar|encontrar|achar|search|find)\s+(.+)", "search"),
        (r"(?i)(registrar|salvar|store|save)\s+(.+)", "store"),
        (r"(?i)(apagar|remover|delete|remove)\s+(.+)", "delete"),
    ]

    def __init__(self):
        self.input_dim = 32
        self.output_dim = 2560
        self.device = "cpu"

    @classmethod
    def project(cls, intent: str) -> Dict[str, Any]:
        for pattern, action in cls._PATTERN_MAP:
            match = re.search(pattern, intent)
            if match:
                return {
                    "action": action,
                    "target": match.group(2).strip(),
                    "confidence": 0.8,
                }
        return {"action": "unknown", "confidence": 0.2}

    def project_inline(self, x, bias: float):
        """Legacy inline projection used by cortex/run.py."""
        W = getattr(self, "_W_torch", None)
        if W is None:
            torch.manual_seed(42)
            self._W_torch = torch.randn(self.output_dim, self.input_dim, dtype=torch.float32) * 0.01
            self._b_torch = torch.full((self.output_dim,), bias, dtype=torch.float32)
        if isinstance(x, torch.Tensor):
            return self._W_torch @ x.float().to(self._W_torch.device) + self._b_torch
        return self._W_torch @ torch.from_numpy(x).float() + self._b_torch
