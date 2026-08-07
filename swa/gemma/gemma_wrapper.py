"""Thin wrapper for loading Gemma HF models with device/dtype resolution."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import get_safe_device, pick_dtype


class GemmaWrapper:
    """Loads a Gemma model + tokenizer and resolves device/dtype."""

    def __init__(self, model_dir: str, device: str = "cpu", *, dtype_override: Optional[torch.dtype] = None):
        self.device_str = get_safe_device(device)
        self.device = torch.device(self.device_str)
        self.dtype = pick_dtype(self.device_str, override=dtype_override)

        print(f"🔧 GemmaWrapper: loading from {model_dir} -> device={self.device_str}, dtype={self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
