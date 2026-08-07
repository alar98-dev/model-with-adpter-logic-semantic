"""Tests for swa.gemma — device resolution and dtype selection.

Actual model loading tests require GPU/XPU hardware and model weights.
These are unit tests for the utility functions only.
"""

from __future__ import annotations

import os

import torch
import pytest

from swa.gemma.device import get_safe_device, pick_dtype


class TestGetSafeDevice:
    def test_cpu(self) -> None:
        assert get_safe_device("cpu") == "cpu"

    def test_none_uses_env(self) -> None:
        # Ensure no override
        old = os.environ.pop("GEMMA_DEVICE", None)
        assert get_safe_device(None) == "cpu"
        if old is not None:
            os.environ["GEMMA_DEVICE"] = old

    def test_cuda_fallback_when_unavailable(self) -> None:
        if not torch.cuda.is_available():
            assert get_safe_device("cuda") == "cpu"
            assert get_safe_device("gpu") == "cpu"

    def test_xpu_fallback_when_unavailable(self) -> None:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            assert get_safe_device("xpu") == "cpu"

    def test_unknown_fallback_to_cpu(self) -> None:
        assert get_safe_device("nonexistent_device") == "cpu"


class TestPickDtype:
    def test_float32(self) -> None:
        os.environ["GEMMA_TORCH_DTYPE"] = "float32"
        assert pick_dtype("cpu") == torch.float32

    def test_bfloat16(self) -> None:
        os.environ["GEMMA_TORCH_DTYPE"] = "bfloat16"
        assert pick_dtype("xpu") == torch.bfloat16

    def test_heuristic_cpu_float32(self) -> None:
        os.environ.pop("GEMMA_TORCH_DTYPE", None)
        assert pick_dtype("cpu") == torch.float32

    def test_heuristic_xpu_bfloat16(self) -> None:
        os.environ.pop("GEMMA_TORCH_DTYPE", None)
        assert pick_dtype("xpu") == torch.bfloat16
