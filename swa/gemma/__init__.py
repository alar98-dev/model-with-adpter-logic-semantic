"""Gemma integration package — split from the monolithic ``gemma_integration.py``.
"""
from swa.gemma.device import get_safe_device, _log_xpu_device_fingerprint
from swa.gemma.generator import GemmaGenerator

__all__ = [
    "get_safe_device",
    "_log_xpu_device_fingerprint",
    "GemmaGenerator",
]
