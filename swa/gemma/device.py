"""Device resolution and XPU detection utilities."""

from __future__ import annotations

import os
from typing import Optional

import torch


def _log_xpu_device_fingerprint() -> None:
    """Log Intel iGPU/XPU properties when available."""
    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print("XPU probe: torch.xpu not available -> falling back to CPU")
            return

        props = torch.xpu.get_device_properties(0)
        name = getattr(props, "name", None)
        eu_count = getattr(props, "gpu_eu_count", None)
        print(f"🔍 Dispositivo XPU detectado (index=0): name={name} EU={eu_count}")

        for attr in ("pci_bus_id", "pci_address", "pci_device_id", "pci_vendor_id",
                     "device_id", "vendor_id"):
            if hasattr(props, attr):
                print(f"🔎 XPU prop {attr}={getattr(props, attr)}")

        vendor_id = getattr(props, "pci_vendor_id", getattr(props, "vendor_id", None))
        device_id = getattr(props, "pci_device_id", getattr(props, "device_id", None))
        if vendor_id is not None and device_id is not None:
            if int(vendor_id) == 0x8086 and int(device_id) == 0xA7AC:
                print("✅ ID iGPU Intel confirmado: PCI\\VEN_8086&DEV_A7AC (8086:A7AC)")
            else:
                print(f"⚠️ IDs XPU divergem: vendor={vendor_id} device={device_id}")
        else:
            print("ℹ️ Não foi possível ler vendor/device id via torch.xpu.get_device_properties()")
    except Exception as exc:
        print(f"⚠️ Falha ao inspecionar XPU: {exc}")


def get_safe_device(requested_device: Optional[str] = None) -> str:
    """Resolve a safe torch device string from user preference.

    Returns ``"cpu"``, ``"cuda"``, or ``"xpu"``.
    """
    r = (requested_device or os.environ.get("GEMMA_DEVICE", "cpu")).strip().lower()

    if r == "cpu":
        return "cpu"
    if r in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if r in ("xpu", "intel", "igpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            _log_xpu_device_fingerprint()
            return "xpu"
        print("WARNING: XPU requested but unavailable. Falling back to CPU.")
        return "cpu"
    return "cpu"


def pick_dtype(device_str: str, override: Optional[str] = None) -> torch.dtype:
    """Choose the torch dtype best suited for *device_str*.

    Respects ``GEMMA_TORCH_DTYPE`` env var when *override* is ``None``.
    """
    dtype_str = (override or os.environ.get("GEMMA_TORCH_DTYPE", "")).strip().lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    # Heuristic
    if device_str == "cpu":
        return torch.float32
    if device_str == "xpu":
        return torch.bfloat16
    return torch.float16
