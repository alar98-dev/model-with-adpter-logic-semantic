#!/usr/bin/env python3
"""Checagem rápida de aceleração (CUDA / Intel XPU) para alinhar GEMMA_DEVICE no .env.

Os IDs PCI do Windows (ex.: VEN_8086&DEV_A7AC) não são lidos aqui; defina opcionalmente
INTEL_GPU_PCI_IDS no ambiente só para aparecer no relatório.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    pci_note = os.environ.get("INTEL_GPU_PCI_IDS", "").strip()
    if pci_note:
        print("INTEL_GPU_PCI_IDS:", pci_note)
    else:
        print("INTEL_GPU_PCI_IDS: (opcional) ex.: VEN_8086&DEV_A7AC — apenas referência")

    try:
        import torch
    except ImportError:
        print("PyTorch não instalado.", file=sys.stderr)
        return 1

    print("torch:", torch.__version__)
    print("CUDA disponível:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device 0:", torch.cuda.get_device_name(0))

    has_xpu_mod = hasattr(torch, "xpu")
    print("torch.xpu presente:", has_xpu_mod)
    if has_xpu_mod:
        print("XPU (Intel GPU) disponível:", torch.xpu.is_available())
        if torch.xpu.is_available():
            try:
                print("  device 0:", torch.xpu.get_device_name(0))
            except Exception as e:
                print("  (nome do device não obtido:", e, ")")

    print()
    if has_xpu_mod and torch.xpu.is_available():
        print("Sugestão: GEMMA_DEVICE=xpu   GEMMA_TORCH_DTYPE=bfloat16")
        print("           SWA_INTRAOP_THREADS=4   # ajuste se CPU ficar saturada")
    elif torch.cuda.is_available():
        print("Sugestão: GEMMA_DEVICE=cuda")
    else:
        print("Sugestão: GEMMA_DEVICE=cpu")
        print("Para iGPU Intel: instale o PyTorch build com suporte XPU (documentação Intel)")
        print("e drivers; no WSL2 confira se a GPU é exposta ao Linux conforme a Microsoft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
