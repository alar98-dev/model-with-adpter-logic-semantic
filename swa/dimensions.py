"""Canonical vector contracts used by the SWA-Aeon runtime.

The runtime has three semantic spaces only:

* 640D  - native subcortex / sensor space
* 1152D - conscious model steering space
* 2560D - persistent memory space

Keeping the conversions here prevents ad-hoc ``repeat``/padding code from
silently changing the meaning of a vector.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn.functional as F

NATIVE_DIM = 640
MODEL_DIM = 1152
MEMORY_DIM = 2560
DIM_640 = NATIVE_DIM
DIM_1152 = MODEL_DIM
DIM_2560 = MEMORY_DIM
CANONICAL_DIMS = (NATIVE_DIM, MODEL_DIM, MEMORY_DIM)
PROJECTION_VERSION = 1


def validate_dim(value: torch.Tensor | Iterable[float], dim: int, *, name: str = "vector") -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() != dim:
        raise ValueError(f"{name} must contain exactly {dim} values, got {tensor.numel()}")
    return tensor.reshape(dim)


def fit_dim(value: torch.Tensor | Iterable[float], dim: int, *, normalize: bool = False) -> torch.Tensor:
    """Compatibility conversion for legacy persisted vectors.

    Truncation/padding is intentionally explicit and only used at a boundary.
    New projections should use :func:`project_vector`.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().reshape(-1)
    if tensor.numel() < dim:
        tensor = F.pad(tensor, (0, dim - tensor.numel()))
    elif tensor.numel() > dim:
        tensor = tensor[:dim]
    if normalize:
        tensor = F.normalize(tensor.float(), p=2, dim=0).to(tensor.dtype)
    return tensor


def projection_seed(source_dim: int, target_dim: int) -> int:
    digest = hashlib.sha256(f"swa-projection-v{PROJECTION_VERSION}:{source_dim}:{target_dim}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def projection_matrix(source_dim: int, target_dim: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a deterministic, normalized ``target_dim × source_dim`` matrix."""
    generator = torch.Generator(device="cpu").manual_seed(projection_seed(source_dim, target_dim))
    matrix = torch.randn((target_dim, source_dim), generator=generator, dtype=torch.float32)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-9)
    return matrix.to(device=device, dtype=dtype)


def project_vector(value: torch.Tensor | Iterable[float], source_dim: int, target_dim: int) -> torch.Tensor:
    source = fit_dim(value, source_dim, normalize=False).float()
    projected = projection_matrix(source_dim, target_dim, device=source.device, dtype=source.dtype) @ source
    return F.normalize(projected, p=2, dim=0)


def to_memory(value: torch.Tensor | Iterable[float], source_dim: int | None = None) -> torch.Tensor:
    source = source_dim or (value.numel() if isinstance(value, torch.Tensor) else len(list(value)))
    if source == MEMORY_DIM:
        return fit_dim(value, MEMORY_DIM, normalize=True).float()
    if source not in CANONICAL_DIMS:
        raise ValueError(f"unsupported semantic source dimension: {source}")
    return project_vector(value, source, MEMORY_DIM)
