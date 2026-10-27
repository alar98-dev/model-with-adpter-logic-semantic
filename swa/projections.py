"""Versioned persistence for semantic projection matrices."""

from __future__ import annotations

import os
import logging
from typing import Optional

import torch

from swa.dimensions import PROJECTION_VERSION, projection_matrix


def load_or_create_projection(
    source_dim: int,
    target_dim: int,
    path: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Linear:
    """Load a compatible projection or create a deterministic one.

    Older installations may have no file, or may contain a raw ``state_dict``;
    both cases are accepted. Incompatible files are replaced rather than
    applied silently.
    """
    layer = torch.nn.Linear(source_dim, target_dim, bias=False, device=device, dtype=dtype)
    loaded = False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else None
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        weight = state.get("weight") if isinstance(state, dict) else None
        if (
            isinstance(weight, torch.Tensor)
            and tuple(weight.shape) == (target_dim, source_dim)
            and (not meta or int(meta.get("version", PROJECTION_VERSION)) == PROJECTION_VERSION)
        ):
            layer.load_state_dict({"weight": weight.to(dtype=dtype)})
            loaded = True
    except (OSError, RuntimeError, ValueError, TypeError):
        loaded = False

    if not loaded:
        layer.weight.data.copy_(projection_matrix(source_dim, target_dim, dtype=dtype))
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(
                {
                    "meta": {
                        "version": PROJECTION_VERSION,
                        "source_dim": source_dim,
                        "target_dim": target_dim,
                    },
                    "state_dict": layer.state_dict(),
                },
                path,
            )
        except OSError as exc:
            # Persistence is best effort; the deterministic in-memory matrix
            # still keeps the current process coherent.
            logging.getLogger("swa.projections").warning(
                "projection artifact unavailable at %s: %s", path, exc
            )
    return layer.to(device=device, dtype=dtype)
