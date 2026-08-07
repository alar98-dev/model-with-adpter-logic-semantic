from __future__ import annotations

from typing import Dict, Optional

import torch

from swa.providers.abc import AbstractStateProvider

_VECTOR_DIM = 2560
_BIAS_DIM = 2560


class RAMStateProvider(AbstractStateProvider):
    """Volatile in-RAM state provider.

    All state slots live in a plain dictionary.  Useful as a safe fallback
    when no block device is available, and as a fast backend for testing.
    """

    def __init__(self, vector_dim: int = _VECTOR_DIM, bias_dim: int = _BIAS_DIM) -> None:
        self._vector_dim = vector_dim
        self._bias_dim = bias_dim
        self._slots: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_state(self, state_name: str, data: torch.Tensor) -> bool:
        self._slots[state_name] = data.detach().cpu().clone()
        return True

    def read_state(self, state_name: str) -> Optional[torch.Tensor]:
        t = self._slots.get(state_name)
        return t.clone() if t is not None else None

    def read_states(self, state_names: list[str]) -> Dict[str, Optional[torch.Tensor]]:
        return {name: self.read_state(name) for name in state_names}

    def available_slots(self) -> frozenset[str]:
        return frozenset(self._slots)

    def close(self) -> None:
        self._slots.clear()
