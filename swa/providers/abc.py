from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol

import torch


class StateMetadata(Protocol):
    offset: int
    size: int
    hash: str
    timestamp: float


class AbstractStateProvider(ABC):
    """Interface for hardware-backed or RAM-backed state persistence.

    Implementations provide multiple named state slots (reality, prediction,
    shadow_bias, recursion, dreamwords, navigation_map) that can be read and
    written as torch tensors.
    """

    @abstractmethod
    def write_state(self, state_name: str, data: torch.Tensor) -> bool:
        """Persist *data* into slot *state_name*."""
        ...

    @abstractmethod
    def read_state(self, state_name: str) -> Optional[torch.Tensor]:
        """Load the tensor stored in slot *state_name*, or ``None``."""
        ...

    @abstractmethod
    def read_states(self, state_names: list[str]) -> Dict[str, Optional[torch.Tensor]]:
        """Read multiple slots in a single call (batched where supported)."""
        ...

    @abstractmethod
    def available_slots(self) -> frozenset[str]:
        """Return the set of slot names this provider supports."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all underlying resources."""
        ...

    def __enter__(self) -> AbstractStateProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
