"""Abstract interfaces for the OODA cognitive cycle components.

Breaks the circular dependency chain::

    TokenizerManager → GemmaGenerator → LogicalEngine → Vectorizer

Each component now depends only on these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Protocol

import torch


# ── Tokenizer ─────────────────────────────────────────────────────────────

class AbstractTokenizer(ABC):
    """Interface for tokenizer operations needed by the rest of the system."""

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ...

    @abstractmethod
    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        ...


# ── Generator (Gemma) ─────────────────────────────────────────────────────

class AbstractGenerator(ABC):
    """Interface for the language model generator."""

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Embedding dimension."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def stream_with_prefix(
        self,
        v_prefix: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        telemetry: Optional[Dict[str, float]] = None,
        logit_bias: Optional[torch.Tensor] = None,
        steer_vector: Optional[torch.Tensor] = None,
        target_layer: int = -1,
    ) -> Generator[str, None, None]:
        ...

    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...


# ── Vectorizer ────────────────────────────────────────────────────────────

class AbstractVectorizer(ABC):
    """Interface for the embedding/vectorizer model."""

    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor:
        ...

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


# ── Adapter ───────────────────────────────────────────────────────────────

class AbstractSWAAdapter(ABC):
    """Interface for the Semantic Warp Adapter layers."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def to(self, device: torch.device, dtype: torch.dtype) -> AbstractSWAAdapter:
        ...


# ── State ─────────────────────────────────────────────────────────────────

class AbstractState(ABC):
    """Interface for the cognitive state (vectors, obstacles, bias)."""

    @property
    @abstractmethod
    def vectors(self) -> Dict[str, float]:
        ...

    @vectors.setter
    def vectors(self, value: Dict[str, float]) -> None:  # noqa: F811
        ...

    @property
    @abstractmethod
    def obstacles(self) -> List[Dict[str, Any]]:
        ...

    @property
    @abstractmethod
    def organic_entropy(self) -> float:
        ...

    @abstractmethod
    def get_embedding_steering(self) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def get_logit_steering(self) -> Optional[torch.Tensor]:
        ...


# ── Provider container (simple DI) ────────────────────────────────────────

class SWAContainer:
    """Lightweight dependency injection container for SWA components."""

    def __init__(self) -> None:
        self.tokenizer: Optional[AbstractTokenizer] = None
        self.generator: Optional[AbstractGenerator] = None
        self.vectorizer: Optional[AbstractVectorizer] = None
        self.adapter_local: Optional[AbstractSWAAdapter] = None
        self.adapter_macro: Optional[AbstractSWAAdapter] = None
        self.state: Optional[AbstractState] = None
