"""Versioned identity and semantic control primitives for observable contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Optional

import blake3


def _vector(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1e-12:
        raise ValueError("vector norm must be positive")
    return [value / norm for value in result]


@dataclass(frozen=True)
class ContextIdentity:
    path: str
    content_hash: str
    content_type: str = "text/plain"
    version: int = 1

    @classmethod
    def from_content(cls, path: str, content: bytes, content_type: str = "text/plain"):
        return cls(path, blake3.blake3(content).hexdigest(), content_type)


@dataclass(frozen=True)
class ContextEmbedding:
    context_hash: str
    model_name: str
    values: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(_vector(self.values)))

    @property
    def dimensions(self) -> int:
        return len(self.values)


@dataclass
class GravityField:
    conversation_key: str
    context_hashes: tuple[str, ...]
    objective_vector: tuple[float, ...]
    strength: float = 1.0
    aligned_tokens: list[int] = field(default_factory=list)
    opposed_tokens: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.objective_vector = tuple(_vector(self.objective_vector))
        self.strength = max(0.0, min(2.0, float(self.strength)))

    def metrics(self) -> dict[str, Any]:
        return {
            "context_hashes": list(self.context_hashes),
            "dimensions": len(self.objective_vector),
            "strength": self.strength,
            "aligned_count": len(self.aligned_tokens),
            "opposed_count": len(self.opposed_tokens),
        }


@dataclass(frozen=True)
class ContextObservation:
    context_hash: str
    conversation_key: str
    wave_index: int
    phase: str
    semantic_distance: float
    convergence: float
    decision: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("semantic_distance", "convergence"):
            value = max(0.0, min(1.0, float(getattr(self, name))))
            object.__setattr__(self, name, value)
        if self.wave_index < 0:
            raise ValueError("wave_index must be non-negative")


@dataclass(frozen=True)
class ContextRelation:
    source_hash: str
    target_hash: str
    relation_type: str
    similarity: float
    weight: float = 1.0

    def __post_init__(self):
        if self.source_hash == self.target_hash:
            raise ValueError("context cannot relate to itself")
        object.__setattr__(self, "similarity", max(0.0, min(1.0, float(self.similarity))))


def context_fingerprint(identity: ContextIdentity) -> str:
    payload = json.dumps(identity.__dict__, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
