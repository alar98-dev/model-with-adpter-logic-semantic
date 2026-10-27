"""Observable experience state for the Aeon runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _bounded(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass
class ExperienceState:
    """Measured perception/integration state associated with one wave."""

    conversation_key: str
    wave_index: int
    duration_ms: float = 0.0
    information_absorption: float = 0.0
    semantic_change: float = 0.0
    novelty: float = 0.0
    salience: float = 0.0
    confidence: float = 0.0
    coherence: float = 0.0
    memory_created: bool = False
    outcome: str = "unknown"
    wave_id: Optional[int] = None

    @property
    def integration_score(self) -> float:
        return _bounded(self.information_absorption) * _bounded(self.coherence)

    @property
    def meaningful_change(self) -> float:
        return _bounded(self.semantic_change) * _bounded(self.salience)

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "duration_ms": round(max(0.0, float(self.duration_ms)), 3),
            "information_absorption": round(_bounded(self.information_absorption), 6),
            "semantic_change": round(_bounded(self.semantic_change), 6),
            "novelty": round(_bounded(self.novelty), 6),
            "salience": round(_bounded(self.salience), 6),
            "confidence": round(_bounded(self.confidence), 6),
            "coherence": round(_bounded(self.coherence), 6),
            "integration_score": round(self.integration_score, 6),
            "meaningful_change": round(self.meaningful_change, 6),
            "memory_created": bool(self.memory_created),
            "outcome": str(self.outcome),
        }
