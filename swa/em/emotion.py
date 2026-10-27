"""
EM Emotional Mapping — Field States to Cognitive Emotions

Maps electromagnetic field dynamics to emotional states:
  - Energy level → arousal (calm ↔ excited)
  - Coherence → clarity (confused ↔ clear)
  - Stability → confidence (anxious ↔ confident)
  - Wave count → processing load (idle ↔ overwhelmed)
  - Shield activity → defensiveness (open ↔ guarded)
  - Domain structure → focus (scattered ↔ concentrated)

The emotion is not a label — it's a continuous vector in emotion space
that can be queried, tracked, and used to modulate LLM behavior.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmotionState:
    """Continuous emotion vector."""
    arousal: float = 0.5       # 0=calm, 1=excited
    valence: float = 0.5       # 0=negative, 1=positive
    clarity: float = 0.5       # 0=confused, 1=clear
    confidence: float = 0.5    # 0=anxious, 1=confident
    focus: float = 0.5         # 0=scattered, 1=concentrated
    openness: float = 0.5      # 0=guarded, 1=open
    load: float = 0.5          # 0=idle, 1=overwhelmed
    timestamp: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to 7-dim emotion vector."""
        return np.array([
            self.arousal, self.valence, self.clarity,
            self.confidence, self.focus, self.openness, self.load
        ], dtype=np.float32)

    def dominant(self) -> str:
        """Get the dominant emotion label."""
        v = self.to_vector()
        labels = ["arousal", "valence", "clarity", "confidence",
                  "focus", "openness", "load"]
        idx = int(np.argmax(v))
        return labels[idx]

    def to_label(self) -> str:
        """Map to a human-readable emotion label."""
        a, v, c, co, f, o, l = self.to_vector()

        if o < 0.3 and co < 0.3:
            return "defensive"
        if l > 0.7:
            return "overwhelmed"
        if a > 0.7 and v > 0.6:
            return "inspired"
        if a > 0.7 and v < 0.4:
            return "anxious"
        if c > 0.7 and co > 0.7:
            return "confident"
        if c > 0.7 and f > 0.7:
            return "focused"
        if c < 0.3 and f < 0.3:
            return "confused"
        if a < 0.3 and v > 0.5:
            return "serene"
        if a < 0.3 and v < 0.5:
            return "melancholic"
        if f > 0.7 and o > 0.7:
            return "curious"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "clarity": round(self.clarity, 3),
            "confidence": round(self.confidence, 3),
            "focus": round(self.focus, 3),
            "openness": round(self.openness, 3),
            "load": round(self.load, 3),
            "label": self.to_label(),
            "dominant": self.dominant(),
        }


class EmotionMapper:
    """
    Maps EM field state to emotion space.

    Uses a simple heuristic mapping:
      field_energy → arousal
      coherence → clarity
      stability → confidence
      wave_count → load
      shield_activity → inverse of openness
      domain_count → focus (fewer = more focused)
    """

    def __init__(self):
        self._history: list[EmotionState] = []
        self._max_history = 256
        self._ema = EmotionState()
        self._ema_alpha = 0.2  # smoothing

    def map_from_field(
        self,
        energy: float,
        coherence: float,
        stability: float,
        wave_count: int,
        shield_responses: int,
        domain_count: int,
        energy_delta: float = 0.0,
    ) -> EmotionState:
        """Map EM field metrics to emotion state."""

        # Arousal: energy level + rate of change
        arousal = np.clip(energy * 10 + abs(energy_delta) * 50, 0, 1)

        # Valence: coherence + stability (organized = positive)
        valence = np.clip(coherence * 0.6 + stability * 0.4, 0, 1)

        # Clarity: pure coherence
        clarity = np.clip(coherence * 2, 0, 1)

        # Confidence: pure stability
        confidence = np.clip(stability * 1.5, 0, 1)

        # Focus: fewer domains = more focused (1 domain = perfect focus)
        if domain_count <= 1:
            focus = 1.0
        elif domain_count <= 3:
            focus = 0.7
        elif domain_count <= 6:
            focus = 0.4
        else:
            focus = 0.2

        # Openness: inverse of shield activity
        openness = np.clip(1.0 - shield_responses * 0.05, 0, 1)

        # Load: wave count
        load = np.clip(wave_count / 15.0, 0, 1)

        raw = EmotionState(
            arousal=float(arousal),
            valence=float(valence),
            clarity=float(clarity),
            confidence=float(confidence),
            focus=float(focus),
            openness=float(openness),
            load=float(load),
            timestamp=time.time(),
        )

        # EMA smoothing
        self._ema.arousal = (1 - self._ema_alpha) * self._ema.arousal + self._ema_alpha * raw.arousal
        self._ema.valence = (1 - self._ema_alpha) * self._ema.valence + self._ema_alpha * raw.valence
        self._ema.clarity = (1 - self._ema_alpha) * self._ema.clarity + self._ema_alpha * raw.clarity
        self._ema.confidence = (1 - self._ema_alpha) * self._ema.confidence + self._ema_alpha * raw.confidence
        self._ema.focus = (1 - self._ema_alpha) * self._ema.focus + self._ema_alpha * raw.focus
        self._ema.openness = (1 - self._ema_alpha) * self._ema.openness + self._ema_alpha * raw.openness
        self._ema.load = (1 - self._ema_alpha) * self._ema.load + self._ema_alpha * raw.load
        self._ema.timestamp = time.time()

        self._history.append(raw)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return raw

    def current(self) -> EmotionState:
        """Get current smoothed emotion state."""
        return self._ema

    def history_summary(self) -> dict:
        """Get summary of emotion history."""
        if not self._history:
            return {"samples": 0}

        labels = [h.to_label() for h in self._history]
        from collections import Counter
        counts = Counter(labels)

        return {
            "samples": len(self._history),
            "dominant_label": counts.most_common(1)[0][0] if counts else "neutral",
            "distribution": dict(counts.most_common(10)),
            "recent_labels": labels[-5:],
        }
