"""Small, model-agnostic state machine for dynamic generation waves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional

from swa.fission_runtime import angle_rad, angular_velocity_rad_per_wave, is_meaningful_fragment


PHASES = ("inicio", "meio", "fim", "deriva")


def _flatten_finite(value: Any) -> list[float]:
    """Convert common tensor/array values without requiring either dependency."""
    if value is None:
        return []
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    reshape = getattr(candidate, "reshape", None)
    if callable(reshape):
        candidate = reshape(-1)
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()

    def walk(item: Any):
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from walk(child)
        else:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                yield number

    return list(walk(candidate))


def normalized_cosine_distance(left: Any, right: Any) -> float:
    """Return cosine distance normalized to ``[0, 1]``.

    Identical vectors have distance ``0``, orthogonal vectors ``0.5`` and
    opposite vectors ``1``. Empty or zero-norm vectors are treated as unknown
    (distance ``0.5``), rather than falsely indicating convergence.
    """
    first = _flatten_finite(left)
    second = _flatten_finite(right)
    if not first or not second:
        return 0.5
    size = min(len(first), len(second))
    if size == 0:
        return 0.5
    first = first[:size]
    second = second[:size]
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return 0.5
    cosine = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return max(0.0, min(1.0, (1.0 - cosine) / 2.0))


# Short alias for callers that already use the conventional name.
cosine_distance = normalized_cosine_distance


def classify_phase(
    progress: float,
    distance: float = 0.0,
    closed: bool = False,
    drift_threshold: float = 0.65,
) -> str:
    """Classify a wave from progress, closure and vector drift."""
    progress = max(0.0, min(1.0, float(progress)))
    distance = max(0.0, min(1.0, float(distance)))
    if closed or progress >= 0.98:
        return "fim"
    if distance >= drift_threshold:
        return "deriva"
    if progress <= 0.20:
        return "inicio"
    return "meio"


def modulate_budget(
    base_budget: int,
    *,
    will: float,
    convergence: float = 0.5,
    drift: float = 0.0,
    minimum: int = 1,
    maximum: int = 4096,
) -> int:
    """Scale a token budget while keeping it within explicit bounds."""
    lower = max(1, int(minimum))
    upper = max(lower, int(maximum))
    base = max(lower, int(base_budget))
    will = max(0.0, min(1.0, float(will)))
    convergence = max(0.0, min(1.0, float(convergence)))
    drift = max(0.0, min(1.0, float(drift)))
    factor = 0.70 + (0.55 * will) + (0.20 * convergence) - (0.25 * drift)
    return max(lower, min(upper, int(round(base * factor))))


def modulate_will(
    base_will: float,
    *,
    convergence: float = 0.5,
    drift: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Adjust semantic will using convergence and drift, with bounded output."""
    lower = max(0.0, min(1.0, float(minimum)))
    upper = max(lower, min(1.0, float(maximum)))
    base = max(0.0, min(1.0, float(base_will)))
    convergence = max(0.0, min(1.0, float(convergence)))
    drift = max(0.0, min(1.0, float(drift)))
    value = base * (0.75 + 0.35 * convergence) - (0.20 * drift)
    return max(lower, min(upper, value))


@dataclass
class WaveState:
    """Mutable, serializable state for one wave in a conversation."""

    conversation_key: str
    wave_index: int
    parent_wave_id: Optional[int] = None
    budget: int = 0
    semantic_will: float = 0.5
    generated_tokens: int = 0
    closed: bool = False
    start_vector: Any = None
    end_vector: Any = None
    micro_steps: int = 0
    micro_fission_count: int = 0
    last_micro_step: int = -10_000
    persisted_id: Optional[int] = None

    @property
    def progress(self) -> float:
        return max(0.0, min(1.0, self.generated_tokens / max(1, self.budget)))

    @property
    def will(self) -> float:
        return max(0.0, min(1.0, float(self.semantic_will)))

    @property
    def drift(self) -> float:
        return normalized_cosine_distance(self.start_vector, self.end_vector)

    @property
    def distance(self) -> float:
        return self.drift

    @property
    def convergence(self) -> float:
        return 1.0 - self.drift

    @property
    def cosine_similarity(self) -> Optional[float]:
        """cos entre o vetor inicial e o final da onda (qualidade da fissão).

        Retorna None quando não há vetores reais para medir (start/end vazios
        ou norma nula), para não confundir "desconhecido" com "ortogonal".
        """
        first = _flatten_finite(self.start_vector)
        second = _flatten_finite(self.end_vector)
        if not first or not second:
            return None
        size = min(len(first), len(second))
        if size == 0:
            return None
        first = first[:size]
        second = second[:size]
        first_norm = math.sqrt(sum(v * v for v in first))
        second_norm = math.sqrt(sum(v * v for v in second))
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            return None
        cosine = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
        return float(max(-1.0, min(1.0, cosine)))

    @property
    def angle_rad(self) -> Optional[float]:
        """Angular displacement between the start and end state of this wave."""
        return angle_rad(self.start_vector, self.end_vector)

    @property
    def angular_velocity_rad_per_wave(self) -> Optional[float]:
        """Per-wave velocity for a wave-local transition."""
        return angular_velocity_rad_per_wave(self.start_vector, self.end_vector)

    @property
    def phase(self) -> str:
        return classify_phase(
            progress=self.progress,
            distance=self.drift,
            closed=self.closed,
        )

    @property
    def is_terminal(self) -> bool:
        return self.phase == "fim"

    @property
    def has_parent(self) -> bool:
        return self.parent_wave_id is not None

    @property
    def next_budget(self) -> int:
        return self.modulated_budget(minimum=1, maximum=4096)

    @property
    def next_will(self) -> float:
        return self.modulated_will()

    def observe_token(
        self,
        vector: Any = None,
        *,
        token_count: int = 1,
        text: Optional[str] = None,
        drift_threshold: float = 0.65,
        cooldown: int = 8,
        max_micro_fissions: int = 8,
    ) -> bool:
        """Observe token progress and trigger a bounded internal correction."""
        if text is not None and not is_meaningful_fragment(text):
            return False
        observed_tokens = max(1, int(token_count))
        self.micro_steps += observed_tokens
        self.generated_tokens += observed_tokens
        if vector is not None:
            self.end_vector = vector
        if (
            self.micro_steps - self.last_micro_step < max(1, int(cooldown))
            or self.micro_fission_count >= max(0, int(max_micro_fissions))
            or self.progress <= 0.05
        ):
            return False
        if self.drift < max(0.0, min(1.0, float(drift_threshold))):
            return False
        self.micro_fission_count += 1
        self.last_micro_step = self.micro_steps
        self.semantic_will = self.next_will
        return True

    @staticmethod
    def classify_phase(
        progress: float, distance: float = 0.0, closed: bool = False
    ) -> str:
        return classify_phase(
            progress=progress, distance=distance, closed=closed
        )

    def modulated_budget(self, *, minimum: int = 1, maximum: int = 4096) -> int:
        return modulate_budget(
            self.budget,
            will=self.semantic_will,
            convergence=self.convergence,
            drift=self.drift,
            minimum=minimum,
            maximum=maximum,
        )

    def modulated_will(self, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return modulate_will(
            self.semantic_will,
            convergence=self.convergence,
            drift=self.drift,
            minimum=minimum,
            maximum=maximum,
        )

    def metrics(self) -> dict[str, Any]:
        """Return persistence-safe metrics without including generated text."""
        return {
            "wave_index": int(self.wave_index),
            "parent_wave_id": self.parent_wave_id,
            "budget": int(self.budget),
            "generated_tokens": int(self.generated_tokens),
            "progress": round(self.progress, 6),
            "semantic_will": round(max(0.0, min(1.0, self.semantic_will)), 6),
            "convergence": round(self.convergence, 6),
            "drift": round(self.drift, 6),
            "cos_sim": (
                round(self.cosine_similarity, 6)
                if self.cosine_similarity is not None
                else None
            ),
            "angle_rad": (
                round(self.angle_rad, 6) if self.angle_rad is not None else None
            ),
            "angular_velocity_rad_per_wave": (
                round(self.angular_velocity_rad_per_wave, 6)
                if self.angular_velocity_rad_per_wave is not None
                else None
            ),
            "closed": bool(self.closed),
            "phase": self.phase,
            "micro_steps": int(self.micro_steps),
            "micro_fission_count": int(self.micro_fission_count),
        }


DynamicWaveState = WaveState


class ConversationTrajectory:
    """Ordered, validated relation of waves for the new architecture."""

    def __init__(self, conversation_key: str, states: Iterable[WaveState] = ()):
        self.conversation_key = str(conversation_key)
        self._states: list[WaveState] = []
        for state in states:
            self.append(state)

    @property
    def states(self) -> tuple[WaveState, ...]:
        return tuple(self._states)

    @property
    def current(self) -> Optional[WaveState]:
        return self._states[-1] if self._states else None

    @property
    def is_complete(self) -> bool:
        return bool(self.current and self.current.is_terminal)

    def append(self, state: WaveState) -> None:
        if state.conversation_key != self.conversation_key:
            raise ValueError("wave belongs to a different conversation")
        if self._states:
            previous = self.current
            assert previous is not None
            if state.wave_index != previous.wave_index + 1:
                raise ValueError("wave indices must be contiguous")
            if state.parent_wave_id is None:
                raise ValueError("child wave must reference its parent")
            if previous.is_terminal:
                pass # removed: raise ValueError("cannot append after terminal wave")
        elif state.wave_index != 0:
            raise ValueError("trajectory must start at wave zero")
        self._states.append(state)

    def property_snapshot(self) -> dict[str, Any]:
        """Expose the current relational state without generated text."""
        current = self.current
        if current is None:
            return {
                "conversation_key": self.conversation_key,
                "wave_index": None,
                "phase": "unknown",
            }
        return {
            "conversation_key": self.conversation_key,
            "wave_index": current.wave_index,
            "parent_wave_id": current.parent_wave_id,
            "phase": current.phase,
            "semantic_distance": round(current.distance, 6),
            "convergence": round(current.convergence, 6),
            "cos_sim": (
                round(current.cosine_similarity, 6)
                if current.cosine_similarity is not None
                else None
            ),
            "budget": current.next_budget,
            "will": round(current.next_will, 6),
            "terminal": current.is_terminal,
        }

    def quality(self) -> dict[str, Any]:
        """Qualidade da fissão acumulada na sessão.

        Média de `cos_sim` sobre as ondas com vetores reais + razão de ondas
        completas (fechadas). É a métrica de deriva/gerga por conversa.
        """
        cos_values = [
            c for c in (w.cosine_similarity for w in self._states) if c is not None
        ]
        n = len(self._states)
        terminal = sum(1 for w in self._states if w.is_terminal)
        return {
            "conversation_key": self.conversation_key,
            "waves": n,
            "terminal_waves": terminal,
            "complete_ratio": round(terminal / n, 3) if n else 0.0,
            "mean_cos_sim": round(sum(cos_values) / len(cos_values), 4) if cos_values else None,
            "min_cos_sim": round(min(cos_values), 4) if cos_values else None,
            "max_cos_sim": round(max(cos_values), 4) if cos_values else None,
            "n_waves_with_vectors": len(cos_values),
        }
