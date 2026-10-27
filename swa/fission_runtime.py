"""Deterministic runtime primitives for semantic wave fission.

This module deliberately has no model, Redis or database dependency.  It is
the decision boundary between streaming and the side-effecting persistence
layers, so malformed micro-fragments can never influence geometric decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable, Optional


_END_MARKERS = ("<end_of_turn>", "<eos>", "<|eot_id|>")
_WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)
_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


def _values(value: Any) -> list[float]:
    if value is None:
        return []
    candidate = value
    for method in ("detach", "cpu"):
        fn = getattr(candidate, method, None)
        if callable(fn):
            candidate = fn()
    reshape = getattr(candidate, "reshape", None)
    if callable(reshape):
        candidate = reshape(-1)
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()

    def flatten(item: Any):
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from flatten(child)
            return
        try:
            number = float(item)
        except (TypeError, ValueError):
            return
        if math.isfinite(number):
            yield number

    return list(flatten(candidate))


def normalize_fragment(text: str | None) -> str:
    """Normalize a generated fragment for analytics, never for client output."""
    value = str(text or "")
    for marker in _END_MARKERS:
        value = value.replace(marker, " ")
    value = " ".join(value.split())
    # A lone control/punctuation character is not a semantic observation.
    if len(_WORD_RE.findall(value)) == 0:
        return ""
    if len(value) == 1:
        return ""
    return value


def is_meaningful_fragment(text: str | None) -> bool:
    return bool(normalize_fragment(text))


@dataclass(frozen=True)
class WaveFragment:
    wave_index: int
    text: str


def segment_waves(text: str | None, *, max_chars: int = 1200) -> list[WaveFragment]:
    """Split text at semantic boundaries and discard non-semantic fragments."""
    limit = max(1, int(max_chars))
    source = str(text or "")
    result: list[WaveFragment] = []
    for paragraph in _BOUNDARY_RE.split(source):
        paragraph = normalize_fragment(paragraph)
        if not paragraph:
            continue
        words = paragraph.split()
        current: list[str] = []
        size = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and size + extra > limit:
                result.append(WaveFragment(len(result), " ".join(current)))
                current, size = [], 0
            current.append(word)
            size += len(word) + (1 if len(current) > 1 else 0)
        if current:
            result.append(WaveFragment(len(result), " ".join(current)))
    return result


def cosine_similarity(left: Any, right: Any) -> Optional[float]:
    """Return a finite cosine in ``[-1, 1]`` or ``None`` for invalid input."""
    a, b = _values(left), _values(right)
    if not a or not b or len(a) != len(b):
        return None
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)))


def angle_rad(left: Any, right: Any) -> Optional[float]:
    similarity = cosine_similarity(left, right)
    return None if similarity is None else math.acos(similarity)


def angular_velocity_rad_per_wave(
    previous: Any, current: Any, *, wave_delta: int = 1
) -> Optional[float]:
    angle = angle_rad(previous, current)
    if angle is None or int(wave_delta) <= 0:
        return None
    return angle / float(wave_delta)


def angular_novelty(previous: Any, current: Any) -> Optional[float]:
    """Angular novelty normalized to ``[0, 1]`` (zero means no change)."""
    angle = angle_rad(previous, current)
    return None if angle is None else angle / math.pi


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _WORD_RE.findall(normalize_fragment(text))]
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def ngram_repetition(previous: str, current: str, *, n: int = 3) -> float:
    old = _ngrams(previous, n)
    new = _ngrams(current, n)
    if not old or not new:
        return 0.0
    return len(old & new) / float(len(new))


@dataclass(frozen=True)
class WaveObservation:
    wave_index: int
    text: str
    embedding: Any
    token_count: int

    @property
    def meaningful(self) -> bool:
        return is_meaningful_fragment(self.text) and self.embedding is not None


@dataclass(frozen=True)
class StagnationDecision:
    confirmed: bool
    streak: int
    ignored: bool = False
    reason: str = ""


class StagnationDetector:
    """Require independent, accumulated evidence before requesting escape."""

    def __init__(
        self,
        *,
        min_confirmations: int = 3,
        cosine_threshold: float = 0.92,
        angular_novelty_threshold: float = 0.08,
        repetition_threshold: float = 0.60,
    ):
        self.min_confirmations = max(2, int(min_confirmations))
        self.cosine_threshold = float(cosine_threshold)
        self.angular_novelty_threshold = float(angular_novelty_threshold)
        self.repetition_threshold = float(repetition_threshold)
        self._previous: Optional[WaveObservation] = None
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def observe(self, observation: WaveObservation) -> StagnationDecision:
        if not observation.meaningful:
            return StagnationDecision(False, self._streak, ignored=True, reason="non_semantic_fragment")
        previous = self._previous
        self._previous = observation
        if previous is None:
            return StagnationDecision(False, self._streak, reason="warmup")
        cosine = cosine_similarity(previous.embedding, observation.embedding)
        novelty = angular_novelty(previous.embedding, observation.embedding)
        repeated = ngram_repetition(previous.text, observation.text)
        grew = int(observation.token_count) > int(previous.token_count)
        stagnant = (
            cosine is not None
            and cosine >= self.cosine_threshold
            and novelty is not None
            and novelty <= self.angular_novelty_threshold
            and repeated >= self.repetition_threshold
            and grew
        )
        self._streak = self._streak + 1 if stagnant else 0
        confirmed = self._streak >= self.min_confirmations
        reason = "confirmed" if confirmed else ("evidence" if stagnant else "insufficient_evidence")
        return StagnationDecision(confirmed, self._streak, reason=reason)

    def reset(self) -> None:
        self._previous = None
        self._streak = 0


def anti_paraphrase_directive(chunk: str) -> str:
    clean = normalize_fragment(chunk)
    if not clean:
        return ""
    return (
        "GROUNDING: use the retrieved physical chunk verbatim as the factual anchor. "
        "Do not paraphrase, summarize, or invent details.\n"
        f"PHYSICAL_CHUNK:\n{clean}"
    )


def automatic_grounding(
    query_embedding: Any,
    fetch_chunk: Callable[[Any], Optional[str]],
) -> Optional[str]:
    """Fetch the physical chunk only after a caller has confirmed stagnation."""
    try:
        return anti_paraphrase_directive(fetch_chunk(query_embedding) or "") or None
    except Exception:
        return None
