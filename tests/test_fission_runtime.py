from __future__ import annotations

import math

from swa.fission_runtime import (
    StagnationDetector,
    WaveObservation,
    angle_rad,
    angular_novelty,
    angular_velocity_rad_per_wave,
    cosine_similarity,
    normalize_fragment,
    segment_waves,
)


def test_geometry_is_finite_and_dimension_safe():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 0]) is None
    assert cosine_similarity([1], [1, 0]) is None
    assert angle_rad([1, 0], [0, 1]) == pytest.approx(math.pi / 2)
    assert angular_velocity_rad_per_wave([1, 0], [0, 1], wave_delta=2) == pytest.approx(math.pi / 4)
    assert angular_novelty([1, 0], [1, 0]) == 0.0


def test_fragment_normalization_drops_control_noise_but_preserves_text():
    assert normalize_fragment("\n\n") == ""
    assert normalize_fragment("A") == ""
    assert normalize_fragment("\nOlá <end_of_turn> mundo") == "Olá mundo"
    waves = segment_waves("\n\nPrimeiro.\n\nSegundo!\n\n")
    assert [wave.text for wave in waves] == ["Primeiro.", "Segundo!"]
    assert [wave.wave_index for wave in waves] == [0, 1]


def test_stagnation_requires_accumulated_evidence_and_token_growth():
    detector = StagnationDetector(min_confirmations=3)
    observations = [
        WaveObservation(i, "a repetição semântica continua", [1.0, 0.0], 10 + i)
        for i in range(4)
    ]
    assert not detector.observe(observations[0]).confirmed
    assert not detector.observe(observations[1]).confirmed
    assert not detector.observe(observations[2]).confirmed
    assert detector.observe(observations[3]).confirmed

    detector.reset()
    assert detector.observe(WaveObservation(0, "\n\n", [1.0, 0.0], 1)).ignored


def test_reality_store_events_are_blake3_chained(tmp_path):
    from swa.reality_store import RealityStore

    store = RealityStore(str(tmp_path), db_uri="dbname=missing")
    first = store.append_event("wave", {"text": "first"})
    second = store.append_event("wave", {"text": "second"})
    assert len(first) == 64
    assert len(second) == 64
    assert first != second
    assert store._last_event_hash == second


# Kept local to avoid a dependency on pytest in the runtime module itself.
import pytest
