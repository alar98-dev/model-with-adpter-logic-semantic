from __future__ import annotations

import torch

from aeonc import AeonCore
from swa.db.episodic import EpisodicMemoryController
from swa.dimensions import MEMORY_DIM, MODEL_DIM, NATIVE_DIM, project_vector, to_memory
from swa.eventual_state import EventualState
from swa.historian import AsynchronousHistorian
from swa.narrative_state import NarrativeState
from swa.projections import load_or_create_projection
from swa.wave import decide_wave, generation_closed_from_result


def test_canonical_dimensions_and_round_trip():
    native = torch.arange(NATIVE_DIM, dtype=torch.float32)
    memory = to_memory(native, NATIVE_DIM)
    assert memory.shape == (MEMORY_DIM,)
    assert torch.isfinite(memory).all()

    episode = {"concept_group": "NARRATIVE_WAVE", "content": "ponte", "embedding_2560": memory.tolist()}
    encoded = EpisodicMemoryController.serialize_episode(episode)
    decoded = EpisodicMemoryController.deserialize_episode(encoded)
    assert len(decoded["embedding_2560"]) == MEMORY_DIM
    assert decoded["content"] == "ponte"


def test_versioned_projection_round_trip(tmp_path):
    path = str(tmp_path / "history_proj.pt")
    first = load_or_create_projection(NATIVE_DIM, MODEL_DIM, path)
    second = load_or_create_projection(NATIVE_DIM, MODEL_DIM, path)
    assert torch.allclose(first.weight, second.weight)


def test_narrative_state_persists_and_recovers(tmp_path):
    path = str(tmp_path / "narrative.jsonl")
    state = NarrativeState(path)
    state.record_turn(
        user="continuar",
        reply="A expedição segue.",
        wave=2,
        summary="A expedição segue.",
        entities=["expedição"],
        objectives=["chegar ao portal"],
        time_context="amanhecer",
        location="vale",
    )
    recovered = NarrativeState(path).recover()
    assert recovered["entities"] == ["expedição"]
    assert "vale" in recovered["context"]

    inferred = NarrativeState.infer_structure("A Expedição deve chegar ao Portal amanhã em Lisboa.")
    assert "Expedição" in inferred[0]
    assert inferred[1]
    assert inferred[2] == "amanhã"
    assert "Lisboa" in inferred[3]


def test_wave_decision_requires_eos_or_budget():
    assert not decide_wave(
        generated_tokens=100,
        budget=100,
        closed=True,
        fission_count=0,
        max_fission=2,
    ).should_fission


def test_eventual_state_updates_are_atomic_and_shareable():
    state = EventualState(vocab_size=32, hidden_size=MODEL_DIM)
    state.update_vectors({"forward": 0.8})
    assert state.vectors["forward"] == 0.8
    coordinate = torch.ones(MODEL_DIM)
    state.set_coordinate_vector(coordinate)
    assert torch.equal(state.get_coordinate_vector(), coordinate)


def test_historian_accepts_episodic_keywords_and_narrative_state(tmp_path):
    state = EventualState(vocab_size=32, hidden_size=MODEL_DIM)
    historian = AsynchronousHistorian(
        state,
        sleep_interval=0.01,
        narrative_state=NarrativeState(str(tmp_path / "history.jsonl")),
    )
    story = historian._compose_story()
    assert story
    historian.begin_thought()
    historian.log_token("onda")
    historian.end_thought()
    assert decide_wave(
        generated_tokens=100,
        budget=100,
        closed=False,
        fission_count=0,
        max_fission=2,
    ).should_fission


def test_padded_eos_is_not_real_generation_eos():
    class Result:
        # 511 actual steps, while the 512th slot is a pad EOS.
        scores = [object()] * 511

    sequence = torch.cat(
        [torch.tensor([[10, 11]]), torch.arange(100, 611).reshape(1, -1), torch.tensor([[2]])],
        dim=1,
    )

    # Prefix is two tokens; token 610 is the final emitted token, not padding 2.
    assert not generation_closed_from_result(
        Result(), sequence, {2}, max_new_tokens=512, prefix_length=2
    )
    decision = decide_wave(
        generated_tokens=511, budget=512, closed=False, fission_count=0, max_fission=1
    )
    assert decision.should_fission

    core = object.__new__(AeonCore)
    core._last_generation_closed = False
    core._last_generation_steps = 511
    assert core._continuation_score("prompt", "partial", 511, 512) >= 0.65
