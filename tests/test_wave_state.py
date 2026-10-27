from __future__ import annotations

from aeonc import AeonCore
import pytest
from swa.reality_store import RealityStore
from swa.wave_state import (
    ConversationTrajectory,
    WaveState,
    classify_phase,
    modulate_budget,
    modulate_will,
    normalized_cosine_distance,
)
from swa.wave import decide_wave, FissionModeBandit


def test_normalized_cosine_distance():
    assert normalized_cosine_distance([1, 0], [1, 0]) == 0.0
    assert normalized_cosine_distance([1, 0], [0, 1]) == 0.5
    assert normalized_cosine_distance([1, 0], [-1, 0]) == 1.0


def test_wave_phases_and_properties():
    assert classify_phase(progress=0.0) == "inicio"
    assert classify_phase(progress=0.5) == "meio"
    assert classify_phase(progress=1.0) == "fim"
    assert classify_phase(progress=0.5, distance=0.8) == "deriva"

    state = WaveState(
        "conversation",
        1,
        parent_wave_id=0,
        budget=100,
        generated_tokens=50,
        start_vector=[1, 0],
        end_vector=[1, 0],
    )
    assert state.phase == "meio"
    assert state.convergence == 1.0
    assert state.drift == 0.0
    assert state.has_parent is True
    assert state.is_terminal is False
    assert state.next_budget >= 1


def test_conversation_trajectory_validates_parent_relation():
    first = WaveState("conversation", 0, budget=100, generated_tokens=20)
    second = WaveState(
        "conversation",
        1,
        parent_wave_id=10,
        budget=100,
        generated_tokens=50,
    )
    trajectory = ConversationTrajectory("conversation", [first, second])
    assert trajectory.current is second
    assert trajectory.property_snapshot()["wave_index"] == 1

    with pytest.raises(ValueError, match="contiguous"):
        trajectory.append(WaveState("conversation", 3, parent_wave_id=10))


def test_budget_and_will_helpers_are_bounded():
    assert 64 <= modulate_budget(512, will=1.0, maximum=1024) <= 1024
    assert 64 <= modulate_budget(512, will=0.0, drift=1.0, minimum=64) <= 1024
    assert 0.15 <= modulate_will(1.5, convergence=0.0, drift=1.0, minimum=0.15) <= 1.0

    state = WaveState("conversation", 0, budget=512, semantic_will=0.8)
    assert 64 <= state.modulated_budget(minimum=64, maximum=1024) <= 1024
    assert 0.15 <= state.modulated_will(minimum=0.15) <= 1.0


def test_wave_decision_closes_loop_with_semantic_drift():
    decision = decide_wave(
        generated_tokens=100,
        budget=100,
        closed=False,
        fission_count=0,
        max_fission=3,
        matrixial_force=1.0,
        residual_energy=200.0,
        semantic_drift=0.9,
        semantic_convergence=0.1,
    )
    assert decision.should_fission is False
    assert decision.reason == "semantic_drift"

    decision = decide_wave(
        generated_tokens=100,
        budget=100,
        closed=False,
        fission_count=0,
        max_fission=3,
        matrixial_force=1.0,
        residual_energy=200.0,
        semantic_drift=0.1,
        semantic_convergence=0.9,
    )
    assert decision.should_fission is True


def test_micro_fission_is_bounded_and_has_cooldown():
    state = WaveState(
        "conversation",
        0,
        budget=10,
        semantic_will=0.8,
        start_vector=[1, 0],
    )
    assert state.observe_token([1, 0], token_count=1) is False
    assert state.observe_token([-1, 0], token_count=1, cooldown=2) is True
    assert state.micro_fission_count == 1
    assert state.observe_token([-1, 0], token_count=1, cooldown=2) is False


# ══════════ OCTOPUS v2: bandit de modo de fissão ══════════

def test_fission_bandit_explores_then_learns_and_resets():
    from swa.wave import FOLLOW, CONTINUAR, DIVERGIR

    bandit = FissionModeBandit()
    arms = set()
    for i in range(6):
        arms.add(bandit.choose(fission_count=1, max_fission=3, semantic_will=0.9,
                               emotion_load=0.5, intent_pull=0.8, repio_precision=0.7))
        assert bandit.last_arm() in (FOLLOW, CONTINUAR, DIVERGIR)
        bandit.observe(0.5 + 0.1 * i)
    # exploração inicial cobre os três braços
    assert arms == {FOLLOW, CONTINUAR, DIVERGIR}
    stats = bandit.stats()
    assert stats, "bandit deveria ter aprendido algo"
    assert all(v["n"] >= 1 for v in stats.values())
    # reset limpa o _last sem perder as estatísticas
    bandit.reset()
    assert bandit.last_arm() is None
    assert bandit.stats(), "reset não deve apagar o aprendizado"


def test_bandit_fecha_no_limite_de_fissao():
    bandit = FissionModeBandit()
    arm = bandit.choose(fission_count=3, max_fission=3, semantic_will=0.9)
    assert arm == "FECHAR"


def test_octopus_governa_modo_mas_nao_timing():
    """Clássico decide SE; o bandit decide COMO (nunca força fissão negada)."""
    # a) clássico aprova (will alta + energia) -> bandit escolhe o modo
    bandit = FissionModeBandit()
    decision = decide_wave(
        generated_tokens=120, budget=128, closed=True, fission_count=1,
        max_fission=3, semantic_will=0.9, residual_energy=150.0,
        intent_pull=0.8, repio_precision=0.7, bandit=bandit,
    )
    assert decision.should_fission is True
    assert decision.mode in ("FOLLOW", "CONTINUAR", "DIVERGIR")
    assert decision.reason.startswith("octopus::")

    # b) clássico nega (will baixa) -> bandit não é consultado nem creditado
    bandit.reset()
    decision = decide_wave(
        generated_tokens=30, budget=64, closed=True, fission_count=0,
        max_fission=3, semantic_will=0.3, residual_energy=150.0,
        intent_pull=0.3, repio_precision=0.3, bandit=bandit,
    )
    assert decision.should_fission is False
    assert decision.mode == "FECHAR"
    assert bandit.last_arm() is None


def test_octopus_respeita_limite_de_fissao():
    bandit = FissionModeBandit()
    decision = decide_wave(
        generated_tokens=120, budget=128, closed=False, fission_count=3,
        max_fission=3, semantic_will=0.9, intent_pull=0.8, repio_precision=0.7,
        bandit=bandit,
    )
    assert decision.should_fission is False
    assert decision.mode == "FECHAR"
    assert bandit.last_arm() is None


def test_reality_store_no_db_fallback_and_parent_method(tmp_path):
    store = RealityStore(str(tmp_path), db_uri="dbname=missing")
    event_hash = store.append_event("wave_state", {"phase": "inicio"}, wave=0)
    tensor_hash = store.append_tensor([1.0, 0.0], wave=0)
    assert event_hash and tensor_hash
    assert store.append_wave_state("conversation", 0, phase="inicio") is None

    class FakeStore:
        last_event_id = 17
        last_tensor_id = 23

        def append_event(self, *args, **kwargs):
            return "event-hash"

        def append_tensor(self, *args, **kwargs):
            return "tensor-hash"

        def append_wave_state(self, conversation_key, wave_index, **kwargs):
            assert conversation_key == "conversation"
            assert kwargs["parent_wave_id"] == 4
            return 5

    core = object.__new__(AeonCore)
    core.reality_store = FakeStore()
    state = WaveState(
        "conversation",
        1,
        parent_wave_id=4,
        budget=100,
        generated_tokens=50,
        start_vector=[1, 0],
        end_vector=[1, 0],
    )
    assert core._persist_wave_state(state, reason="budget_not_exhausted") == 5
