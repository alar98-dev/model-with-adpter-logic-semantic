"""Gate de Vontade Autônoma — Markov+bandit decide o ponto de ruptura."""

import os
import sys
import tempfile

sys.path.insert(0, ".")

from swa.autonomous_will import AutonomousWillGate
from swa.metareality_organ import RealitySnapshot


class StubStore:
    def __init__(self):
        self.events = []

    def append_event(self, event_type, payload, *, wave=0):
        self.events.append((event_type, payload))
        return "digest"


class StubOrgan:
    def __init__(self, **kw):
        snap = RealitySnapshot()
        snap.will = kw.get("will", 0.5)
        snap.reality_index = kw.get("reality_index", 0.5)
        snap.memory_pull = kw.get("memory_pull", 0.0)
        snap.emotion_load = kw.get("emotion_load", 0.5)
        snap.metamatter_magnitude = kw.get("metamatter", 0.0)

        def integrate(*a, **k):
            return snap

        self.integrate = integrate


class StubCore:
    def __init__(self, **kw):
        self.metareality = StubOrgan(**kw.pop("organ", {}))
        self.reality_store = kw.get("store", StubStore())


def test_silence_preferred_when_calm():
    store = StubStore()
    core = StubCore(organ={"will": 0.3, "reality_index": 0.35, "metamatter": 5}, store=store)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gate = AutonomousWillGate(save_path=f.name, reality_store=store)
    d = gate.decide(core)
    assert d["arm_name"] == "SKIP", d["arm_name"]
    assert d["state_name"] == "SILENCIO"
    assert not d["should_generate"]


def test_rupture_favors_speech():
    store = StubStore()
    core = StubCore(
        organ={"will": 0.95, "reality_index": 0.9, "memory_pull": 0.6, "metamatter": 70},
        store=store,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gate = AutonomousWillGate(save_path=f.name, reality_store=store)
    d = gate.decide(core)
    assert d["state_name"] in ("IMPULSO", "RUPTURA")
    assert d["should_generate"], d["arm_name"]


def test_reward_updates_and_persists():
    store = StubStore()
    core = StubCore(
        organ={"will": 0.9, "reality_index": 0.85, "memory_pull": 0.5, "metamatter": 60},
        store=store,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gate = AutonomousWillGate(save_path=f.name, reality_store=store)
    d = gate.decide(core)
    r = gate.reward({"substance": 1.0, "reality_index": 0.88})
    assert 0.0 <= r <= 1.0
    gate.observe_decision(core, gate.status()["last_decision"], emitted=True)
    assert any(t == "will_decision" for t, _ in store.events)
    assert os.path.exists(gate.save_path)


def test_skip_reward_learns_quiet():
    store = StubStore()
    core = StubCore(organ={"will": 0.2, "reality_index": 0.3, "metamatter": 3}, store=store)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gate = AutonomousWillGate(save_path=f.name, reality_store=store)
    d = gate.decide(core)
    assert not d["should_generate"]
    r = gate.reward({"impulse": d["impulse"]})
    assert r >= 0.5  # silêncio em calma é bom -> reforça SKIP


def test_state_machine_bounds():
    store = StubStore()
    core = StubCore(organ={}, store=store)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        gate = AutonomousWillGate(save_path=f.name, reality_store=store)
    for _ in range(20):
        d = gate.decide(core)
        assert d["state"] in range(4)
        assert d["arm"] in range(3)
        gate.reward({"impulse": d["impulse"]})


if __name__ == "__main__":
    for name in (
        test_silence_preferred_when_calm,
        test_rupture_favors_speech,
        test_reward_updates_and_persists,
        test_skip_reward_learns_quiet,
        test_state_machine_bounds,
    ):
        name()
        print(f"OK {name.__name__}")
    print("ALL OK")