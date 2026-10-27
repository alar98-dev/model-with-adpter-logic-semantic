"""Tábua dos Reinados — cada ciclo um passo (cycle_history / eco / âncora)."""

import sys

sys.path.insert(0, ".")

from swa.metareality_organ import MetarealityOrgan


class StubReality:
    def __init__(self):
        self._rows = [
            {
                "id": 2,
                "conversation_key": "k1",
                "wave_index": 1,
                "phase": "fim",
                "created_at": "2026-09-08 15:35:12",
                "metrics": {"reality_index": 0.842, "emotion": "neutral"},
            },
            {
                "id": 1,
                "conversation_key": "k1",
                "wave_index": 0,
                "phase": "fim",
                "created_at": "2026-09-08 15:34:26",
                "metrics": {"drift": 0.12, "decision": "octopus::follow", "budget": 645},
            },
        ]

    def wave_series(self, *, conversation_key=None, limit=40):
        return self._rows

    def latest_wave_score(self, *, conversation_key=None, metric_key="reality_index"):
        vals = [r["metrics"].get(metric_key) for r in self._rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return vals[-1] if vals else None


class StubCore:
    reality_store = StubReality()


def test_cycle_history_shape():
    org = MetarealityOrgan(StubCore())
    h = org.cycle_history()
    assert h["count"] == 2
    assert h["echo"] == 0.842
    assert h["totals"]["mean_reality_index"] == 0.842
    assert h["totals"]["mean_drift"] == 0.12
    assert h["totals"]["decision_entropy"] == 0.0
    assert abs(h["anchor"] - 1.24) < 0.01


def test_last_reality_index():
    org = MetarealityOrgan(StubCore())
    assert org.last_reality_index() == 0.842
    assert org.last_reality_index(conversation_key="k1") == 0.842


def test_cycle_weight_uses_drift():
    org = MetarealityOrgan(StubCore())
    assert 0.6 <= org.cycle_weight() <= 1.5


def test_context_carries_echo_and_anchor():
    org = MetarealityOrgan(StubCore())
    ctx = org.context("teste")
    assert 'cycle_echo' in ctx
    assert '"anchor"' in ctx


if __name__ == "__main__":
    for name in (
        test_cycle_history_shape,
        test_last_reality_index,
        test_cycle_weight_uses_drift,
        test_context_carries_echo_and_anchor,
    ):
        name()
        print(f"OK {name.__name__}")
    print("ALL OK")