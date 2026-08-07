"""Planner — Planejamento multi-step com memória episódica."""

from cortex.obelisk.dreamer.simulator import Simulator
from cortex.obelisk.dreamer.world_model import WorldModel


def _compute_outcome_quality(episodes: list[dict]) -> dict:
    """Avalia outcomes históricos para calibrar risco."""
    if not episodes:
        return {}
    outcomes = {}
    for ep in episodes:
        plan = ep.get("plan", {})
        action = plan.get("action")
        if action is None:
            continue
        if action not in outcomes:
            outcomes[action] = {"count": 0, "transitions": 0}
        outcomes[action]["count"] += 1
        tb = ep.get("tensor_before", [])
        ta = ep.get("tensor_after", [])
        if tb and ta and len(tb) >= 6 and len(ta) >= 6:
            outcomes[action]["transitions"] += 1
    return outcomes


class Planner:
    def __init__(self, world_model: WorldModel):
        self.sim = Simulator(world_model)
        self._last_plan = None

    def plan(self, tensor: list, immune_label: str, immune_action: str | None, episodes: list | None = None) -> dict:
        candidates = [immune_action, None]
        if immune_action == "rollback":
            candidates.append("cool")

        outcomes = _compute_outcome_quality(episodes or [])

        results = self.sim.compare_actions(tensor, candidates)

        for r in results:
            a = r["action"]
            if a and a in outcomes:
                pass

        best = results[0]
        inhibit = False

        if best["action"] != immune_action:
            inhibit = True

        if best["risk"] > 0.8 and best["confidence"] < 0.3:
            inhibit = True

        no_action_risk = next(
            (r["risk"] for r in results if r["action"] is None), 0.5
        )
        action_risk = next(
            (r["risk"] for r in results if r["action"] == immune_action), 0.5
        )

        if no_action_risk < action_risk * 0.7:
            inhibit = True

        if immune_action and outcomes.get(immune_action, {}).get("count", 0) > 3:
            avg_outcome = 1.0
            inhibit = avg_outcome < 0.3

        final_action = immune_action if not inhibit else None

        no_risk = next(
            (r["risk"] for r in results if r["action"] is None), 0.5
        )
        sim_risk = next(
            (r["risk"] for r in results if r["action"] == final_action), 0.5
        )

        self._last_plan = {
            "action": final_action,
            "inhibit": inhibit,
            "rationale": self._build_rationale(
                final_action, inhibit, immune_action, results
            ),
            "simulations": results,
            "no_action_risk": no_risk,
            "simulated_risk": sim_risk,
            "timestamp": __import__("time").time(),
        }
        return self._last_plan

    def _build_rationale(self, action: str | None, inhibit: bool, immune_action: str | None, results: list) -> str:
        parts = []
        if inhibit:
            parts.append(
                f"INIBIDO: sistema imune pediu '{immune_action}' "
                f"mas simulacao mostrou risco menor sem acao"
            )
        for r in results[:3]:
            a = r["action"] if r["action"] else "nenhuma"
            parts.append(
                f"  {a}: risco={r['risk']:.2f}, confianca={r['confidence']:.2f}"
            )
        return "\n".join(parts)
