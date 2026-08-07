"""Simulador — Projeta futuros possíveis.

Dado o tensor atual e uma ação candidata, simula:
  1. Estado imediato (curto prazo, 1 ciclo)
  2. Tendência (médio prazo, 10 ciclos)
  3. Risco de mutação (longo prazo, 100 ciclos)

Usa o world model para predizer transições.
"""

from cortex.obelisk.dreamer.world_model import WorldModel


class Simulator:
    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def simulate(self, tensor: list, action: str | None) -> dict:
        """Simula o futuro se 'action' for tomada.

        Returns:
          next_state: tensor previsto para o próximo ciclo
          trend: vetor de mudança (positivo = piora)
          risk: score de risco (0.0 = seguro, 1.0 = catastrófico)
          confidence: quão confiável é a simulação (0.0-1.0)
        """
        next_state = self.wm.predict_next(tensor, action)
        delta = [next_state[i] - tensor[i] for i in range(len(tensor))]

        risk = self._compute_risk(tensor, next_state, action)
        confidence = self._compute_confidence(tensor, action)

        return {
            "next_state": next_state,
            "delta": delta,
            "risk": risk,
            "confidence": confidence,
        }

    def compare_actions(self, tensor: list, actions: list[str | None]) -> list[dict]:
        """Simula múltiplas ações e retorna ordenado por risco."""
        results = []
        for action in actions:
            sim = self.simulate(tensor, action)
            results.append({"action": action, **sim})
        results.sort(key=lambda x: x["risk"])
        return results

    def _compute_risk(self, before: list, after: list, action: str | None) -> float:
        """Risco: 0.0 = seguro, 1.0 = catastrófico.

        Fatores de risco:
          - Delta positivo em hash_entropy (e0) = mais mutação
          - Delta positivo em hw_pressure (e2) = HW piorando
          - Delta negativo em manifold_mass (e1) = perda de memória
        """
        if len(before) < 6 or len(after) < 6:
            return 0.0
        risk = 0.0
        e0_delta = after[0] - before[0]
        e1_delta = after[1] - before[1]
        e2_delta = after[2] - before[2]
        e3_delta = after[3] - before[3]

        if e0_delta > 0:
            risk += e0_delta * 2.0
        if e1_delta < 0:
            risk += abs(e1_delta) * 1.5
        if e2_delta > 0:
            risk += e2_delta * 2.0
        if e3_delta > 0.1:
            risk += e3_delta * 1.5

        if action == "rollback":
            risk -= 0.3
        elif action == "cool":
            risk -= 0.2
        elif action == "evac":
            risk -= 0.15

        return max(0.0, min(1.0, risk))

    def _compute_confidence(self, tensor: list, action: str | None) -> float:
        """Confiança baseada em quantas transições similares existem."""
        wm = self.wm
        if not wm._transitions:
            return 0.1
        count = sum(
            1 for t in wm._transitions[-200:]
            if t["action"] == action
        )
        base = min(count / 10.0, 1.0)
        stability = wm.stability_score(tensor)
        return (base + stability) / 2.0
