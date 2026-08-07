"""World Model — Modelo interno do comportamento do sistema.

Mantém uma representação comprimida de:
  - Sequência histórica de tensores de gravidade
  - Ações tomadas e seus resultados
  - Transições de estado (tensorₜ, açãoₜ → tensorₜ₊₁)

Usado pelo simulador para prever futuros.
"""

import os
import json
import time
import glob

MEMORY_DIR = os.environ.get("OBELISK_IMMUNE_MEMORY", "/var/lib/obelisk/immune_memory")
TRANSITION_LOG = "/var/lib/obelisk/transitions.jsonl"


class WorldModel:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._transitions: list[dict] = []
        self._cache = {}

    def observe(self, tensor_before: list, action: str | None, tensor_after: list):
        """Registra uma transição de estado."""
        entry = {
            "t": time.time(),
            "state_before": tensor_before,
            "action": action,
            "state_after": tensor_after,
            "delta": [a - b for a, b in zip(tensor_after, tensor_before)],
        }
        self._transitions.append(entry)
        if len(self._transitions) > self.max_history:
            self._transitions = self._transitions[-self.max_history:]
        self._persist(entry)

    def predict_next(self, tensor: list, action: str | None) -> list[float]:
        """Estima o tensor resultante se 'action' for tomada.

        Heurística: encontra transições similares no histórico e
        calcula a média do delta.
        """
        if not self._transitions:
            return tensor

        similares = []
        for t in self._transitions:
            if t["action"] == action:
                sim = 1.0 - sum(
                    abs(a - b) for a, b in zip(tensor, t["state_before"])
                ) / max(len(tensor), 1)
                similares.append((sim, t["delta"]))
        if not similares:
            return tensor

        similares.sort(key=lambda x: -x[0])
        top = similares[:5]
        avg_delta = [0.0] * len(tensor)
        total_w = 0.0
        for sim, delta in top:
            w = max(sim, 0.0)
            for i in range(len(tensor)):
                avg_delta[i] += delta[i] * w
            total_w += w
        if total_w > 0:
            avg_delta = [d / total_w for d in avg_delta]
        return [tensor[i] + avg_delta[i] for i in range(len(tensor))]

    def stability_score(self, tensor: list) -> float:
        """Quão estável o sistema está: 1.0 = totalmente estável."""
        if not self._transitions:
            return 1.0
        recent = self._transitions[-50:]
        if not recent:
            return 1.0
        var = [0.0] * len(tensor)
        for t in recent:
            for i in range(len(tensor)):
                var[i] += abs(t["delta"][i])
        n = len(recent)
        avg_var = sum(v / n for v in var) / max(len(tensor), 1)
        return max(0.0, 1.0 - avg_var)

    def _persist(self, entry: dict):
        try:
            os.makedirs(os.path.dirname(TRANSITION_LOG), exist_ok=True)
            with open(TRANSITION_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def load_history(self, n: int = 500):
        """Carrega transições do log persistente."""
        try:
            if os.path.exists(TRANSITION_LOG):
                with open(TRANSITION_LOG) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self._transitions.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                self._transitions = self._transitions[-n:]
        except Exception:
            pass
