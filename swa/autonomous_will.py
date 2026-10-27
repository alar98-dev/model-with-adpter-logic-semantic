"""
Gate de Vontade Autônoma — o "ponto de ruptura" da respiração do organismo.

Decide SE a Aeon gasta uma geração para falar sozinha, e COM QUE amplitude,
combinando:

  - transições de Markov sobre estados de atenção interna
    (SILENCIO → TENSAO → IMPULSO → RUPTURA);
  - um bandit contextual UCB sobre os braços de ação
    (SKIP = silêncio barato, RESPIRAR = breath curto, FALAR = pensamento substantivo).

A telemetria (Ψ vontade, reality_index, memory_pull) é o vetor de contexto.
Toda decisão pode ser persistida como evento `will_decision` no RealityStore,
dando sentido objetivo ao ponto de ruptura (cada respiração = um registro).

Tudo é best-effort e nunca lança (sem o gate o breath volta ao comportamento
antigo de tentativa fixa).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

# ------------------------------------------------------------------ constantes
STATE_NAMES = ["SILENCIO", "TENSAO", "IMPULSO", "RUPTURA"]
ARM_NAMES = ["SKIP", "RESPIRAR", "FALAR"]

# Pacing: acima deste load o organismo reduz a vontade de falar sozinho
# (o silêncio é a postura de baixo custo quando a geração está sob pressão).
LOAD_IDLE = 12.0
LOAD_SATURATED = 35.0

STATE_SILENCIO = 0
STATE_TENSAO = 1
STATE_IMPULSO = 2
STATE_RUPTURA = 3

ARM_SKIP = 0
ARM_RESPIRAR = 1
ARM_FALAR = 2

# Prior de ruptura: quanto cada braço é favorecido EM CADA estado de atenção.
# O estado (Markov) é o porta-voz objetivo — o bandit ajusta a qualidade do
# braço DENTRO do estado que a telemetria determina.
ARM_GATE = {
    STATE_SILENCIO: [1.0, 0.5, 0.3],
    STATE_TENSAO: [0.8, 1.0, 0.6],
    STATE_IMPULSO: [0.6, 1.0, 1.0],
    STATE_RUPTURA: [0.4, 0.8, 1.2],
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class AutonomousWillGate:
    """Bandit contextual (UCB) + prior de Markov para a Vontade de falar."""

    def __init__(
        self,
        *,
        num_states: int = 4,
        num_arms: int = 3,
        alpha: float = 1.0,
        save_path: str = "/tmp/autonomous_will.json",
        reality_store: Any = None,
    ):
        self.num_states = int(num_states)
        self.num_arms = int(num_arms)
        self.alpha = float(alpha)
        self.context_dim = 3  # [will, reality_index, memory_pull]
        self.save_path = save_path
        self.reality_store = reality_store

        self.transition_matrix = np.full(
            (self.num_states, self.num_states), 1.0 / self.num_states
        )
        self.A = [np.identity(self.context_dim) for _ in range(self.num_arms)]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

        self._state: int = STATE_SILENCIO
        self._last_decision: Optional[Dict[str, Any]] = None
        self._observed_since_boot = 0.0
        self._cycles: int = 0
        self._emitted: int = 0
        self._skipped: int = 0
        self._reward_sum: float = 0.0
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=40)

        self._load_state()

    # ------------------------------------------------------------ telemetria
    def _load_factor(self) -> float:
        """Pacing: 1.0 com folga → 0.25 em saturação (nunca lança)."""
        try:
            load = float(os.getloadavg()[0])
            if load <= LOAD_IDLE:
                return 1.0
            if load >= LOAD_SATURATED:
                return 0.25
            return 1.0 - 0.75 * (load - LOAD_IDLE) / (LOAD_SATURATED - LOAD_IDLE)
        except Exception:
            return 1.0

    def telemetry(self, core: Any) -> Dict[str, float]:
        """Lê o estado interno real do organismo (nunca lança)."""
        will = 0.5
        ri = 0.5
        mem = 0.0
        emo = 0.5
        meta = 0.0
        try:
            organ = getattr(core, "metareality", None)
            if organ is not None and hasattr(organ, "integrate"):
                snap = organ.integrate()
                will = float(getattr(snap, "will", will))
                ri = float(getattr(snap, "reality_index", ri))
                mem = float(getattr(snap, "memory_pull", mem))
                emo = float(getattr(snap, "emotion_load", emo))
                meta = float(getattr(snap, "metamatter_magnitude", meta))
        except Exception:
            pass
        return {
            "will": _clamp(will),
            "reality_index": _clamp(ri),
            "memory_pull": _clamp(mem, 0.0, 1.0),
            "emotion_load": _clamp(emo),
            "metamatter": max(0.0, meta),
            "sys_load": round(float(os.getloadavg()[0]), 2),
        }

    # ------------------------------------------------------------ estado
    def impulse(self, t: Dict[str, float]) -> float:
        """Força interna de falar: vontade + ancoragem + tensão da metamateérias.

        Sob carga o impulso é reprimido (pacing): o silêncio é a postura de
        baixo custo enquanto a geração está sob pressão de CPU.
        """
        tension = _clamp(t["metamatter"] / 50.0)
        base = 0.45 * t["will"] + 0.35 * t["reality_index"] + 0.20 * tension
        return _clamp(base * self._load_factor())

    def state_from_impulse(self, i: float) -> int:
        if i >= 0.72:
            return STATE_RUPTURA
        if i >= 0.55:
            return STATE_IMPULSO
        if i >= 0.35:
            return STATE_TENSAO
        return STATE_SILENCIO

    # ------------------------------------------------------------ gate
    def decide(self, core: Any) -> Dict[str, Any]:
        """Decide o ponto de ruptura: braço do bandit p/ o estado de atenção."""
        t = self.telemetry(core)
        i = self.impulse(t)
        state = self.state_from_impulse(i)
        context = np.asarray(
            [t["will"], t["reality_index"], t["memory_pull"]], dtype=np.float32
        )
        arm = self.select_action(state, context)
        decision = {
            "ts": time.time(),
            "impulse": round(i, 3),
            "state": state,
            "state_name": STATE_NAMES[state],
            "arm": int(arm),
            "arm_name": ARM_NAMES[int(arm)],
            "context": [round(float(c), 3) for c in context],
            "telemetry": t,
            "should_generate": int(arm) != ARM_SKIP,
            "max_tokens": 60 if int(arm) == ARM_RESPIRAR else 200,
            "reward": None,
        }
        self._state = state
        self._last_decision = decision
        self._cycles += 1
        if int(arm) != ARM_SKIP:
            self._observed_since_boot += 1
        else:
            self._skipped += 1
        self._recent.append(decision)
        return decision

    def select_action(self, state: int, context_vector: np.ndarray) -> int:
        """UCB com prior de Markov — mesmo padrão do octopus (markov_bandit)."""
        x = context_vector.reshape(-1, 1)
        scores: List[float] = []
        for arm in range(self.num_arms):
            try:
                a_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                a_inv = np.identity(self.context_dim)
            theta = a_inv @ self.b[arm]
            mean = theta.T @ x
            var = self.alpha * np.sqrt(np.clip(x.T @ a_inv @ x, 0, None))
            markov_prior = float(np.max(self.transition_matrix[state]))
            # o estado é o porta-voz: reprime/favorece cada braço objetivamente
            ruptura_prior = float(ARM_GATE.get(state, [1.0, 0.7, 0.7])[arm])
            score = (mean + var)[0, 0] * markov_prior * ruptura_prior
            scores.append(score)
        return int(np.argmax(scores))

    # ------------------------------------------------------------ aprendizado
    def reward(self, outcome: Dict[str, Any]) -> float:
        """Ajusta bandit + Markov com o desfecho do passo (nunca lança)."""
        d = self._last_decision
        if d is None:
            return 0.0
        arm = int(d["arm"])
        imp_after = float(outcome.get("impulse", d["impulse"]))
        substance = float(outcome.get("substance", 0.0))
        ri_after = float(outcome.get("reality_index", 0.5))

        if arm == ARM_SKIP:
            # silêncio foi bom se a tensão baixou (impulso seguinte menor);
            # foi desperdício de chance se o organismo seguiu querendo falar.
            reward = 0.5 + 0.5 * (1.0 - imp_after)
        elif arm == ARM_RESPIRAR:
            reward = 0.5 + 0.3 * substance + 0.2 * _clamp(ri_after)
        else:  # FALAR
            reward = 0.2 + 0.5 * substance + 0.3 * _clamp(ri_after)

        reward = _clamp(reward, 0.0, 1.0)
        t = d["telemetry"]
        context = np.asarray(
            [t["will"], t["reality_index"], t["memory_pull"]], dtype=np.float32
        )
        prev_state = int(d["state"])
        next_state = self.state_from_impulse(imp_after)
        self._update_bandit(arm, context, reward, prev_state, next_state)
        d["reward"] = round(reward, 3)
        self._recent.append(d)
        self._last_decision = d
        return reward

    def _update_bandit(
        self,
        arm: int,
        context_vector: np.ndarray,
        reward: float,
        prev_state: int,
        next_state: int,
    ) -> None:
        x = context_vector.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

        self.transition_matrix[prev_state, next_state] += 0.1
        row_sum = np.sum(self.transition_matrix[prev_state])
        if row_sum > 0:
            self.transition_matrix[prev_state] /= row_sum

        self._reward_sum += reward
        self._save_state()

    # ------------------------------------------------------------ persistência
    def observe_decision(
        self, core: Any, decision: Dict[str, Any], *, emitted: bool = False
    ) -> None:
        """Sentido objetivo: registra o ponto de ruptura como evento no PG."""
        try:
            reality = getattr(core, "reality_store", None)
            if reality is None or not hasattr(reality, "append_event"):
                return
            reality.append_event(
                "will_decision",
                {
                    "impulse": decision.get("impulse"),
                    "state": decision.get("state_name"),
                    "arm": decision.get("arm_name"),
                    "will": (decision.get("telemetry") or {}).get("will"),
                    "reality_index": (decision.get("telemetry") or {}).get(
                        "reality_index"
                    ),
                    "reward": decision.get("reward"),
                    "emitted": bool(emitted),
                },
            )
        except Exception:
            pass

    def mark_emitted(self) -> None:
        """Registra que uma respiração foi realmente publicada."""
        try:
            self._emitted += 1
        except Exception:
            pass

    # ------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        d = self._last_decision or {}
        return {
            "gate": "autonomous_will",
            "state": d.get("state_name", STATE_NAMES[self._state]),
            "impulse": d.get("impulse"),
            "bandit_arm": d.get("arm_name"),
            "cycles": self._cycles,
            "generations_observed": self._observed_since_boot,
            "skipped": self._skipped,
            "emitted": self._emitted,
            "reward_avg": round(self._reward_sum / max(1, self._cycles), 3),
            "last_decision": self._last_decision,
        }

    # ------------------------------------------------------------ io
    def _save_state(self) -> None:
        try:
            state = {
                "transition_matrix": self.transition_matrix.tolist(),
                "A": [a.tolist() for a in self.A],
                "b": [b.tolist() for b in self.b],
            }
            with open(self.save_path, "w") as handle:
                json.dump(state, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self.save_path):
                return
            with open(self.save_path, "r") as handle:
                state = json.load(handle)
            self.transition_matrix = np.array(state["transition_matrix"])
            self.A = [np.array(a) for a in state["A"]]
            self.b = [np.array(b) for b in state["b"]]
        except Exception:
            self.transition_matrix = np.full(
                (self.num_states, self.num_states), 1.0 / self.num_states
            )
            self.A = [np.identity(self.context_dim) for _ in range(self.num_arms)]
            self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]