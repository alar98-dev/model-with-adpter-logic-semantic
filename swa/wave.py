"""Shared wave/EOS/fission decisions for both generation entry points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class WaveDecision:
    closed: bool
    pressure: float
    should_fission: bool
    reason: str
    mode: str = "FECHAR"


# Modos de fissão (octopus v2): o polvo escolhe COMO continuar.
FOLLOW = "FOLLOW"        # re-ancorar na k-base (v_prefix do polvo)
CONTINUAR = "CONTINUAR"  # continuar com o fluxo clássico (ban_eos + fase)
DIVERGIR = "DIVERGIR"    # exploração: temperatura elevada, recomeço livre
FECHAR = "FECHAR"        # encerrar (EOS)


class FissionModeBandit:
    """UCB multi-armado sobre os modos de fissão, agrupado por macro-estado.

    Aprende qual modo (FOLLOW/CONTINUAR/DIVERGIR) historicamente rendeu mais
    qualidade por estado (budget de fissão × vontade × carga emocional).
    Estado puro: sem dependência da árvore aeon-engine (roda até sem polvo).
    """

    def __init__(self, alpha: float = 2.0):
        self._alpha = alpha
        self._counts: dict = {}
        self._sums: dict = {}
        self._plays: dict = {}
        self._last: Optional[tuple] = None  # (bucket, arm) escolhido

    # ------------------------------------------------------------- util
    @staticmethod
    def _bucket(fission_count: int, semantic_will: float, emotion_load: float):
        will_bin = "low" if semantic_will < 0.6 else ("mid" if semantic_will < 0.85 else "high")
        return (
            min(int(fission_count), 3),
            will_bin,
            "over" if float(emotion_load or 0.0) > 0.8 else "ok",
        )

    def arms_for(self, ctx: dict) -> list:
        arms = [FOLLOW, CONTINUAR, DIVERGIR]
        if int(ctx.get("fission_count", 0)) >= int(ctx.get("max_fission", 0)):
            arms = [FECHAR]
        return arms

    # ------------------------------------------------------------- choose
    def choose(self, **ctx) -> str:
        bucket = self._bucket(
            ctx.get("fission_count", 0),
            ctx.get("semantic_will", 0.5),
            ctx.get("emotion_load", 0.5),
        )
        arms = self.arms_for(ctx)
        if not arms:
            return FECHAR
        if len(arms) == 1 and arms[0] == FECHAR:
            self._last = (bucket, FECHAR)
            return FECHAR

        total = self._plays.get(bucket, 0) + 1  # inclui esta escolha
        best_arm, best_score = arms[0], float("-inf")
        for arm in arms:
            n = self._counts.get((bucket, arm), 0)
            mean = self._sums.get((bucket, arm), 0.0) / n if n else 0.0
            if n == 0:
                score = float("inf")  # explora arm não testado
            else:
                score = mean + self._alpha * ((math.log(total) / n) ** 0.5)
            if score > best_score:
                best_arm, best_score = arm, score
        self._last = (bucket, best_arm)
        return best_arm

    # ----------------------------------------------------------- observe
    def last_arm(self) -> Optional[str]:
        return self._last[1] if self._last else None

    def reset(self) -> None:
        """Zera a referência do último braço ao iniciar um novo turno.

        Preserva as estatísticas aprendidas (n/mean por braço); apenas
        impede que um braço escolhido e nunca observado seja creditado
        por engano no próximo turno.
        """
        self._last = None

    def observe(self, reward: float) -> None:
        """Aplica a recompensa da onda ao braço que a lançou (last_arm)."""
        if self._last is None:
            return
        bucket, arm = self._last
        key = (bucket, arm)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._sums[key] = self._sums.get(key, 0.0) + float(reward)
        self._plays[bucket] = self._plays.get(bucket, 0) + 1
        self._last = None  # consome a escolha (uma observação por onda)

    # ------------------------------------------------------------ stats
    def stats(self) -> dict:
        out = {}
        for (bucket, arm), n in self._counts.items():
            out[f"{bucket[0]}-{bucket[1]}-{bucket[2]}/{arm}"] = {
                "n": n,
                "mean": round(self._sums[(bucket, arm)] / n, 4),
            }
        return out


def eos_ids_for(model, tokenizer) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {value}
    return {int(v) for v in (value or [])}


def generation_closed_from_result(
    result,
    sequences,
    eos_ids: Iterable[int],
    *,
    max_new_tokens: int | None = None,
    prefix_length: int | None = None,
) -> bool:
    """Check EOS using only tokens actually emitted by ``generate``.

    Transformers may pad ``sequences`` with ``pad_token_id`` (often EOS) after
    an early stop. ``result.scores`` has one entry per real generation step, so
    it is the authoritative generated length.
    """
    scores = getattr(result, "scores", None)
    if scores is None:
        scores = ()
    generated_steps = len(scores)
    if generated_steps <= 0 or sequences is None or not getattr(sequences, "numel", lambda: 0)():
        return False

    sequence_length = int(sequences.shape[1])
    budget = int(max_new_tokens or generated_steps)
    if prefix_length is None:
        # Early-stop results are padded to the requested budget. Otherwise the
        # sequence contains exactly the emitted prefix plus generated tokens.
        if generated_steps < budget and sequence_length >= budget:
            prefix_length = sequence_length - budget
        else:
            prefix_length = sequence_length - generated_steps
    index = int(prefix_length) + generated_steps - 1
    if index < 0 or index >= sequence_length:
        return False
    return int(sequences[0, index]) in {int(token) for token in eos_ids}


def decide_wave(
    *,
    generated_tokens: int,
    budget: int,
    closed: bool,
    fission_count: int,
    max_fission: int,
    shield_throttle: float = 0.0,
    matrix_stability: float = 1.0,
    emotion_load: float = 0.5,
    emotion_label: str = "neutral",
    semantic_will: float = 0.0,
    residual_energy: float = 0.0,
    field_coherence: float = 1.0,
    explicit_continuation: bool = False,
    matrixial_force: float = 1.0,
    degenerate: bool = False,
    semantic_drift: float | None = None,
    semantic_convergence: float | None = None,
    # --- Octopus v2: fissão orientada pela k-base ---
    bandit: FissionModeBandit | None = None,
    intent_pull: float | None = None,
    repio_precision: float | None = None,
    fruitfulness: float | None = None,
) -> WaveDecision:
    """Decide a próxima onda.

    O clássico decide SE há fissão (timing). Quando `bandit` está online e a
    fissão já foi decidida, o polvo governa COMO continuar (FOLLOW/CONTINUAR/
    DIVERGIR). O polvo nunca força uma fissão que o clássico negaria.
    """
    pressure = min(1.0, generated_tokens / max(1, budget))
    if degenerate:
        return WaveDecision(False, pressure, False, "repetition_collapse", FECHAR)
    if semantic_drift is not None and float(semantic_drift) >= 0.85:
        return WaveDecision(False, pressure, False, "semantic_drift", FECHAR)

    locked = (
        shield_throttle > 0.5
        or matrix_stability < 0.30
        or emotion_label == "overwhelmed"
        or emotion_load > 0.85
    )

    # ------------------------------------------------------------ timing
    # (ramo clássico: determina APENAS should_fission; o modo é o default)
    should_fission = False
    reason = "eos"
    mode = FECHAR
    wave_closed = False

    if locked:
        wave_closed = closed
        reason = "eos_locked" if closed else "locked"
    elif closed:
        wave_closed = True
        process_capacity = min(1.0, residual_energy / 200.0)
        convergence = (
            max(0.0, min(1.0, float(semantic_convergence)))
            if semantic_convergence is not None
            else 0.5
        )
        vector_demand = (
            0.35 * max(0.0, min(1.0, semantic_will))
            + 0.30 * process_capacity
            + 0.20 * max(0.0, min(1.0, field_coherence))
            + 0.15 * convergence
        )
        if (
            fission_count < max_fission
            and (explicit_continuation or semantic_will >= 0.85)
            and semantic_will >= 0.70
            and vector_demand >= 0.72
            and shield_throttle <= 0.5
            and matrix_stability >= 0.30
            and emotion_label != "overwhelmed"
            and emotion_load <= 0.85
        ):
            should_fission = True
            reason = "dynamic_vector_budget"
            mode = CONTINUAR
        else:
            reason = "eos"
    elif fission_count >= max_fission:
        reason = "fission_limit"
    elif pressure < 0.98:
        reason = "budget_not_exhausted"
    elif matrixial_force < 0.25:
        reason = "insufficient_matrixial_force"
    elif shield_throttle > 0.5:
        reason = "shield"
    elif matrix_stability < 0.30:
        reason = "unstable"
    elif emotion_label == "overwhelmed" or emotion_load > 0.85:
        reason = "overloaded"
    else:
        should_fission = True
        reason = "residual_energy"
        mode = CONTINUAR

    # --------------------------------------------------------- v2 governança
    # Octopus governa COMO (modo) — nunca SE (timing). Só entra quando a
    # fissão clássica já aprovou; o braço escolhido é creditado via observe().
    octopus_online = (
        should_fission
        and bandit is not None
        and (intent_pull is not None or repio_precision is not None)
    )
    if octopus_online:
        arm = bandit.choose(
            fission_count=fission_count,
            max_fission=max_fission,
            semantic_will=semantic_will,
            emotion_load=emotion_load,
            pressure=pressure,
            intent_pull=intent_pull,
            repio_precision=repio_precision,
            fruitfulness=fruitfulness,
        )
        if arm == FOLLOW:
            mode = FOLLOW
            reason = (
                f"octopus::follow pull={float(intent_pull):.2f}"
                if intent_pull is not None
                else "octopus::follow"
            )
        elif arm == DIVERGIR:
            mode, reason = DIVERGIR, "octopus::diverge"
        else:
            mode, reason = CONTINUAR, "octopus::continuar"

    return WaveDecision(wave_closed, pressure, should_fission, reason, mode)


def clean_wave_text(text: str) -> str:
    return (text or "").replace("<end_of_turn>", "").replace("<eos>", "").rstrip()
