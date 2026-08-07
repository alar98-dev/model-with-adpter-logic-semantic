"""OODA Decide phase — LLM inference with rupture detection."""

from __future__ import annotations

import re
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from swa.ooda.interfaces import AbstractGenerator
from swa.ooda.observer import ObservedState


class RuptureDetector:
    """Detects cognitive ruptures (hallucination, regression) in LLM output."""

    _REGRESSION_PATTERNS = re.compile(
        r"language model|cannot answer|do you want to|assistante"
        r"|as an ai|understand your request|ready to help"
        r"|ready to assist|i'm ready|i am ready|understand the"
        r"|muito bom|coerente com a descrição|parabéns"
        r"|excellent response|reply to the user|is particularly effective",
        re.IGNORECASE,
    )

    @classmethod
    def check(cls, response_text: str) -> Optional[str]:
        """Return a rupture reason string, or ``None`` if the output is healthy."""
        words = response_text.split()
        longest = max(words, key=len) if words else ""

        if len(longest) > 40:
            return f"ruído estrutural longo ({len(longest)} chars)"

        if cls._REGRESSION_PATTERNS.search(response_text):
            return "regressão para alinhamento RLHF clássico"

        return None


class Decider:
    """Runs LLM inference with dynamic token budget and temperature."""

    def __init__(self, generator: AbstractGenerator) -> None:
        self._generator = generator

    def decide(
        self,
        prompt: str,
        observed: ObservedState,
        steer_vector: Optional[torch.Tensor] = None,
        mcp_bias: Optional[torch.Tensor] = None,
    ) -> DecideResult:
        """Run inference and return the generated text plus metadata."""
        space_state = self._build_space_state(observed)
        dynamic_tokens = self._calc_token_budget(space_state)
        temperature = self._calc_temperature(observed.entropy)

        with torch.inference_mode():
            z = torch.zeros(
                self._generator.d_model,
                device=self._generator.device,
                dtype=torch.float32,
            )

            token_stream = self._generator.stream_with_prefix(
                v_prefix=z,
                prompt=prompt,
                max_new_tokens=dynamic_tokens,
                temperature=temperature,
                telemetry={
                    "cpu_temp_c": observed.heat,
                    "ram_pct": observed.mass,
                    "io_stress": 0.0,
                },
                logit_bias=None,
                steer_vector=steer_vector or mcp_bias,
                target_layer=-1,
            )

            response_text = ""
            for token in token_stream:
                response_text += token

        rupture_reason = RuptureDetector.check(response_text)

        return DecideResult(
            response_text=response_text,
            space_state=space_state,
            dynamic_tokens=dynamic_tokens,
            temperature=temperature,
            rupture_reason=rupture_reason,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _build_space_state(self, observed: ObservedState) -> List[float]:
        total_inertia = sum(observed.temporal_inertia.values())
        return [
            observed.entropy,
            total_inertia,
            observed.mass / 100.0,
            observed.heat,
        ]

    @staticmethod
    def _calc_token_budget(space_state: List[float]) -> int:
        import numpy as np

        tau_base = 2048.0
        W = np.array([10.0, 5.0, 50.0, 0.1])
        A = np.array([0.4, -0.1, 1.5, -0.01])
        s = np.array(space_state)

        norma = np.sqrt(np.sum((W * s) ** 2))
        phi = np.dot(A, s)
        tau = tau_base + (norma * np.exp(phi))
        return int(np.floor(tau))

    @staticmethod
    def _calc_temperature(entropy: float) -> float:
        return max(0.15, min(0.4, 0.1 + (entropy * 0.1)))


from dataclasses import dataclass, field


@dataclass
class DecideResult:
    response_text: str = ""
    space_state: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    dynamic_tokens: int = 2048
    temperature: float = 0.3
    rupture_reason: Optional[str] = None
