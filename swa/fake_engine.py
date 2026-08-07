from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional


@dataclass
class FakeHealth:
    state: str = "fake"
    reason: str = "Model runtime disabled or unavailable"


class FakeEngine:
    """Engine de fallback.

    Objetivos:
    - NUNCA importar/carregar transformers/tokenizer/model
    - Garantir que REST API suba com lifecycle resiliente
    - Fornecer interface mínima usada pelos handlers
    """

    state: str = "fake"

    def health(self) -> Dict[str, Any]:
        return {"status": "engine_fake", "backend_state": self.state, "reason": FakeHealth().reason}

    def capabilities(self) -> Dict[str, Any]:
        return {"chat": False, "stream": False, "process": True, "engine": self.state}

    def run_optimized_stream(
        self,
        payload: Any,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        # Handlers em fake mode devem retornar 503 antes; isso é segurança extra.
        raise RuntimeError("engine_fake: model runtime disabled")

    def process_request(self, req: dict):
        # Endpoint /api/v1/process pode decidir entre 503 ou placeholder.
        # Mantemos um placeholder para observabilidade.
        return {
            "status": "engine_fake",
            "backend_state": self.state,
            "reason": FakeHealth().reason,
            "request_echo": {k: req.get(k) for k in ("schema", "texts", "prompt", "message") if k in req},
        }

