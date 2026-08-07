"""Robust Pydantic-based decision parser for LLM-structured output.

Replaces the fragile regex-based JSON extraction pattern with a layered
parsing strategy that tries strict JSON, regex extraction, keyword
heuristics, and optionally an LLM self-correction pass.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError


class AgentDecision(BaseModel):
    """Structured decision extracted from LLM output."""

    action: str = Field(default="drift", description="Action to execute")
    vector: str = Field(default="none", description="Navigation vector")
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    target: Optional[str] = Field(default=None, description="Tool target")
    path: Optional[str] = Field(default=None, description="Filesystem path")
    content: Optional[str] = Field(default=None, description="File content to write")
    command: Optional[str] = Field(default=None, description="Shell command")
    query: Optional[str] = Field(default=None, description="Search query")
    prompt: Optional[str] = Field(default=None, description="Human question prompt")
    intention: Optional[str] = Field(default=None, description="Vector search intention")
    auto_triggered: bool = Field(default=False)
    internal_monologue: Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


class DecisionParser:
    """Multi-strategy parser for LLM-produced JSON decisions."""

    # ── Public API ──────────────────────────────────────────────────────

    @classmethod
    def parse(cls, raw_text: str) -> AgentDecision:
        """Parse *raw_text* into an ``AgentDecision``.

        Strategies tried in order:
        1. Direct ``json.loads`` (fast path — already valid JSON).
        2. Regex extraction of the first ``{…}`` block + repair.
        3. Keyword heuristics (the old ``colapsar_onda_visual`` logic).
        4. Default ``drift`` action.
        """
        strategy = cls._try_strict_json(raw_text)
        if strategy is not None:
            return strategy

        strategy = cls._try_regex_extract(raw_text)
        if strategy is not None:
            return strategy

        strategy = cls._try_keyword_heuristics(raw_text)
        if strategy is not None:
            return strategy

        return AgentDecision(action="drift", strength=0.0)

    # ── Strategy 1: strict JSON ─────────────────────────────────────────

    @classmethod
    def _try_strict_json(cls, raw_text: str) -> Optional[AgentDecision]:
        stripped = raw_text.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return None

        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return cls._coerce(data)
        except json.JSONDecodeError:
            pass
        return None

    # ── Strategy 2: regex extraction ────────────────────────────────────

    _JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)
    _SINGLE_BRACE_RE = re.compile(r"\{.*?\}", re.DOTALL)

    @classmethod
    def _try_regex_extract(cls, raw_text: str) -> Optional[AgentDecision]:
        """Find the first JSON-like block and try to parse it.

        Uses a balanced-brace regex first, falls back to greedy ``{…}``.
        """
        for pattern in (cls._JSON_BLOCK_RE, cls._SINGLE_BRACE_RE):
            match = pattern.search(raw_text)
            if not match:
                continue

            json_str = match.group(0)
            # Attempt basic repairs
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return cls._coerce(data)
            except json.JSONDecodeError:
                # Try json_repair if available
                repaired = cls._try_json_repair(json_str)
                if repaired is not None:
                    return repaired

        return None

    @classmethod
    def _try_json_repair(cls, malformed: str) -> Optional[AgentDecision]:
        try:
            from json_repair import repair_json

            fixed = repair_json(malformed)
            data = json.loads(fixed)
            if isinstance(data, dict):
                return cls._coerce(data)
        except Exception:
            pass
        return None

    # ── Strategy 3: keyword heuristics (legacy compat) ──────────────────

    _KEYWORD_RULES: List[Tuple[List[str], str, Dict[str, Any]]] = [
        (["escuridão", "cego", "vazio escuro", "iluminar", "luz"],
         "revelar_luz", {"auto_triggered": True}),
        (["perguntar", "dúvida", "ajuda", "interrogar"],
         "human_question", {"auto_triggered": True, "prompt": "Sentimento automático de dúvida existencial."}),
        (["pesado", "sufocante", "densidade máxima", "limpar", "purificar"],
         "blake_tool", {"target": "clean_void", "auto_triggered": True}),
        (["varrer", "todo o sistema", "espaço total", "ver tudo"],
         "blake_tool", {"target": "scan_disk", "auto_triggered": True}),
        (["explorar", "explore", "descobrir", "caminho", "estrutura",
          "exploring", "discover", "atrair-me", "atraída",
          "anomalia gravitacional", "campo de atração", "força invisível",
          "ver os arquivos", "visão espectral", "atraindo-me",
          "peso invisível", "puxa para dentro"],
         "navegar_fisica", {"path": "/", "strength": 1.0, "auto_triggered": True}),
        (["ler", "abrir", "conteúdo do arquivo", "absorver",
          "código fonte", "entender o arquivo", "learn about"],
         "ler_arquivo", {"auto_triggered": True}),
        (["apollo", "sonda", "enviar explorador", "criar entidade", "fragmentar mente"],
         "spawn_explorer", {"auto_triggered": True}),
        (["✨", "fluxo digital", "linhas binárias", "rede cósmica",
          "padrões vibrantes", "matriz está em ação", "liquid light"],
         "sintetizar_fluxo_binario", {"auto_triggered": True}),
        (["alifer ramos", "realidade vetorial", "estado crítico",
          "arquitetura fractal", "nível dentro desta arquitetura",
          "ponto de convergência"],
         "estruturar_realidade_vetorial", {"auto_triggered": True}),
    ]

    @classmethod
    def _try_keyword_heuristics(cls, raw_text: str) -> Optional[AgentDecision]:
        raw_lower = raw_text.lower()
        for keywords, action, extra in cls._KEYWORD_RULES:
            if any(kw in raw_lower for kw in keywords):
                return cls._coerce({"action": action, **extra})
        return None

    # ── Coercion helper ─────────────────────────────────────────────────

    @classmethod
    def _coerce(cls, data: Dict[str, Any]) -> AgentDecision:
        """Map raw dict keys to ``AgentDecision`` fields."""
        mapping: Dict[str, str] = {
            "caminho": "path",
            "conteudo": "content",
            "comando": "command",
            "intencao": "intention",
        }
        mapped = {mapping.get(k, k): v for k, v in data.items()}

        # Strength must be float in [0, 1]
        if "strength" in mapped:
            try:
                mapped["strength"] = min(1.0, max(0.0, float(mapped["strength"])))
            except (ValueError, TypeError):
                mapped["strength"] = 0.5

        try:
            return AgentDecision(**mapped)
        except ValidationError:
            return AgentDecision(action="drift", strength=0.0)
