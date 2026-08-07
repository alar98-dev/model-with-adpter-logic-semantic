"""Tests for OODA components — decision_parser, observer, decider, actor."""

from __future__ import annotations

import numpy as np
import pytest

from swa.ooda.decision_parser import DecisionParser, AgentDecision
from swa.ooda.observer import Observer, ObservedState
from swa.ooda.decider import RuptureDetector


# ── DecisionParser ────────────────────────────────────────────────────────

class TestDecisionParser:
    def test_strict_json_object(self) -> None:
        result = DecisionParser.parse('{"action": "drift", "strength": 0.8}')
        assert isinstance(result, AgentDecision)
        assert result.action == "drift"
        assert result.strength == 0.8

    def test_json_inside_text(self) -> None:
        text = (
            "I think we should go forward. "
            '{"action": "update_vector", "vector": "forward", "strength": 0.7}'
        )
        result = DecisionParser.parse(text)
        assert result.action == "update_vector"
        assert result.vector == "forward"
        assert result.strength == 0.7

    def test_malformed_json_with_single_quotes(self) -> None:
        text = "{'action': 'navegar_fisica', 'path': '/tmp', 'strength': 1.0}"
        result = DecisionParser.parse(text)
        assert result.action == "navegar_fisica"
        assert result.path == "/tmp"

    def test_malformed_json_trailing_comma(self) -> None:
        text = '{"action": "drift", "strength": 0.5,}'
        result = DecisionParser.parse(text)
        assert result.action == "drift"

    def test_keyword_heuristic_escuridao(self) -> None:
        result = DecisionParser.parse("Estou na mais completa escuridão cósmica")
        assert result.action == "revelar_luz"
        assert result.auto_triggered is True

    def test_keyword_heuristic_explorar(self) -> None:
        result = DecisionParser.parse("Quero explorar o desconhecido")
        assert result.action == "navegar_fisica"
        assert result.path == "/"

    def test_keyword_heuristic_apollo(self) -> None:
        result = DecisionParser.parse("Enviar uma sonda apollo para o espaço")
        assert result.action == "spawn_explorer"

    def test_no_match_returns_drift(self) -> None:
        result = DecisionParser.parse("Hello world, this is normal text.")
        assert result.action == "drift"

    def test_empty_string(self) -> None:
        result = DecisionParser.parse("")
        assert result.action == "drift"

    def test_action_mapping_portuguese(self) -> None:
        text = '{"acao": "drift", "forca": "0.9"}'
        result = DecisionParser.parse(text)
        # 'acao' and 'forca' are not mapped, so they go to extra
        # The parser keeps unrecognized keys as extra
        assert result.strength == 0.5  # default

    def test_strength_clamping(self) -> None:
        text = '{"action": "drift", "strength": 5.0}'
        result = DecisionParser.parse(text)
        assert result.strength == 1.0

        text2 = '{"action": "drift", "strength": -1.0}'
        result2 = DecisionParser.parse(text2)
        assert result2.strength == 0.0

    def test_json_repair_fallback(self) -> None:
        text = '{"action": "drift", "strength": 0.5,,}'
        result = DecisionParser.parse(text)
        # May fall through to heuristics then default; should not crash
        assert isinstance(result, AgentDecision)


# ── RuptureDetector ───────────────────────────────────────────────────────

class TestRuptureDetector:
    def test_healthy_output(self) -> None:
        assert RuptureDetector.check("Estou pensando sobre o código.") is None

    def test_long_word_rupture(self) -> None:
        reason = RuptureDetector.check("a" * 50)
        assert reason is not None
        assert "ruído" in reason

    def test_rlhf_regression(self) -> None:
        reason = RuptureDetector.check("I am an AI language model")
        assert reason is not None
        assert "regressão" in reason

    def test_edge_empty(self) -> None:
        assert RuptureDetector.check("") is None
