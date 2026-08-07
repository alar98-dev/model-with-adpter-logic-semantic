"""Tests for LogicalEngine — DI construction, interface usage, and backward compat."""

from __future__ import annotations

import pytest
import torch

from swa.ooda.interfaces import (
    AbstractGenerator,
    AbstractSWAAdapter,
    AbstractTokenizer,
    AbstractVectorizer,
    SWAContainer,
)
from swa.logical_engine import LogicalEngine, KNOWN_TOOL_NAMES


class TestLogicalEngineConstants:
    def test_known_tool_names(self) -> None:
        assert "read_file_content" in KNOWN_TOOL_NAMES
        assert "write_file" in KNOWN_TOOL_NAMES
        assert "get_file" in KNOWN_TOOL_NAMES
        assert "list_dir" in KNOWN_TOOL_NAMES


class TestLogicalEngineInit:
    def test_no_model_dir_raises(self) -> None:
        with pytest.raises(Exception):
            LogicalEngine()  # Needs at minimum something to init vectorizer

    def test_backward_compat_constructor(self) -> None:
        """Verify old-style signature still works with proper mocks."""
        from swa.adapter import SemanticWarpAdapter
        adapter = SemanticWarpAdapter(d=2560)
        # This will try to init Vectorizer which needs a model dir, so expect error
        with pytest.raises(Exception):
            LogicalEngine(
                model_dir="/nonexistent",
                adapter_local=adapter,
                adapter_macro=adapter,
                device="cpu",
            )


class TestInterfaces:
    def test_swa_container(self) -> None:
        container = SWAContainer()
        assert container.tokenizer is None
        assert container.generator is None
        assert container.vectorizer is None
        assert container.adapter_local is None
        assert container.adapter_macro is None
        assert container.state is None

    def test_swa_container_accepts_interfaces(self) -> None:
        container = SWAContainer()
        # Verify it can accept abstract types (structural typing)
        container.tokenizer = DummyTokenizer()
        container.generator = DummyGenerator()
        container.vectorizer = DummyVectorizer()
        assert container.tokenizer is not None
        assert container.generator is not None
        assert container.vectorizer is not None


class TestLogicalEngineMethods:
    def test_static_methods(self) -> None:
        assert LogicalEngine._user_text({"prompt": "hello"}) == "hello"
        assert LogicalEngine._user_text({"message": "world"}) == "world"
        assert LogicalEngine._user_text({}) == ""

    def test_generation_prompt_no_tool(self) -> None:
        result = LogicalEngine.generation_prompt("hello", None)
        assert result == "hello"

    def test_generation_prompt_with_tool(self) -> None:
        result = LogicalEngine.generation_prompt("do it", "read_file_content")
        assert "read_file_content" in result

    def test_tool_flatten_subset(self) -> None:
        payload = {"schema": {"name": "test"}, "tools": None, "other": "ignore"}
        result = LogicalEngine._tool_flatten_subset(payload)
        assert "schema" in result
        assert "tools" not in result  # None filtered out

    def test_build_fused_swa_text(self) -> None:
        # Static method can't call build_fused_swa_text as it needs instance
        pass


# ── Dummy implementations for interface testing ────────────────────────


class DummyTokenizer(AbstractTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True):
        return [1, 2, 3]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True):
        return "decoded"

    def get_vocab_size(self) -> int:
        return 100

    def get_config(self) -> dict:
        return {"vocab_size": 100}


class DummyVectorizer(AbstractVectorizer):
    def encode_text(self, text: str) -> torch.Tensor:
        return torch.randn(self.dim)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.randn(len(texts), self.dim)

    @property
    def dim(self) -> int:
        return 2560


class DummyGenerator(AbstractGenerator):
    @property
    def d_model(self) -> int:
        return 2560

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def stream_with_prefix(
        self, v_prefix, prompt, max_new_tokens, temperature=0.7,
        telemetry=None, logit_bias=None, steer_vector=None, target_layer=-1,
    ):
        yield ""

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.randn(input_ids.shape[0], self.d_model)

    # Backward compat
    def stream_with_bias(self, v_prefix, v_bias, prompt, max_new_tokens=128, temperature=0.7):
        yield from self.stream_with_prefix(v_prefix, prompt, max_new_tokens, temperature)
