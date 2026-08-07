"""Tests for TokenizerManager — AbstractTokenizer conformance and backward compat."""

from __future__ import annotations

from swa.tokenizer_manager import (
    TokenizerManager,
    TokenizerCompatibilityError,
    TokenizerCorruptedError,
    TokenizerMissingError,
    TokenizerVersionMismatchError,
    SentencePieceMissingError,
    TokenizerHealth,
)
from swa.ooda.interfaces import AbstractTokenizer


class TestTokenizerExceptions:
    def test_exception_types_exist(self) -> None:
        assert issubclass(TokenizerCompatibilityError, Exception)
        assert issubclass(TokenizerCorruptedError, Exception)
        assert issubclass(TokenizerMissingError, Exception)
        assert issubclass(TokenizerVersionMismatchError, Exception)
        assert issubclass(SentencePieceMissingError, Exception)

    def test_raise_and_catch(self) -> None:
        try:
            raise TokenizerCompatibilityError("test")
        except TokenizerCompatibilityError:
            pass


class TestTokenizerHealth:
    def test_default_values(self) -> None:
        h = TokenizerHealth()
        assert h.state == "loading"
        assert h.fast_tokenizer_available is False
        assert h.slow_tokenizer_active is False
        assert h.model_type is None
        assert h.error is None

    def test_asdict_keys(self) -> None:
        from dataclasses import asdict
        h = TokenizerHealth()
        d = asdict(h)
        for key in ("state", "fast_tokenizer_available", "slow_tokenizer_active", "model_type"):
            assert key in d


class TestTokenizerManagerInterface:
    def test_implements_abstract_tokenizer(self) -> None:
        assert issubclass(TokenizerManager, AbstractTokenizer)

    def test_init(self) -> None:
        tm = TokenizerManager()
        assert tm._health.state == "loading"
        assert tm._tokenizer is None

    def test_encode_no_tokenizer(self) -> None:
        tm = TokenizerManager()
        assert tm.encode("hello") == []

    def test_decode_no_tokenizer(self) -> None:
        tm = TokenizerManager()
        assert tm.decode([0, 1, 2]) == ""

    def test_get_vocab_size_no_tokenizer(self) -> None:
        tm = TokenizerManager()
        assert tm.get_vocab_size() == 0

    def test_get_config_no_tokenizer(self) -> None:
        tm = TokenizerManager()
        assert tm.get_config() == {}

    def test_has_backward_compat_properties(self) -> None:
        tm = TokenizerManager()
        assert hasattr(tm, "validate")
        assert hasattr(tm, "load")
        assert hasattr(tm, "warmup")
        assert hasattr(tm, "health")
        assert hasattr(tm, "metadata")
        assert hasattr(tm, "get_tokenizer")
        assert hasattr(tm, "tokenizer")

    def test_health_returns_dict(self) -> None:
        tm = TokenizerManager()
        h = tm.health()
        assert isinstance(h, dict)
        assert "state" in h

    def test_metadata_returns_dict(self) -> None:
        tm = TokenizerManager()
        m = tm.metadata()
        assert isinstance(m, dict)
        assert m["available"] is False

    def test_get_tokenizer_returns_none(self) -> None:
        tm = TokenizerManager()
        assert tm.get_tokenizer() is None

    def test_tokenizer_property_returns_none(self) -> None:
        tm = TokenizerManager()
        assert tm.tokenizer is None

    def test_warmup_no_tokenizer(self) -> None:
        tm = TokenizerManager()
        tm.warmup()  # Should not raise

    def test_validate_missing_dir(self) -> None:
        tm = TokenizerManager()
        report = tm.validate("/nonexistent/path")
        assert report["exists"] is False
        assert "model_dir_missing" in report.get("issues", [])
