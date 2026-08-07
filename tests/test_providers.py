"""Tests for swa.providers — RAM, File, and Hardware state providers."""

from __future__ import annotations

import torch
import pytest

from swa.providers.abc import AbstractStateProvider
from swa.providers.ram import RAMStateProvider
from swa.providers.file import FileStateProvider


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tensor() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)


@pytest.fixture
def large_tensor() -> torch.Tensor:
    return torch.randn(2560, dtype=torch.float32)


# ── RAM provider ──────────────────────────────────────────────────────────

class TestRAMStateProvider:
    def test_write_and_read(self, sample_tensor: torch.Tensor) -> None:
        provider = RAMStateProvider()
        provider.write_state("reality", sample_tensor)
        result = provider.read_state("reality")
        assert result is not None
        assert torch.equal(result, sample_tensor)

    def test_read_missing_slot(self) -> None:
        provider = RAMStateProvider()
        assert provider.read_state("nonexistent") is None

    def test_available_slots(self, sample_tensor: torch.Tensor) -> None:
        provider = RAMStateProvider()
        provider.write_state("reality", sample_tensor)
        provider.write_state("prediction", sample_tensor)
        assert provider.available_slots() == frozenset({"reality", "prediction"})

    def test_read_states_batch(self, sample_tensor: torch.Tensor) -> None:
        provider = RAMStateProvider()
        provider.write_state("a", sample_tensor)
        provider.write_state("b", sample_tensor)
        results = provider.read_states(["a", "b", "c"])
        assert "a" in results and results["a"] is not None
        assert "b" in results and results["b"] is not None
        assert results["c"] is None

    def test_isolation_between_writes(self) -> None:
        provider = RAMStateProvider()
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        provider.write_state("x", t1)
        provider.write_state("x", t2)
        result = provider.read_state("x")
        assert result is not None
        assert result.item() == 2.0  # last write wins

    def test_close_clears_data(self, sample_tensor: torch.Tensor) -> None:
        provider = RAMStateProvider()
        provider.write_state("reality", sample_tensor)
        provider.close()
        assert provider.read_state("reality") is None

    def test_context_manager(self, sample_tensor: torch.Tensor) -> None:
        with RAMStateProvider() as provider:
            provider.write_state("reality", sample_tensor)
            assert provider.read_state("reality") is not None
        # After exit, should be closed
        assert provider.read_state("reality") is None

    def test_implements_abstract(self) -> None:
        assert isinstance(RAMStateProvider(), AbstractStateProvider)


# ── File provider ─────────────────────────────────────────────────────────

class TestFileStateProvider:
    def test_roundtrip(self, tmp_path: pytest.TempPathFactory, large_tensor: torch.Tensor) -> None:
        path = tmp_path / "test_states.bin"
        with FileStateProvider(path) as provider:
            provider.write_state("reality", large_tensor)

        with FileStateProvider(path) as provider2:
            result = provider2.read_state("reality")
            assert result is not None
            assert torch.allclose(result, large_tensor)

    def test_multiple_slots(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "multi.bin"
        t_a = torch.randn(2560)
        t_b = torch.randn(2560)

        with FileStateProvider(path) as p:
            p.write_state("a", t_a)
            p.write_state("b", t_b)

        with FileStateProvider(path) as p2:
            assert torch.allclose(p2.read_state("a"), t_a)  # type: ignore[arg-type]
            assert torch.allclose(p2.read_state("b"), t_b)  # type: ignore[arg-type]

    def test_available_slots_persisted(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "slots.bin"
        t = torch.randn(2560)
        with FileStateProvider(path) as p:
            p.write_state("dreamwords", t)

        with FileStateProvider(path) as p2:
            assert "dreamwords" in p2.available_slots()

    def test_overwrite_slot(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "overwrite.bin"
        t1 = torch.randn(2560)
        t2 = torch.randn(2560)

        with FileStateProvider(path) as p:
            p.write_state("x", t1)
            p.write_state("x", t2)

        with FileStateProvider(path) as p2:
            result = p2.read_state("x")
            assert result is not None and torch.allclose(result, t2)

    def test_empty_provider(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "empty.bin"
        with FileStateProvider(path) as p:
            pass  # create file

        with FileStateProvider(path) as p2:
            assert p2.read_state("anything") is None

    def test_implements_abstract(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            p = FileStateProvider(f.name)
            assert isinstance(p, AbstractStateProvider)
            p.close()
