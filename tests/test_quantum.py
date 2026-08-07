"""Tests for swa/quantum/ sub-packages — MCPActivityTracker, SemanticMemory, AnchorStorage."""

from __future__ import annotations

import pytest
import torch

from swa.quantum.bridge import QuantumPhaseBridge
from swa.quantum.mcp import MCPActivityTracker
from swa.quantum.memory import SemanticMemory
from swa.quantum.storage import AnchorStorage


class TestMCPActivityTracker:
    def test_init_without_db(self) -> None:
        tracker = MCPActivityTracker(None, None)
        assert tracker.conn is None
        assert tracker.cursor is None

    def test_log_activity_no_db(self) -> None:
        tracker = MCPActivityTracker(None, None)
        # Should not raise
        tracker.log_activity("s1", "patch_edit_file", {"file_path": "/tmp/x.py"})

    def test_fetch_context_no_db(self) -> None:
        tracker = MCPActivityTracker(None, None)
        ctx = tracker.fetch_context(limit=3)
        assert isinstance(ctx, str)
        assert "Unavailable" in ctx or "physics" in ctx.lower()

    def test_fetch_mcp_context_no_db(self) -> None:
        tracker = MCPActivityTracker(None, None)
        ctx = tracker.fetch_mcp_context(limit=5)
        assert isinstance(ctx, str)

    def test_compute_gravity_no_db(self) -> None:
        tracker = MCPActivityTracker(None, None)
        bias = tracker.compute_gravity(device="cpu")
        assert bias is None

    def test_classify_impact_patch(self) -> None:
        tracker = MCPActivityTracker(None, None)
        result = tracker._classify_impact("patch_edit_file", {"file_path": "x.py", "replace_block": "a\nb\nc"})
        assert "Topological deformation" in result
        assert "magnitude 3" in result

    def test_classify_impact_project_scripts(self) -> None:
        tracker = MCPActivityTracker(None, None)
        result = tracker._classify_impact("project_scripts", {"action": "run", "script": "test.sh"})
        assert "Kinetic resonance" in result

    def test_classify_impact_create_files(self) -> None:
        tracker = MCPActivityTracker(None, None)
        result = tracker._classify_impact("create_project_files", {"files": ["a.py", "b.py"]})
        assert "Condensation" in result
        assert "2" in result

    def test_classify_impact_unknown(self) -> None:
        tracker = MCPActivityTracker(None, None)
        result = tracker._classify_impact("unknown_tool", {"some": "data"})
        assert "unidentified" in result.lower()


class TestSemanticMemory:
    def test_init(self) -> None:
        mem = SemanticMemory(None, None)
        assert mem.gold_dimension == 2560
        assert mem.phi_ratio == 1.61803398875
        assert mem._cached_L is None
        assert mem._cached_w is None
        assert mem._recursive_thought_accumulator is None

    def test_load_cache_no_db_creates_defaults(self) -> None:
        mem = SemanticMemory(None, None, gold_dimension=2560)
        mem.load_cache(device="cpu", dtype=torch.float32)
        assert mem._cached_L is not None
        assert mem._cached_L.shape == (1, 2560)
        assert mem._cached_w is not None
        assert mem._cached_w.shape == (1,)

    def test_compute_mass_on_empty_cache(self) -> None:
        mem = SemanticMemory(None, None, gold_dimension=2560)
        x = torch.randn(2560)
        mass = mem.compute_mass(x, sigma=1.0, device="cpu", dtype=torch.float32)
        assert isinstance(mass, torch.Tensor)
        assert mass.shape == (2560,)

    def test_apply_phase_rotation(self) -> None:
        mem = SemanticMemory(None, None)
        x = torch.randn(10)
        rotated = mem.apply_phase_rotation(x, theta=0.5)
        assert rotated.shape == x.shape
        # Verify norm is preserved (rotation is orthogonal)
        assert torch.allclose(x.norm(), rotated.norm(), atol=1e-5)

    def test_apply_phase_rotation_odd_dim(self) -> None:
        mem = SemanticMemory(None, None)
        x = torch.randn(11)  # odd dimension
        rotated = mem.apply_phase_rotation(x, theta=1.0)
        assert rotated.shape == x.shape

    def test_apply_phase_rotation_none(self) -> None:
        mem = SemanticMemory(None, None)
        assert mem.apply_phase_rotation(None, 0.5) is None

    def test_decay_vector_property(self) -> None:
        mem = SemanticMemory(None, None, gold_dimension=2560)
        mem.load_cache(device="cpu", dtype=torch.float32)
        dv = mem.decay_vector
        assert isinstance(dv, torch.Tensor)
        assert dv.shape == (2560,)
        assert dv.min() >= 0.60
        assert dv.max() <= 0.98

    def test_invalidate_cache(self) -> None:
        mem = SemanticMemory(None, None)
        mem.load_cache(device="cpu", dtype=torch.float32)
        assert mem._cached_L is not None
        mem.invalidate_cache()
        assert mem._cached_L is None

    def test_deform_manifold_no_recursion(self) -> None:
        mem = SemanticMemory(None, None)
        # Should not raise when no recursive thought is accumulated
        mem.deform_manifold(device="cpu", dtype=torch.float32)

    def test_feed_thought_accumulates(self) -> None:
        mem = SemanticMemory(None, None, gold_dimension=2560)
        mem.load_cache(device="cpu", dtype=torch.float32)
        v = torch.randn(2560)
        mem.feed_thought(v)
        assert mem._recursive_thought_accumulator is not None
        assert mem._recursive_thought_accumulator.shape == (2560,)


class TestAnchorStorage:
    def test_init_without_db(self) -> None:
        mem = SemanticMemory(None, None)
        storage = AnchorStorage(None, None, memory=mem)
        assert storage.conn is None

    def test_resolve_phase_jump_no_db(self) -> None:
        mem = SemanticMemory(None, None)
        storage = AnchorStorage(None, None, memory=mem)
        result = storage.resolve_phase_jump(torch.randn(2560))
        assert result == (None, None, None, None, None)

    def test_resolve_by_physical_path_no_shm(self) -> None:
        mem = SemanticMemory(None, None)
        storage = AnchorStorage(None, None, memory=mem)
        result = storage.resolve_by_physical_path("/nonexistent/path")
        assert result is None

    def test_upsert_anchor_no_db(self) -> None:
        mem = SemanticMemory(None, None)
        storage = AnchorStorage(None, None, memory=mem)
        result = storage.upsert_structural_anchor_from_mmap(
            "/tmp/test.py",
            "print('hello')",
        )
        assert result is None


class TestQuantumPhaseBridge:
    def test_init_no_db(self) -> None:
        bridge = QuantumPhaseBridge()
        # Without a real Postgres, conn may be None (caught in except)
        assert bridge.gold_dimension == 2560
        assert bridge.phi_ratio == 1.61803398875

    def test_sub_components_attached(self) -> None:
        bridge = QuantumPhaseBridge()
        assert hasattr(bridge, "mcp")
        assert hasattr(bridge, "memory")
        assert hasattr(bridge, "storage")

    def test_backward_compat_methods(self) -> None:
        bridge = QuantumPhaseBridge()
        for name in [
            "log_mcp_activity",
            "get_recent_mcp_activity",
            "fetch_mcp_context",
            "compute_mcp_gravity",
            "load_semantic_mass_cache",
            "compute_blake_semantic_mass",
            "apply_phase_rotation",
            "feed_recursive_thought",
            "deform_semantic_manifold",
            "resolve_phase_jump",
            "resolve_by_physical_path",
            "upsert_structural_anchor_from_mmap",
        ]:
            assert callable(getattr(bridge, name)), f"{name} not callable"
        assert hasattr(bridge, "decay_vector")

    def test_property_forwarding(self) -> None:
        bridge = QuantumPhaseBridge()
        bridge.memory._cached_L = torch.randn(5, 2560)
        assert bridge._cached_L is bridge.memory._cached_L

        bridge.memory._phase_theta = 1.5
        assert bridge._phase_theta == 1.5

        bridge.memory.apoptosis_threshold = 500
        assert bridge.apoptosis_threshold == 500

    def test_decay_vector_forwards(self) -> None:
        bridge = QuantumPhaseBridge()
        bridge.memory.load_cache(device="cpu", dtype=torch.float32)
        dv = bridge.decay_vector
        assert isinstance(dv, torch.Tensor)
        assert dv.shape == (2560,)
