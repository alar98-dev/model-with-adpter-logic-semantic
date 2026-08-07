"""Tests for domain-specific database controllers."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from swa.db.base import BaseController
from swa.db.memory import MemoryPageController
from swa.db.cpu import CpuAffinityController, PCoreSchedulerController
from swa.db.io import PostgresBridgeController
from swa.db.anchors import FileCrawlerController
from swa.db.episodic import EpisodicMemoryController
from swa.db.state import MatrixStateController
from swa.db.async_worker import DBAsyncWorker
from swa.db.projector import ProceduralProjector


# ── Base ──────────────────────────────────────────────────────────────────

class TestBaseController:
    def test_compute_state_hash(self) -> None:
        ctrl = BaseController()
        h1 = ctrl.compute_state_hash("hello")
        h2 = ctrl.compute_state_hash("hello")
        assert h1 == h2
        assert len(h1) == 16

    def test_cosine_similarity_identical(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert BaseController.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        assert BaseController.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_cosine_similarity_zero_norm(self) -> None:
        assert BaseController.cosine_similarity([0, 0], [1, 1]) == pytest.approx(0.0)


# ── Memory ────────────────────────────────────────────────────────────────

class TestMemoryPageController:
    def test_log_memory_pages_returns_dict(self) -> None:
        ctrl = MemoryPageController()
        result = ctrl.log_memory_pages()
        assert isinstance(result, dict)
        assert "total_ram_gb" in result
        assert "percent_used" in result
        assert result["percent_used"] > 0


# ── CPU ───────────────────────────────────────────────────────────────────

class TestCpuAffinityController:
    def test_save_affinity_state(self) -> None:
        ctrl = CpuAffinityController()
        result = ctrl.save_affinity_state(0, [0, 1, 2, 3])
        assert result["core_id"] == 0
        assert result["logical_processors"] == "0,1,2,3"
        assert len(result["semantic_trace"]) == 2560


class TestPCoreSchedulerController:
    def test_log_scheduler_state(self) -> None:
        ctrl = PCoreSchedulerController()
        result = ctrl.log_scheduler_state(45.0, 62.0, 8, 0, "Normal operation")
        assert result["utilization_average"] == 45.0
        assert result["active_cores"] == 8


# ── IO ────────────────────────────────────────────────────────────────────

class TestPostgresBridgeController:
    def test_verify_schema_no_db(self) -> None:
        ctrl = PostgresBridgeController(db_uri="dbname=nonexistent")
        assert ctrl.verify_schema_and_migrations() is False

    def test_check_vector_extension_no_db(self) -> None:
        ctrl = PostgresBridgeController(db_uri="dbname=nonexistent")
        result = ctrl.check_vector_extension()
        assert result["pgvector_available"] is False


# ── Anchors ───────────────────────────────────────────────────────────────

class TestFileCrawlerController:
    def test_register_anchor(self) -> None:
        ctrl = FileCrawlerController()
        result = ctrl.register_anchor("/tmp/test.txt", [0.1] * 2560, "abc123")
        assert result["file_path"] == "/tmp/test.txt"
        assert result["signature_hash"] == "abc123"


# ── Episodic ──────────────────────────────────────────────────────────────

class TestEpisodicMemoryController:
    def test_store_episode(self) -> None:
        ctrl = EpisodicMemoryController()
        result = ctrl.store_episode("test", "hello world", [0.5] * 2560)
        assert result["concept_group"] == "test"
        assert result["weight"] == 1.0

    def test_search_similar_no_db(self) -> None:
        ctrl = EpisodicMemoryController(db_uri="dbname=nonexistent")
        results = ctrl.search_similar([0.5] * 2560, limit=3)
        assert isinstance(results, list)


# ── State ─────────────────────────────────────────────────────────────────

class TestMatrixStateController:
    def test_save_and_get_latest(self) -> None:
        ctrl = MatrixStateController()
        ctrl.save_state([0.1, 0.2], "sig-001")
        latest = ctrl.get_latest_state()
        assert latest is not None
        assert latest["signature"] == "sig-001"


# ── Async Worker ──────────────────────────────────────────────────────────

class TestDBAsyncWorker:
    def test_enqueue_and_flush(self) -> None:
        worker = DBAsyncWorker(flush_interval=0.1)
        collected: list = []

        def handler(batch):
            collected.extend(batch)

        worker.set_handler(handler)
        worker.enqueue({"msg": "hello"})
        worker.flush()
        assert len(collected) == 1
        assert collected[0]["msg"] == "hello"

    def test_stop(self) -> None:
        worker = DBAsyncWorker()
        worker.start()
        worker.stop()
        assert not worker._running


# ── Projector ─────────────────────────────────────────────────────────────

class TestProceduralProjector:
    def test_project_search(self) -> None:
        result = ProceduralProjector.project("buscar arquivo de configuração")
        assert result["action"] == "search"
        assert "configuração" in result["target"]

    def test_project_store(self) -> None:
        result = ProceduralProjector.project("registrar novo episódio")
        assert result["action"] == "store"

    def test_project_unknown(self) -> None:
        result = ProceduralProjector.project("olá mundo")
        assert result["action"] == "unknown"
