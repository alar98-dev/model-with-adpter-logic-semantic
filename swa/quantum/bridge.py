from __future__ import annotations

import os
from typing import Any

import psycopg2
import torch

from swa.quantum.mcp import MCPActivityTracker
from swa.quantum.memory import SemanticMemory
from swa.quantum.storage import AnchorStorage


class QuantumPhaseBridge:
    """Cognitive bridge between MCP activity, semantic memory, and pgvector anchors.

    Composes three specialised sub-components:

    * ``mcp`` — :class:`MCPActivityTracker` — MCP tool logging, context, gravity
    * ``memory`` — :class:`SemanticMemory` — in-RAM cache, decay, rotation, recursion
    * ``storage`` — :class:`AnchorStorage` — pgvector phase jumps, anchor upserts
    """

    def __init__(self, dsn: str = "dbname=swa_matrix user=alifer host=localhost") -> None:
        password = os.environ.get("SWA_DB_PASSWORD")
        if password:
            dsn += f" password={password} host=localhost"

        try:
            self.conn = psycopg2.connect(dsn)
            self.cursor = self.conn.cursor()

            # Bootstrap buffer-log table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS mcp_blake_buffer_logs (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    tool_name VARCHAR(255),
                    parameters JSONB,
                    ipv6_signature VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()

            self.gold_dimension = 2560
            self.phi_ratio = 1.61803398875

            self.mcp = MCPActivityTracker(self.conn, self.cursor)
            self.memory = SemanticMemory(
                self.conn,
                self.cursor,
                gold_dimension=self.gold_dimension,
                phi_ratio=self.phi_ratio,
            )
            self.storage = AnchorStorage(
                self.conn,
                self.cursor,
                memory=self.memory,
                gold_dimension=self.gold_dimension,
            )

        except Exception as e:
            self.conn = None
            self.cursor = None
            self.mcp = MCPActivityTracker(None, None)
            self.memory = SemanticMemory(None, None)
            self.storage = AnchorStorage(None, None, memory=self.memory)

    # ------------------------------------------------------------------
    # Properties forwarded from SemanticMemory (backward compat)
    # ------------------------------------------------------------------

    @property
    def _recursive_thought_accumulator(self) -> torch.Tensor | None:
        return self.memory._recursive_thought_accumulator

    @_recursive_thought_accumulator.setter
    def _recursive_thought_accumulator(self, value: torch.Tensor | None) -> None:
        self.memory._recursive_thought_accumulator = value

    @property
    def _recursion_decay(self) -> float:
        return self.memory._recursion_decay

    @_recursion_decay.setter
    def _recursion_decay(self, value: float) -> None:
        self.memory._recursion_decay = value

    @property
    def _phase_theta(self) -> float:
        return self.memory._phase_theta

    @_phase_theta.setter
    def _phase_theta(self, value: float) -> None:
        self.memory._phase_theta = value

    @property
    def _cached_L(self) -> torch.Tensor | None:
        return self.memory._cached_L

    @_cached_L.setter
    def _cached_L(self, value: torch.Tensor | None) -> None:
        self.memory._cached_L = value

    @property
    def _cached_w(self) -> torch.Tensor | None:
        return self.memory._cached_w

    @_cached_w.setter
    def _cached_w(self, value: torch.Tensor | None) -> None:
        self.memory._cached_w = value

    @property
    def _anchor_irrelevance(self) -> torch.Tensor | None:
        return self.memory._anchor_irrelevance

    @_anchor_irrelevance.setter
    def _anchor_irrelevance(self, value: torch.Tensor | None) -> None:
        self.memory._anchor_irrelevance = value

    @property
    def apoptosis_threshold(self) -> int:
        return self.memory.apoptosis_threshold

    @apoptosis_threshold.setter
    def apoptosis_threshold(self, value: int) -> None:
        self.memory.apoptosis_threshold = value

    @property
    def current_jitter_mag(self) -> float:
        return self.memory.current_jitter_mag

    @current_jitter_mag.setter
    def current_jitter_mag(self, value: float) -> None:
        self.memory.current_jitter_mag = value

    @property
    def decay_vector(self) -> torch.Tensor:
        return self.memory.decay_vector

    # ------------------------------------------------------------------
    # MCP delegation
    # ------------------------------------------------------------------

    def log_mcp_activity(
        self,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        ipv6_signature: str = "unknown",
    ) -> None:
        self.mcp.log_activity(session_id, tool_name, parameters, ipv6_signature)

    def get_recent_mcp_activity(self, limit: int = 3) -> str:
        return self.mcp.fetch_context(limit)

    def fetch_mcp_context(
        self, limit: int = 10, context: list[str] | None = None
    ) -> str:
        return self.mcp.fetch_mcp_context(limit, context)

    def compute_mcp_gravity(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        return self.mcp.compute_gravity(device, dtype)

    # ------------------------------------------------------------------
    # Semantic memory delegation
    # ------------------------------------------------------------------

    def load_semantic_mass_cache(
        self,
        d_model: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.memory.load_cache(d_model, device, dtype)

    def compute_blake_semantic_mass(
        self,
        x: torch.Tensor,
        sigma: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.memory.compute_mass(x, sigma, device, dtype)

    def apply_phase_rotation(
        self, x: torch.Tensor, theta: float
    ) -> torch.Tensor:
        return self.memory.apply_phase_rotation(x, theta)

    def feed_recursive_thought(
        self, current_thought_vector: torch.Tensor
    ) -> None:
        self.memory.feed_thought(current_thought_vector)

    def deform_semantic_manifold(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.memory.deform_manifold(device, dtype)

    # ------------------------------------------------------------------
    # Storage delegation
    # ------------------------------------------------------------------

    def resolve_phase_jump(
        self, current_steer_vector: torch.Tensor
    ) -> tuple[Any, ...]:
        return self.storage.resolve_phase_jump(current_steer_vector)

    def resolve_by_physical_path(
        self, physical_path: str
    ) -> tuple | None:
        return self.storage.resolve_by_physical_path(physical_path)

    def upsert_structural_anchor_from_mmap(
        self,
        file_path: str,
        content_sample: str,
        vector_640: list | None = None,
    ) -> str | None:
        return self.storage.upsert_structural_anchor_from_mmap(
            file_path, content_sample, vector_640
        )
