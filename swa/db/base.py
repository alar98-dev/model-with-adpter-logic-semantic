"""Base controller with PostgreSQL / in-memory dual-mode support."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor


class BaseController:
    """Dual-mode database controller: PostgreSQL + in-memory fallback.

    All domain controllers inherit from this class to get automatic
    connection management and RAM-only degradation when PostgreSQL is
    unreachable.
    """

    _RAM_DATABASE: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, db_uri: str = "dbname=swa_matrix host=/var/run/postgresql user=alifer"):
        self.db_uri = db_uri
        self._db_active: Optional[bool] = None

    # ── Connection management ───────────────────────────────────────────

    def _get_connection(self):
        """Open a PostgreSQL connection or raise ``OperationalError``."""
        try:
            conn = psycopg2.connect(self.db_uri, cursor_factory=RealDictCursor)
            self._db_active = True
            return conn
        except Exception as exc:
            self._db_active = False
            raise psycopg2.OperationalError(f"Database connection failed: {exc}") from exc

    def is_db_active(self) -> bool:
        """Lazily probe PostgreSQL connectivity."""
        if self._db_active is None:
            try:
                conn = self._get_connection()
                conn.close()
                self._db_active = True
            except Exception:
                self._db_active = False
        return self._db_active

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def compute_state_hash(data_str: str) -> str:
        """Short SHA-256 fingerprint."""
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors (NumPy)."""
        arr_a = np.array(a, dtype=np.float32)
        arr_b = np.array(b, dtype=np.float32)
        dot = np.dot(arr_a, arr_b)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _normalized_trace_vector(seed: int, dim: int = 2560) -> List[float]:
        """Deterministic normalised vector for a given seed."""
        np.random.seed(seed)
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
