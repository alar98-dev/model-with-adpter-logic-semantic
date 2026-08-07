"""System state and eventual embedding domain controllers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from swa.db.base import BaseController


class MatrixStateController(BaseController):
    """Persists and retrieves the global matrix system state."""

    def save_state(self, state_vector: List[float], signature: str) -> Dict[str, Any]:
        payload = {
            "state_vector": state_vector,
            "signature": signature,
            "timestamp": time.time(),
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO matrix_system_state (state_vector, signature, timestamp)
                           VALUES (%s, %s, NOW())
                           ON CONFLICT (signature) DO UPDATE
                           SET state_vector = EXCLUDED.state_vector,
                               timestamp = NOW();""",
                        (state_vector, signature),
                    )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE.setdefault("matrix_system_state", []).append(payload)
        return payload

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        if self.is_db_active():
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """SELECT state_vector, signature, timestamp
                           FROM matrix_system_state
                           ORDER BY timestamp DESC LIMIT 1;"""
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
            except Exception:
                pass

        states = self._RAM_DATABASE.get("matrix_system_state", [])
        return states[-1] if states else None


class EventualEmbeddingController(BaseController):
    """Manages eventual consistency for embeddings across states."""

    def store_embedding_relationship(
        self, source: str, target: str, embedding: List[float], weight: float = 1.0
    ) -> Dict[str, Any]:
        payload = {
            "source": source,
            "target": target,
            "embedding": embedding,
            "weight": weight,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO swa_eventual_embeddings
                           (source_id, target_id, embedding_2560, weight)
                           VALUES (%s, %s, %s, %s);""",
                        (source, target, embedding, weight),
                    )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE.setdefault("swa_eventual_embeddings", []).append(payload)
        return payload
