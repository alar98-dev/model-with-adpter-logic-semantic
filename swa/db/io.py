"""I/O and hardware bridge domain controllers."""

from __future__ import annotations

from typing import Any, Dict

from swa.db.base import BaseController


class UnixSocketController(BaseController):
    """Unix Domain Socket monitoring placeholder."""


class DeviceHalController(BaseController):
    """Physical sensor telemetry placeholder."""


class PostgresBridgeController(BaseController):
    """PostgreSQL connectivity, extensions, and schema migrations."""

    def verify_schema_and_migrations(self) -> bool:
        if not self.is_db_active():
            return False

        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute(
                    """SELECT column_name FROM information_schema.columns
                       WHERE table_name='blake_semantic_anchors' AND column_name='weight';"""
                )
                if not cur.fetchone():
                    cur.execute("ALTER TABLE blake_semantic_anchors ADD COLUMN weight REAL DEFAULT 1.0;")

                cur.execute(
                    """CREATE TABLE IF NOT EXISTS swa_episodic_episodes (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        concept_group VARCHAR(64) NOT NULL,
                        content TEXT NOT NULL,
                        embedding_2560 vector(2560) NOT NULL,
                        weight REAL DEFAULT 1.0
                    );"""
                )
                conn.commit()
            return True
        except Exception:
            return False

    def check_vector_extension(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"pgvector_available": False, "version": None}
        if not self.is_db_active():
            return result

        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector';")
                row = cur.fetchone()
                if row:
                    result["pgvector_available"] = True
                    result["version"] = row["extversion"]
        except Exception:
            pass
        return result
