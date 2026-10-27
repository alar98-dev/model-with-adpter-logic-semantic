"""Episodic memory domain controllers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from swa.db.base import BaseController
from swa.dimensions import MEMORY_DIM, to_memory


class EpisodicMemoryController(BaseController):
    """Stores and retrieves cognitive episodes with vector embeddings."""

    def store_episode(
        self,
        concept_group: str,
        content: str,
        embedding: Optional[List[float]] = None,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(content, (list, tuple)) and isinstance(embedding, str):
            concept_group, content, embedding = embedding, concept_group, content
        embedding = embedding if embedding is not None else kwargs.pop("embedding_2560", None)
        if embedding is None:
            raise ValueError("embedding is required")
        embedding = to_memory(embedding).tolist()
        payload = {
            "concept_group": concept_group,
            "content": content,
            "embedding_2560": embedding,
            "weight": weight,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO swa_episodic_episodes
                           (concept_group, content, embedding_2560, weight)
                           VALUES (%s, %s, %s, %s);""",
                        (concept_group, content, embedding, weight),
                    )
                    conn.commit()
            except Exception as exc:
                self._record_db_error("store_episode", exc)

        self._RAM_DATABASE.setdefault("swa_episodic_episodes", []).append(payload)
        return payload

    def search_similar(
        self, query_embedding: List[float], limit: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        query_embedding = to_memory(query_embedding).tolist()

        if self.is_db_active():
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    sql = """SELECT concept_group, content, weight,
                           1 - (embedding_2560 <=> %s::vector) AS similarity
                           FROM swa_episodic_episodes
                    """
                    params: list[Any] = [query_embedding]
                    if categories:
                        sql += " WHERE concept_group = ANY(%s)"
                        params.append(categories)
                    sql += " ORDER BY similarity DESC LIMIT %s;"
                    params.append(limit)
                    cur.execute(sql, params)
                    for row in cur.fetchall():
                        results.append(dict(row))
                return results
            except Exception as exc:
                self._record_db_error("search_similar", exc)

        # RAM fallback
        episodes = self._RAM_DATABASE.get("swa_episodic_episodes", [])
        scored = []
        allowed = set(categories or [])
        for ep in episodes:
            if allowed and ep.get("concept_group") not in allowed:
                continue
            sim = self.cosine_similarity(
                to_memory(query_embedding).tolist(), ep.get("embedding_2560", [])
            )
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def query_similar_episodes(
        self, query_embedding: List[float], limit: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.search_similar(query_embedding, limit=limit, categories=categories)

    @staticmethod
    def serialize_episode(episode: Dict[str, Any]) -> str:
        import json
        return json.dumps(
            {
                "concept_group": episode.get("concept_group", "GENERAL"),
                "content": episode.get("content", ""),
                "embedding_2560": to_memory(episode.get("embedding_2560", [])).tolist(),
                "weight": float(episode.get("weight", 1.0)),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def deserialize_episode(payload: str | Dict[str, Any]) -> Dict[str, Any]:
        import json
        value = json.loads(payload) if isinstance(payload, str) else dict(payload)
        value["embedding_2560"] = to_memory(value.get("embedding_2560", [])).tolist()
        value["weight"] = float(value.get("weight", 1.0))
        return value


class LatentSignatureController(BaseController):
    """Latent signature placeholder."""


class SequenceLenController(BaseController):
    """Sequence length tracking placeholder."""


class DynamicTemperatureController(BaseController):
    """Dynamic temperature tracking placeholder."""


class PhaseShifterController(BaseController):
    """Phase shifter state placeholder."""
