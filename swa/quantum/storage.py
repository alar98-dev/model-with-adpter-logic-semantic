from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any

import torch


class AnchorStorage:
    """PostgreSQL persistence for semantic anchors and phase-jump resolution.

    Handles pgvector queries, SHM-based O(1) lookups, and structural anchor
    upserts via ``BlakeNativeCrawler``.
    """

    def __init__(
        self,
        conn: Any,
        cursor: Any,
        memory: Any,
        gold_dimension: int = 2560,
        origin_log_path: str | None = None,
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.memory = memory  # SemanticMemory instance (for cache invalidation and gold_dimension)
        self.gold_dimension = gold_dimension
        self.origin_log_path = origin_log_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "structural_origins.jsonl",
        )
        self._blake_crawler: Any = None

    # ------------------------------------------------------------------
    # pgvector phase-jump resolution
    # ------------------------------------------------------------------

    def resolve_phase_jump(
        self, current_steer_vector: torch.Tensor
    ) -> tuple[Any, ...]:
        """Cosine-similarity search in pgvector with gravitational attraction."""
        if self.conn is None:
            return (None, None, None, None, None)

        x_tensor = current_steer_vector.detach().cpu().view(-1)
        if x_tensor.size(0) < self.gold_dimension:
            import torch.nn.functional as F

            x_tensor = F.pad(
                x_tensor,
                (0, self.gold_dimension - x_tensor.size(0)),
                "constant",
                0.0,
            )
        elif x_tensor.size(0) > self.gold_dimension:
            x_tensor = x_tensor[: self.gold_dimension]

        vec_list = x_tensor.numpy().tolist()

        query = """
            SELECT id, file_path, concept_group,
                   (embedding_2560 <=> %s::vector) AS cosine_distance,
                   raw_content
            FROM blake_semantic_anchors
            WHERE embedding_2560 IS NOT NULL
            ORDER BY (1.0 - (embedding_2560 <=> %s::vector)) * COALESCE(weight, 1.0) DESC
            LIMIT 1;
        """
        try:
            self.cursor.execute(query, (vec_list, vec_list))
            row = self.cursor.fetchone()
            if row:
                unique_id, file_path, concept_group, distance, raw_content = row
                return (unique_id, file_path, concept_group, distance, raw_content)
        except Exception:
            self.conn.rollback()
        return (None, None, None, None, None)

    # ------------------------------------------------------------------
    # SHM-based physical path lookup
    # ------------------------------------------------------------------

    def resolve_by_physical_path(self, physical_path: str) -> tuple | None:
        """O(1) lookup in the SHM manifold index, before pgvector."""
        try:
            from swa.vector_manifold import VectorManifold

            interdim_id = VectorManifold.resolve_path(physical_path)
            if not interdim_id:
                return None
            node = VectorManifold.resolve_interdim(interdim_id)
            if node:
                return (interdim_id, node.physical_path, node.concept, 0.0)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Structural anchor upsert
    # ------------------------------------------------------------------

    def upsert_structural_anchor_from_mmap(
        self,
        file_path: str,
        content_sample: str,
        vector_640: list | None = None,
    ) -> str | None:
        """Translate mmap touch (640D) into a Blake anchor (2560D) stored in pgvector."""
        if self.conn is None:
            return None

        try:
            from swa.file_crawler import BlakeNativeCrawler
        except Exception:
            return None

        if self._blake_crawler is None:
            self._blake_crawler = BlakeNativeCrawler()

        crawler = self._blake_crawler
        sample = (content_sample or "").strip()
        if not sample:
            return None

        structural_type = crawler.classify_paragraph(
            sample, os.path.basename(file_path)
        )
        use_real = os.environ.get("SWA_RUNTIME_REAL_EMBEDDINGS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if use_real:
            try:
                v_2560 = crawler.generate_real_2560_vector(sample)
            except Exception:
                v_2560 = crawler.generate_native_2560_vector(sample)
        else:
            v_2560 = crawler.generate_native_2560_vector(sample)

        fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
        interdim_id = hashlib.blake2b(
            f"{file_path}:mmap:{fingerprint}".encode("utf-8"), digest_size=16
        ).hexdigest()
        composite_path = f"mmap://{interdim_id}#{file_path}"

        weight = math.log(len(sample) + 1)
        if structural_type == "CODE_BLOCK":
            weight *= 1.5
        elif structural_type == "CONFIG_SETTING":
            weight *= 1.2

        try:
            self.cursor.execute(
                """
                INSERT INTO blake_semantic_anchors
                    (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE
                SET raw_content = EXCLUDED.raw_content,
                    embedding_2560 = EXCLUDED.embedding_2560,
                    weight = EXCLUDED.weight,
                    concept_group = EXCLUDED.concept_group,
                    signature_hash = EXCLUDED.signature_hash;
                """,
                (
                    composite_path,
                    structural_type,
                    interdim_id,
                    sample[:1000],
                    v_2560,
                    weight,
                ),
            )
            self.conn.commit()
            self.memory.invalidate_cache()

            os.makedirs(os.path.dirname(self.origin_log_path), exist_ok=True)
            with open(self.origin_log_path, "a", encoding="utf-8") as origin_log:
                origin_log.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "interdim_id": interdim_id,
                            "origin": file_path,
                            "concept": structural_type,
                            "vector_640_preview": (vector_640 or [])[:8],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return interdim_id
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None
