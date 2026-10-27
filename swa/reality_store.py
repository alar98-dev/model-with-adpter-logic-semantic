"""Append-only binary reality store with optional PostgreSQL relations."""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import blake3
from swa.context_domain import (
    ContextIdentity,
    ContextEmbedding,
    ContextObservation,
    ContextRelation,
    GravityField,
)


class RealityStore:
    """Persists world events and tensors without putting tensor bytes in JSON."""

    MAGIC = b"AEONR001"
    HEADER = struct.Struct("<8sIdQ32s")

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        world_key: str = "aeon-main",
        db_uri: str = "dbname=swa_matrix host=/var/run/postgresql user=alifer",
    ):
        self.root = Path(root or os.environ.get(
            "SWA_REALITY_ROOT",
            os.path.join(os.getcwd(), ".aeon-reality"),
        ))
        self._world_key = str(world_key)
        self.db_uri = db_uri
        self.events_path = self.root / "events.bin"
        self.tensors_path = self.root / "tensors"
        self._lock = threading.RLock()
        self._logger = logging.getLogger("swa.reality")
        self._world_id: Optional[int] = None
        self._last_event_id: Optional[int] = None
        self._last_tensor_id: Optional[int] = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensors_path.mkdir(parents=True, exist_ok=True)
        self._last_event_hash = self._load_event_head()
        self._ensure_world()

    @property
    def world_id(self) -> Optional[int]:
        """Identificador relacional do mundo, quando o PostgreSQL está ativo."""
        return self._world_id

    @property
    def world_key(self) -> str:
        """Chave estável usada para relacionar todos os registros do mundo."""
        return self._world_key

    @property
    def storage_uri(self) -> str:
        """Local físico atual do journal e dos tensores."""
        return str(self.root)

    @property
    def events_uri(self) -> str:
        """Arquivo binário append-only de eventos."""
        return str(self.events_path)

    @property
    def tensor_uri(self) -> str:
        """Diretório dos blobs tensoriais deduplicados."""
        return str(self.tensors_path)

    @property
    def relational(self) -> bool:
        """Indica se o mundo possui vínculo ativo no PostgreSQL."""
        return self._world_id is not None

    @property
    def last_event_id(self) -> Optional[int]:
        """ID relacional do último evento persistido, quando disponível."""
        return self._last_event_id

    @property
    def last_tensor_id(self) -> Optional[int]:
        """ID relacional do último tensor persistido, quando disponível."""
        return self._last_tensor_id

    def _ensure_world(self) -> None:
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_reality_worlds
                            (world_key, storage_uri, root_hash)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (world_key) DO UPDATE
                            SET storage_uri = EXCLUDED.storage_uri,
                                updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        (self.world_key, str(self.root), self._root_hash()),
                    )
                    row = cur.fetchone()
                    if row:
                        self._world_id = int(row[0])
        except Exception as exc:
            self._logger.warning("reality database unavailable: %s", exc)

    def _root_hash(self) -> str:
        if not self.events_path.exists():
            return blake3.blake3(b"").hexdigest()
        return blake3.blake3(self.events_path.read_bytes()).hexdigest()

    def _load_event_head(self) -> Optional[str]:
        """Read the last journal digest without loading the journal into RAM."""
        if not self.events_path.exists():
            return None
        head: Optional[str] = None
        try:
            with self.events_path.open("rb") as handle:
                while True:
                    header = handle.read(self.HEADER.size)
                    if not header:
                        break
                    if len(header) != self.HEADER.size:
                        break
                    magic, _version, _timestamp, length, digest = self.HEADER.unpack(header)
                    if magic != self.MAGIC:
                        break
                    body = handle.read(length)
                    if len(body) != length:
                        break
                    head = digest.hex()
        except OSError:
            return None
        return head

    @staticmethod
    def _event_digest(body: bytes, previous_hash: Optional[str]) -> bytes:
        hasher = blake3.blake3()
        hasher.update((previous_hash or "").encode("ascii"))
        hasher.update(body)
        return hasher.digest()

    def append_event(self, event_type: str, payload: dict[str, Any], *, wave: int = 0) -> str:
        body = json.dumps(
            {"event_type": event_type, "wave": int(wave), "payload": payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with self._lock:
            previous_hash = self._last_event_hash
            digest = self._event_digest(body, previous_hash)
            record = self.HEADER.pack(self.MAGIC, 1, time.time(), len(body), digest) + body
            with self.events_path.open("ab") as handle:
                handle.write(record)
                handle.flush()
                os.fsync(handle.fileno())
            self._last_event_hash = digest.hex()
            self._last_event_id = self._persist_event(
                event_type, payload, wave, digest.hex(), previous_hash
            )
        return digest.hex()

    def append_tensor(self, tensor: Any, *, wave: int = 0, entity_id: Optional[int] = None) -> str:
        array = np.asarray(tensor, dtype=np.float32)
        raw = np.ascontiguousarray(array).tobytes()
        digest = blake3.blake3(raw).hexdigest()
        self._last_tensor_id = None
        path = self.tensors_path / f"{digest}.f32"
        with self._lock:
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
        if self._world_id is not None:
            try:
                import psycopg2

                with psycopg2.connect(self.db_uri) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO aeon_reality_tensor_states
                                (world_id, entity_id, wave, dimension, dtype,
                                 storage_uri, content_hash, norm)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (world_id, content_hash) DO NOTHING
                            RETURNING id
                            """,
                            (
                                self._world_id,
                                entity_id,
                                int(wave),
                                int(array.size),
                                "float32",
                                str(path),
                                digest,
                                float(np.linalg.norm(array)),
                            ),
                        )
                        row = cur.fetchone()
                        if row:
                            self._last_tensor_id = int(row[0])
                        else:
                            cur.execute(
                                """
                                SELECT id FROM aeon_reality_tensor_states
                                WHERE world_id = %s AND content_hash = %s
                                """,
                                (self._world_id, digest),
                            )
                            row = cur.fetchone()
                            if row:
                                self._last_tensor_id = int(row[0])
            except Exception as exc:
                self._logger.warning("tensor relation unavailable: %s", exc)
        return digest

    def append_wave_state(
        self,
        conversation_key: str,
        wave_index: int,
        *,
        parent_wave_id: Optional[int] = None,
        phase: str = "unknown",
        metrics: Optional[dict[str, Any]] = None,
        state_tensor_id: Optional[int] = None,
        event_id: Optional[int] = None,
        merge_metrics: bool = False,
    ) -> Optional[int]:
        """Persiste uma onda e sua ligação temporal com a onda anterior.

        `merge_metrics=True` combina as métricas novas com as já gravadas da onde
        (a observação da metarealidade acrescenta `anchor_vec`/`cos_sim` sem
        apagar os escalares de fissão) e preserva o `event_id` existente.
        """
        if self._world_id is None:
            return None
        if phase not in {"inicio", "meio", "fim", "deriva", "unknown"}:
            raise ValueError(f"invalid wave phase: {phase}")
        if int(wave_index) < 0:
            raise ValueError("wave_index must be non-negative")
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    if merge_metrics:
                        cur.execute(
                        """
                        INSERT INTO aeon_reality_waves
                            (world_id, conversation_key, wave_index,
                             parent_wave_id, phase, metrics,
                             state_tensor_id, event_id)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (world_id, conversation_key, wave_index)
                        DO UPDATE SET
                            parent_wave_id = EXCLUDED.parent_wave_id,
                            phase = EXCLUDED.phase,
                            metrics = COALESCE(
                                (EXCLUDED.metrics || aeon_reality_waves.metrics),
                                EXCLUDED.metrics,
                                aeon_reality_waves.metrics
                            ),
                            state_tensor_id = COALESCE(
                                EXCLUDED.state_tensor_id,
                                aeon_reality_waves.state_tensor_id
                            ),
                            event_id = COALESCE(
                                EXCLUDED.event_id,
                                aeon_reality_waves.event_id
                            )
                        RETURNING id
                        """,
                        (
                            self._world_id,
                            str(conversation_key),
                            int(wave_index),
                            parent_wave_id,
                            phase,
                            json.dumps(metrics or {}, ensure_ascii=False),
                            state_tensor_id,
                            event_id,
                        ),
                        )
                        row = cur.fetchone()
                        return int(row[0]) if row else None
                    else:
                        cur.execute(
                        """
                        INSERT INTO aeon_reality_waves
                            (world_id, conversation_key, wave_index,
                             parent_wave_id, phase, metrics,
                             state_tensor_id, event_id)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (world_id, conversation_key, wave_index)
                        DO UPDATE SET
                            parent_wave_id = EXCLUDED.parent_wave_id,
                            phase = EXCLUDED.phase,
                            metrics = EXCLUDED.metrics,
                            state_tensor_id = EXCLUDED.state_tensor_id,
                            event_id = EXCLUDED.event_id
                        RETURNING id
                        """,
                        (
                            self._world_id,
                            str(conversation_key),
                            int(wave_index),
                            parent_wave_id,
                            phase,
                            json.dumps(metrics or {}, ensure_ascii=False),
                            state_tensor_id,
                            event_id,
                        ),
                    )
                        row = cur.fetchone()
                        return int(row[0]) if row else None
        except (OSError, TypeError, ValueError) as exc:
            self._logger.warning("wave relation unavailable: %s", exc)
        except Exception as exc:
            self._logger.warning("wave relation persistence failed: %s", exc)
        return None

    def append_experience(self, experience: Any) -> Optional[int]:
        """Persist measured experience metrics linked to a conversation wave."""
        if self._world_id is None:
            return None

        metrics = dict(getattr(experience, "metrics", {}) or {})
        conversation_key = str(getattr(experience, "conversation_key"))
        wave_index = int(getattr(experience, "wave_index"))
        outcome = str(getattr(experience, "outcome", "unknown"))
        wave_id = getattr(experience, "wave_id", None)
        if wave_index < 0:
            raise ValueError("wave_index must be non-negative")
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_reality_experiences
                            (world_id, conversation_key, wave_index,
                             wave_id, metrics, outcome)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (world_id, conversation_key, wave_index)
                        DO UPDATE SET
                            wave_id = EXCLUDED.wave_id,
                            metrics = EXCLUDED.metrics,
                            outcome = EXCLUDED.outcome
                        RETURNING id
                        """,
                        (
                            self._world_id,
                            conversation_key,
                            wave_index,
                            wave_id,
                            json.dumps(metrics, ensure_ascii=False),
                            outcome,
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except (OSError, TypeError, ValueError) as exc:
            self._logger.warning("experience relation unavailable: %s", exc)
        except Exception as exc:
            self._logger.warning("experience persistence failed: %s", exc)
        return None

    def register_context(
        self,
        identity: ContextIdentity,
        embedding: Optional[ContextEmbedding] = None,
    ) -> Optional[int]:
        """Register a content-addressed context and its optional embedding."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_contexts
                            (world_id, path, content_hash, content_type, version)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (world_id, content_hash) DO UPDATE SET
                            path = EXCLUDED.path,
                            content_type = EXCLUDED.content_type,
                            version = EXCLUDED.version,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        (self._world_id, identity.path, identity.content_hash,
                         identity.content_type, identity.version),
                    )
                    context_id = int(cur.fetchone()[0])
                    if embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO aeon_context_embeddings
                                (world_id, context_id, model_name, dimensions,
                                 embedding, norm)
                            VALUES (%s, %s, %s, %s, %s::vector, %s)
                            ON CONFLICT (context_id, model_name) DO UPDATE SET
                                dimensions = EXCLUDED.dimensions,
                                embedding = EXCLUDED.embedding,
                                norm = EXCLUDED.norm
                            """,
                            (
                                self._world_id, context_id, embedding.model_name,
                                embedding.dimensions, list(embedding.values), 1.0,
                            ),
                        )
                    return context_id
        except Exception as exc:
            self._logger.warning("context persistence failed: %s", exc)
            return None

    def append_gravity_field(self, field: GravityField) -> Optional[int]:
        """Persist the contextual gravity field used by a conversation."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_gravity_fields
                            (world_id, conversation_key, objective_vector,
                             strength, context_hashes, aligned_tokens, opposed_tokens)
                        VALUES (%s, %s, %s::vector, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            self._world_id, field.conversation_key,
                            list(field.objective_vector), field.strength,
                            json.dumps(list(field.context_hashes)),
                            json.dumps(field.aligned_tokens),
                            json.dumps(field.opposed_tokens),
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("gravity field persistence failed: %s", exc)
            return None

    def append_context_observation(self, observation: ContextObservation) -> Optional[int]:
        """Persist semantic feedback for a context and wave."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_context_observations
                            (world_id, context_hash, conversation_key, wave_index,
                             phase, semantic_distance, convergence, decision, metrics)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            self._world_id, observation.context_hash,
                            observation.conversation_key, observation.wave_index,
                            observation.phase, observation.semantic_distance,
                            observation.convergence, observation.decision,
                            json.dumps(observation.metrics, ensure_ascii=False),
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("context observation persistence failed: %s", exc)
            return None

    def append_context_relation(self, relation: ContextRelation) -> Optional[int]:
        """Persist a typed relation between two content-addressed contexts."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_context_relations
                            (world_id, source_hash, target_hash, relation_type,
                             similarity, weight)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (world_id, source_hash, target_hash, relation_type)
                        DO UPDATE SET similarity = EXCLUDED.similarity,
                                      weight = EXCLUDED.weight
                        RETURNING id
                        """,
                        (
                            self._world_id, relation.source_hash,
                            relation.target_hash, relation.relation_type,
                            relation.similarity, relation.weight,
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("context relation persistence failed: %s", exc)
            return None

    def create_task(
        self,
        task_key: str,
        title: str,
        *,
        description: str = "",
        priority: int = 0,
        parent_task_id: Optional[int] = None,
    ) -> Optional[int]:
        """Create or retrieve a persistent task for this reality."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_reality_tasks
                            (world_id, parent_task_id, task_key, title,
                             description, priority)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (world_id, task_key) DO UPDATE
                            SET title = EXCLUDED.title,
                                description = EXCLUDED.description,
                                priority = EXCLUDED.priority,
                                updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        (
                            self._world_id,
                            parent_task_id,
                            task_key,
                            title,
                            description,
                            int(priority),
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("task creation unavailable: %s", exc)
            return None

    def update_task(self, task_key: str, status: str, *, completion_event_id: Optional[int] = None) -> bool:
        """Advance a task through the constrained persistent state machine."""
        if status not in {"pending", "in_progress", "blocked", "done", "cancelled"}:
            raise ValueError(f"invalid task status: {status}")
        if self._world_id is None:
            return False
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE aeon_reality_tasks
                        SET status = %s,
                            completion_event_id = COALESCE(%s, completion_event_id),
                            completed_at = CASE WHEN %s = 'done'
                                THEN CURRENT_TIMESTAMP ELSE completed_at END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE world_id = %s AND task_key = %s
                        """,
                        (status, completion_event_id, status, self._world_id, task_key),
                    )
                    return cur.rowcount == 1
        except Exception as exc:
            self._logger.warning("task update unavailable: %s", exc)
            return False

    def ready_tasks(self, *, limit: int = 32) -> list[dict[str, Any]]:
        """Return pending tasks ordered by priority."""
        if self._world_id is None:
            return []
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            with psycopg2.connect(self.db_uri, cursor_factory=RealDictCursor) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, task_key, title, description, status, priority
                        FROM aeon_reality_tasks
                        WHERE world_id = %s AND status = 'pending'
                        ORDER BY priority DESC, id ASC
                        LIMIT %s
                        """,
                        (self._world_id, max(1, min(256, int(limit)))),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            self._logger.warning("task query unavailable: %s", exc)
            return []

    def wave_series(
        self,
        *,
        conversation_key: Optional[str] = None,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Tábua dos reinados: série das ondas relacionais (cada parede = um passo)."""
        if self._world_id is None:
            return []
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            with psycopg2.connect(self.db_uri, cursor_factory=RealDictCursor) as conn:
                with conn.cursor() as cur:
                    if conversation_key:
                        cur.execute(
                            """
                            SELECT id, conversation_key, wave_index, phase, metrics,
                                   parent_wave_id, state_tensor_id, event_id, created_at
                            FROM aeon_reality_waves
                            WHERE world_id = %s AND conversation_key = %s
                            ORDER BY id DESC LIMIT %s
                            """,
                            (self._world_id, str(conversation_key), max(1, min(256, int(limit)))),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, conversation_key, wave_index, phase, metrics,
                                   parent_wave_id, state_tensor_id, event_id, created_at
                            FROM aeon_reality_waves
                            WHERE world_id = %s
                            ORDER BY id DESC LIMIT %s
                            """,
                            (self._world_id, max(1, min(256, int(limit)))),
                        )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            self._logger.warning("wave query unavailable: %s", exc)
            return []

    def latest_wave_score(
        self,
        *,
        conversation_key: Optional[str] = None,
        metric_key: str = "reality_index",
    ) -> Optional[float]:
        """Último valor de uma métrica de onda (eco do ciclo anterior)."""
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    if conversation_key:
                        cur.execute(
                            f"""
                            SELECT metrics->>%s FROM aeon_reality_waves
                            WHERE world_id = %s AND conversation_key = %s
                              AND metrics ? %s
                            ORDER BY id DESC LIMIT 1
                            """,
                            (metric_key, self._world_id, str(conversation_key), metric_key),
                        )
                    else:
                        cur.execute(
                            f"""
                            SELECT metrics->>%s FROM aeon_reality_waves
                            WHERE world_id = %s AND metrics ? %s
                            ORDER BY id DESC LIMIT 1
                            """,
                            (metric_key, self._world_id, metric_key),
                        )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return float(row[0])
        except Exception as exc:
            self._logger.warning("latest wave metric unavailable: %s", exc)
        return None

    def latest_wave_metrics(
        self,
        *,
        conversation_key: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Métricas da onda mais recente (eco do ciclo anterior persistido)."""
        rows = self.wave_series(conversation_key=conversation_key, limit=1)
        if not rows:
            return None
        metrics = rows[0].get("metrics") or {}
        return metrics if isinstance(metrics, dict) else None

    def latest_wave_index_if_recent(
        self, *, conversation_key: Optional[str] = None, window_s: int = 45
    ) -> Optional[int]:
        """Índice da onda mais recente do reinado, se gravada na janela.

        O `observe` reutiliza a onda de fissão recém-gravada pelo `stream_messages`
        (mesmo `conversation_key`, últimos segundos) em vez de abrir um índice novo —
        colapsando a dupla onda por turno em uma única linha da Tábua.
        """
        if not conversation_key or self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri, connect_timeout=8) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        r"""
                        SELECT wave_index FROM aeon_reality_waves
                        WHERE world_id = %s AND conversation_key = %s
                          AND created_at >= CURRENT_TIMESTAMP - make_interval(secs => %s)
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        (self._world_id, str(conversation_key), int(window_s)),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("latest wave index unavailable: %s", exc)
            return None

    def next_wave_index(self, *, conversation_key: Optional[str] = None) -> int:
        """Próximo índice de onda do reinado (auto-avanço da Tábua).

        Cada turno observado de um mesmo conversation_key gera uma onda nova
        (índices 0, 1, 2…) em vez de sobrescrever sempre a onda 0 — mesmo quando
        o caminho híbrido chega com `wave=0` fixo.
        """
        if not conversation_key or self._world_id is None:
            return 0
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri, connect_timeout=8) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(MAX(wave_index), -1) FROM aeon_reality_waves
                        WHERE world_id = %s AND conversation_key = %s
                        """,
                        (self._world_id, str(conversation_key)),
                    )
                    row = cur.fetchone()
                    return (int(row[0]) + 1) if row else 0
        except Exception as exc:
            self._logger.warning("next wave index unavailable: %s", exc)
            return 0

    def metareality_seen(
        self,
        digest: str,
        *,
        window_s: int = 300,
    ) -> bool:
        """Idempotência da observação: o turno já foi registrado recentemente?

        Alguns caminhos do runtime (`hybrid_generate`, `stream`, `stream_messages`)
        observam o mesmo turno em camadas diferentes → o dedup por `turn_digest`
        impede eventos `metareality` e ondas duplicadas dentro da janela.
        """
        if not digest or self._world_id is None:
            return False
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri, connect_timeout=8) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        r"""
                        SELECT 1 FROM aeon_reality_events
                        WHERE world_id = %s AND event_type = 'metareality'
                          AND payload->>'turn_digest' = %s
                          AND created_at >= CURRENT_TIMESTAMP - make_interval(secs => %s)
                        LIMIT 1
                        """,
                        (self._world_id, str(digest), int(window_s)),
                    )
                    return cur.fetchone() is not None
        except Exception as exc:
            self._logger.warning("metareality dedup unavailable: %s", exc)
            return False

    def session_quality(
        self,
        *,
        conversation_key: Optional[str] = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        """Qualidade da fissão por sessão: cos_sim médio + razão de ondas completas.

        Deriva o cos_sim das métricas persistidas (já gravam `drift`/`cos_sim`).
        Backward-compat: ondas antigas sem `cos_sim` são convertidas de `drift`
        (`cos = 1 - 2·drift`); `drift` ausente/vazio não entra na média.
        """
        rows = self.wave_series(conversation_key=conversation_key, limit=limit)
        n = len(rows)
        if n == 0:
            return {
                "conversation_key": conversation_key,
                "waves": 0,
                "terminal_waves": 0,
                "complete_ratio": 0.0,
                "mean_cos_sim": None,
                "min_cos_sim": None,
                "max_cos_sim": None,
                "n_waves_with_vectors": 0,
            }
        terminal = 0
        cos_values: list[float] = []
        for row in rows:
            metrics = row.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            if row.get("phase") == "fim" or metrics.get("closed"):
                terminal += 1
            cos: Optional[float] = None
            raw = metrics.get("cos_sim")
            if isinstance(raw, (int, float)) and raw is not None:
                cos = float(raw)
            else:
                drift = metrics.get("drift")
                if isinstance(drift, (int, float)):
                    cos = 1.0 - 2.0 * float(drift)
            if cos is not None:
                cos = max(-1.0, min(1.0, cos))
                cos_values.append(cos)
        if not cos_values:
            return {
                "conversation_key": conversation_key,
                "waves": n,
                "terminal_waves": terminal,
                "complete_ratio": round(terminal / n, 3),
                "mean_cos_sim": None,
                "min_cos_sim": None,
                "max_cos_sim": None,
                "n_waves_with_vectors": 0,
            }
        return {
            "conversation_key": conversation_key,
            "waves": n,
            "terminal_waves": terminal,
            "complete_ratio": round(terminal / n, 3),
            "mean_cos_sim": round(sum(cos_values) / len(cos_values), 4),
            "min_cos_sim": round(min(cos_values), 4),
            "max_cos_sim": round(max(cos_values), 4),
            "n_waves_with_vectors": len(cos_values),
        }

    def _persist_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        wave: int,
        digest: str,
        previous_hash: Optional[str],
    ) -> Optional[int]:
        if self._world_id is None:
            return None
        try:
            import psycopg2

            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aeon_reality_events
                            (world_id, wave, event_type, payload,
                             previous_hash, event_hash)
                        VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (world_id, event_hash) DO NOTHING
                        RETURNING id
                        """,
                        (
                            self._world_id,
                            int(wave),
                            event_type,
                            json.dumps(payload, ensure_ascii=False),
                            previous_hash,
                            digest,
                        ),
                    )
                    row = cur.fetchone()
                    if row:
                        return int(row[0])
                    cur.execute(
                        """
                        SELECT id FROM aeon_reality_events
                        WHERE world_id = %s AND event_hash = %s
                        """,
                        (self._world_id, digest),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            self._logger.warning("event relation unavailable: %s", exc)
        return None
