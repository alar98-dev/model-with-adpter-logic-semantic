"""Identity domain controller — facial identity CRUD with pgvector."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from swa.db.base import BaseController


class IdentityController(BaseController):
    """Manages facial identities, face embeddings, and recognition sessions.

    Dual-mode: PostgreSQL (pgvector) + in-memory fallback.
    """

    # ── Identity CRUD ────────────────────────────────────────────────

    def create_identity(
        self,
        nome: str,
        apelido: Optional[str] = None,
        email: Optional[str] = None,
        papel: str = "usuario",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new identity (ficha cadastral). Returns the created record."""
        identity_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {})

        payload = {
            "id": identity_id,
            "nome": nome,
            "apelido": apelido,
            "email": email,
            "papel": papel,
            "ativo": True,
            "metadata": metadata or {},
        }

        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO swa_identities
                               (id, nome, apelido, email, papel, metadata)
                               VALUES (%s, %s, %s, %s, %s, %s::jsonb);""",
                            (identity_id, nome, apelido, email, papel, meta_json),
                        )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass

        self._RAM_DATABASE.setdefault("swa_identities", []).append(payload)
        return payload

    def get_identity(self, identity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single identity by UUID."""
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT * FROM swa_identities WHERE id = %s;",
                            (identity_id,),
                        )
                        row = cur.fetchone()
                        return dict(row) if row else None
                finally:
                    conn.close()
            except Exception:
                pass

        for rec in self._RAM_DATABASE.get("swa_identities", []):
            if rec.get("id") == identity_id:
                return rec
        return None

    def list_identities(self, ativo_only: bool = True) -> List[Dict[str, Any]]:
        """List all identities."""
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        if ativo_only:
                            cur.execute(
                                "SELECT * FROM swa_identities WHERE ativo = TRUE ORDER BY nome;"
                            )
                        else:
                            cur.execute("SELECT * FROM swa_identities ORDER BY nome;")
                        return [dict(row) for row in cur.fetchall()]
                finally:
                    conn.close()
            except Exception:
                pass

        results = self._RAM_DATABASE.get("swa_identities", [])
        if ativo_only:
            results = [r for r in results if r.get("ativo", True)]
        return results

    def update_identity(self, identity_id: str, **fields) -> bool:
        """Update identity fields. Returns True if updated."""
        allowed = {"nome", "apelido", "email", "papel", "ativo", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False

        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        updates["updated_at"] = datetime.utcnow()

        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    set_parts = []
                    values = []
                    for k, v in updates.items():
                        if k == "metadata":
                            set_parts.append(f"{k} = %s::jsonb")
                        else:
                            set_parts.append(f"{k} = %s")
                        values.append(v)
                    values.append(identity_id)

                    with conn.cursor() as cur:
                        cur.execute(
                            f"UPDATE swa_identities SET {', '.join(set_parts)} WHERE id = %s;",
                            values,
                        )
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()
            except Exception:
                pass

        for rec in self._RAM_DATABASE.get("swa_identities", []):
            if rec.get("id") == identity_id:
                rec.update(fields)
                return True
        return False

    def delete_identity(self, identity_id: str) -> bool:
        """Soft-delete (set ativo=FALSE)."""
        return self.update_identity(identity_id, ativo=False)

    # ── Face Embeddings ──────────────────────────────────────────────

    def add_face_encoding(
        self,
        identity_id: str,
        embedding_128: List[float],
        angle: str = "front",
        photo_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        subtleties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a face embedding linked to an identity."""
        sub_json = json.dumps(subtleties or {})
        payload = {
            "identity_id": identity_id,
            "embedding_128": embedding_128,
            "angle": angle,
            "photo_path": photo_path,
            "confidence_threshold": confidence_threshold,
            "subtleties": subtleties or {},
        }

        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO swa_faces
                               (identity_id, embedding_128, angle, photo_path,
                                confidence_threshold, subtleties)
                               VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                               RETURNING id;""",
                            (identity_id, embedding_128, angle, photo_path,
                             confidence_threshold, sub_json),
                        )
                        row = cur.fetchone()
                        payload["id"] = row["id"] if row else None
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass

        self._RAM_DATABASE.setdefault("swa_faces", []).append(payload)
        return payload

    def get_face_encodings(
        self, identity_id: str, angle: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all face encodings for an identity, optionally filtered by angle."""
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        if angle:
                            cur.execute(
                                """SELECT * FROM swa_faces
                                   WHERE identity_id = %s AND angle = %s;""",
                                (identity_id, angle),
                            )
                        else:
                            cur.execute(
                                "SELECT * FROM swa_faces WHERE identity_id = %s;",
                                (identity_id,),
                            )
                        return [dict(row) for row in cur.fetchall()]
                finally:
                    conn.close()
            except Exception:
                pass

        return [
            f for f in self._RAM_DATABASE.get("swa_faces", [])
            if f.get("identity_id") == identity_id
            and (angle is None or f.get("angle") == angle)
        ]

    def find_identity_by_face(
        self,
        embedding_128: List[float],
        threshold: float = 0.50,
    ) -> Optional[Dict[str, Any]]:
        """Find the closest identity by face embedding using pgvector cosine search.

        Returns None if no match above threshold.
        """
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT f.id AS face_id, f.identity_id, f.angle,
                                      f.confidence_threshold, f.subtleties,
                                      i.nome, i.apelido, i.papel, i.metadata,
                                      1 - (f.embedding_128 <=> %s::vector(128)) AS similarity
                               FROM swa_faces f
                               JOIN swa_identities i ON i.id = f.identity_id
                               WHERE i.ativo = TRUE
                               ORDER BY similarity DESC
                               LIMIT 1;""",
                            (embedding_128,),
                        )
                        row = cur.fetchone()
                        if row:
                            result = dict(row)
                            if result.get("similarity", 0) >= (1.0 - threshold):
                                return result
                        return None
                finally:
                    conn.close()
            except Exception:
                pass

        # RAM fallback
        best_match = None
        best_dist = 1.0
        for face in self._RAM_DATABASE.get("swa_faces", []):
            enc = face.get("embedding_128", [])
            if enc:
                d = self.cosine_similarity(embedding_128, enc)
                dist = 1.0 - d
                if dist < best_dist:
                    best_dist = dist
                    best_match = face

        if best_match and best_dist < threshold:
            identity = self.get_identity(best_match.get("identity_id", ""))
            if identity:
                return {
                    "face_id": best_match.get("id"),
                    "identity_id": best_match["identity_id"],
                    "angle": best_match.get("angle"),
                    "similarity": 1.0 - best_dist,
                    "nome": identity.get("nome"),
                    "apelido": identity.get("apelido"),
                    "papel": identity.get("papel"),
                    "metadata": identity.get("metadata"),
                }
        return None

    def update_face_encoding(
        self,
        face_id: int,
        new_embedding: List[float],
        subtleties: Optional[Dict[str, Any]] = None,
        blend_factor: float = 0.05,
    ) -> bool:
        """Auto-learn: blend new embedding with existing (exponential moving average)."""
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        if blend_factor > 0:
                            cur.execute(
                                """UPDATE swa_faces SET
                                   embedding_128 = (%s::vector(128) * %s) + (embedding_128 * (1.0 - %s)),
                                   subtleties = COALESCE(%s::jsonb, subtleties)
                                   WHERE id = %s;""",
                                (new_embedding, blend_factor, blend_factor,
                                 json.dumps(subtleties) if subtleties else None, face_id),
                            )
                        else:
                            cur.execute(
                                """UPDATE swa_faces SET
                                   embedding_128 = %s,
                                   subtleties = COALESCE(%s::jsonb, subtleties)
                                   WHERE id = %s;""",
                                (new_embedding,
                                 json.dumps(subtleties) if subtleties else None, face_id),
                            )
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()
            except Exception:
                pass

        for face in self._RAM_DATABASE.get("swa_faces", []):
            if face.get("id") == face_id:
                old = np.array(face["embedding_128"], dtype=np.float32)
                new = np.array(new_embedding, dtype=np.float32)
                blended = old * (1.0 - blend_factor) + new * blend_factor
                face["embedding_128"] = blended.tolist()
                if subtleties:
                    face["subtleties"] = subtleties
                return True
        return False

    # ── Recognition Logging ──────────────────────────────────────────

    def log_recognition(
        self,
        identity_id: str,
        face_id: int,
        confidence: float,
        angle: Optional[str] = None,
        photo_snapshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a recognition event."""
        payload = {
            "identity_id": identity_id,
            "face_id": face_id,
            "confidence": confidence,
            "angle": angle,
        }

        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO swa_recognition_sessions
                               (identity_id, face_id, confidence, angle, photo_snapshot_path)
                               VALUES (%s, %s, %s, %s, %s);""",
                            (identity_id, face_id, confidence, angle, photo_snapshot_path),
                        )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass

        self._RAM_DATABASE.setdefault("swa_recognition_sessions", []).append(payload)
        return payload

    def get_recognition_history(
        self, identity_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent recognition events for an identity."""
        if self.is_db_active():
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT * FROM swa_recognition_sessions
                               WHERE identity_id = %s
                               ORDER BY detected_at DESC LIMIT %s;""",
                            (identity_id, limit),
                        )
                        return [dict(row) for row in cur.fetchall()]
                finally:
                    conn.close()
            except Exception:
                pass

        sessions = [
            s for s in self._RAM_DATABASE.get("swa_recognition_sessions", [])
            if s.get("identity_id") == identity_id
        ]
        sessions.sort(key=lambda x: x.get("detected_at", ""), reverse=True)
        return sessions[:limit]
