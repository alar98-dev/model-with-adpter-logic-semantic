"""Decision Trace DB — Persiste tensores Blake no PostgreSQL pgvector.

Armazena:
  - blake_signature[6]: geometria de decisão (tool call, reasoning, etc.)
  - semantic_embedding[384]: embedding semântico do texto da conversa
  - full_text: conteúdo completo do prompt/resposta

Permite busca por:
  - similaridade de decisão (blake_signature <=>)
  - similaridade semântica (semantic_embedding <=> query)
  - filtro por classificação, sessão, tipo
"""

import os
import json
import uuid
import time
import hashlib
import threading
import numpy as np
from datetime import datetime
from typing import Optional

PG_DSN = os.environ.get(
    "DECISION_TRACE_DSN",
    f"dbname='matriz_vetorial' user='postgres' password='{os.environ.get('SWA_DB_PASSWORD', 'antigravity')}' host='localhost'"
)

EMBED_DIM = 384
_BATCH = []
_BATCH_LOCK = threading.Lock()
_BATCH_SIZE = 50


def _vec_str(vec: list[float]) -> str:
    return "[" + ",".join(str(round(v, 6)) for v in vec) + "]"


def _parse_vec(v, dim: int = 6) -> list[float]:
    if isinstance(v, str):
        try:
            parsed = [float(x) for x in v.strip("[]").split(",")]
            return parsed[:dim] if len(parsed) >= dim else parsed + [0.0] * (dim - len(parsed))
        except (ValueError, AttributeError):
            return [0.0] * dim
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v][:dim]
    return [0.0] * dim


def _conn():
    import psycopg2
    return psycopg2.connect(PG_DSN)


def semantic_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Gera embedding semântico determinístico de 384D a partir do hash SHA-256.

    Usa o texto como semente para um gerador pseudo-aleatório, produzindo
    o mesmo vetor para o mesmo texto sempre (determinístico). Zero overhead
    de ML, consistente entre processos.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed[:4], byteorder="little")
    rng = np.random.RandomState(seed_int)
    vec = rng.randn(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def persist_part_trace(
    session_id: str,
    part_id: str,
    classification: str,
    part_type: str,
    blake_signature: list[float],
    text_snippet: str = "",
    token_count: int = 0,
    tool_name: str = "",
    finish_reason: str = "",
    agent: str = "opencode",
    geometry_centroid: Optional[list] = None,
    geometry_span: Optional[list] = None,
    metadata: Optional[dict] = None,
    full_text: str = "",
):
    # --- TOKEN MODULAR DA BASE 60 (SUBSTITUI TEXTOS PESADOS POR TUPLA DE FASE) ---
    reino_id = 1
    casa_id = 1
    amplitude = 0.0
    try:
        with open("/dev/shm/aeon/sexagesimal_phase.json", "r") as sf:
            shm_info = json.load(sf)
            reino_id = int(shm_info.get("realm_id", 1))
            casa_id = int(shm_info.get("house_id", 1))
            amplitude = float(shm_info.get("amplitude", 0.0))
    except Exception:
        if blake_signature and len(blake_signature) >= 6:
            reino_id = (max(range(6), key=lambda i: abs(blake_signature[i])) % 6) + 1
            amplitude = sum(abs(v) for v in blake_signature) / 6.0
            casa_id = ((reino_id - 1) * 10) + int(min(9, amplitude * 10)) + 1

    phase_token_str = f"PHASE_TOKEN:({reino_id},{casa_id},{amplitude:.4f})"
    meta = metadata or {}
    meta.update({
        "reino_id": reino_id,
        "casa_id": casa_id,
        "amplitude": round(amplitude, 4),
        "phase_deg": (casa_id - 1) * 6.0,
        "modular_token": True
    })

    entry = {
        "session_id": session_id,
        "part_id": part_id,
        "classification": classification,
        "part_type": part_type,
        "trace_class": classification,
        "agent": agent,
        "blake_signature": _vec_str(blake_signature),
        "geometry_centroid": _vec_str(geometry_centroid) if geometry_centroid else None,
        "geometry_span": _vec_str(geometry_span) if geometry_span else None,
        # The phase token is metadata; the emitted text remains replayable.
        "text_snippet": text_snippet or phase_token_str,
        "token_count": token_count,
        "tool_name": tool_name,
        "finish_reason": finish_reason,
        "metadata": json.dumps(meta),
        "full_text": full_text or text_snippet or phase_token_str,
        "semantic_embedding": None,
    }
    with _BATCH_LOCK:
        _BATCH.append(entry)
        if len(_BATCH) >= _BATCH_SIZE:
            _flush_batch()


def _flush_batch():
    global _BATCH
    if not _BATCH:
        return
    batch = _BATCH[:]
    _BATCH = []
    try:
        conn = _conn()
        cur = conn.cursor()
        for e in batch:
            cur.execute(
                """INSERT INTO decision_traces
                   (session_id, part_id, classification, part_type, trace_class,
                    agent, blake_signature, geometry_centroid, geometry_span,
                    text_snippet, token_count, tool_name, finish_reason, metadata,
                    full_text, semantic_embedding)
                   VALUES (%(session_id)s, %(part_id)s, %(classification)s,
                    %(part_type)s, %(trace_class)s, %(agent)s,
                    %(blake_signature)s::vector,
                    CASE WHEN %(geometry_centroid)s IS NOT NULL THEN %(geometry_centroid)s::vector ELSE NULL END,
                    CASE WHEN %(geometry_span)s IS NOT NULL THEN %(geometry_span)s::vector ELSE NULL END,
                    %(text_snippet)s, %(token_count)s, %(tool_name)s,
                    %(finish_reason)s, %(metadata)s::jsonb,
                    %(full_text)s, CASE WHEN %(semantic_embedding)s IS NOT NULL THEN %(semantic_embedding)s::vector ELSE NULL END)
                   ON CONFLICT (part_id) WHERE part_id IS NOT NULL AND part_id != ''
                   DO UPDATE SET
                       classification = EXCLUDED.classification,
                       token_count = EXCLUDED.token_count,
                       text_snippet = EXCLUDED.text_snippet,
                       full_text = EXCLUDED.full_text,
                       semantic_embedding = EXCLUDED.semantic_embedding,
                       created_at = NOW()""",
                e,
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[DECISION_TRACE] Flush error ({len(batch)} items): {exc}")


def persist_geometry(
    session_id: str,
    classification: str,
    centroid: list[float],
    span: list[float],
    type_distribution: dict,
    part_count: int,
    total_tokens: int,
):
    metadata = {
        "type_distribution": type_distribution,
        "part_count": part_count,
        "total_tokens": total_tokens,
    }
    desc = f"geometry:{classification}:{part_count}parts:{total_tokens}tokens"
    sem_emb = semantic_embed(desc)
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO decision_traces
               (session_id, classification, trace_class, agent,
                blake_signature, geometry_centroid, geometry_span,
                text_snippet, token_count, metadata,
                full_text, semantic_embedding)
               VALUES (%s, %s, 'geometry', 'opencode',
                %s::vector, %s::vector, %s::vector,
                %s, %s, %s::jsonb,
                %s, %s::vector)
               ON CONFLICT (session_id) WHERE trace_class = 'geometry'
               DO UPDATE SET
                   classification = EXCLUDED.classification,
                   blake_signature = EXCLUDED.blake_signature,
                   token_count = EXCLUDED.token_count,
                   semantic_embedding = EXCLUDED.semantic_embedding,
                   created_at = NOW()""",
            (
                session_id,
                classification,
                _vec_str(centroid),
                _vec_str(centroid),
                _vec_str(span),
                desc,
                total_tokens,
                json.dumps(metadata),
                desc,
                _vec_str(sem_emb),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[DECISION_TRACE] Geometry persist error: {exc}")


def search_similar(
    tensor: list[float],
    k: int = 5,
    classification: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[dict]:
    """Busca por similaridade de decisão (blake_signature[6])."""
    vec = _vec_str(tensor)
    conditions = []
    params: list = []
    if classification:
        conditions.append("classification = %s")
        params.append(classification)
    if session_id:
        conditions.append("session_id = %s")
        params.append(session_id)
    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT id, session_id, classification, part_type, trace_class,
               blake_signature, geometry_centroid, text_snippet,
               token_count, tool_name, created_at, full_text,
               blake_signature <=> %s::vector AS distance
        FROM decision_traces
        WHERE {where}
        ORDER BY distance ASC
        LIMIT %s
    """
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(sql, [vec] + params + [k])
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "classification": r[2],
                "part_type": r[3],
                "trace_class": r[4],
                "blake_signature": _parse_vec(r[5]),
                "geometry_centroid": _parse_vec(r[6]) if r[6] else None,
                "text_snippet": r[7] or "",
                "token_count": r[8],
                "tool_name": r[9],
                "created_at": str(r[10]) if r[10] else "",
                "full_text": r[11] or "",
                "distance": float(r[12]),
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[DECISION_TRACE] Search error: {exc}")
        return []


def search_semantic(query: str, k: int = 5, classification: Optional[str] = None) -> list[dict]:
    """Busca por similaridade semântica — acha traces pelo significado do texto.

    Exemplo:
        search_semantic("shared context window between agents", k=5)
        → retorna os traces de conversa que mais se aproximam semanticamente
    """
    query_emb = semantic_embed(query)
    vec = _vec_str(query_emb)
    conditions = []
    params: list = []
    if classification:
        conditions.append("classification = %s")
        params.append(classification)
    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT id, session_id, classification, part_type, trace_class,
               text_snippet, full_text, token_count, tool_name, created_at,
               semantic_embedding <=> %s::vector AS distance
        FROM decision_traces
        WHERE {where} AND full_text IS NOT NULL AND full_text != ''
        ORDER BY distance ASC
        LIMIT %s
    """
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("SET enable_seqscan = ON")
        cur.execute(sql, [vec] + params + [k])
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "classification": r[2],
                "part_type": r[3],
                "trace_class": r[4],
                "text_snippet": r[5] or "",
                "full_text": r[6] or "",
                "token_count": r[7],
                "tool_name": r[8],
                "created_at": str(r[9]) if r[9] else "",
                "distance": float(r[10]),
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[DECISION_TRACE] Semantic search error: {exc}")
        return []


def get_latest_traces(limit: int = 20, classification: Optional[str] = None) -> list[dict]:
    where = "WHERE classification = %s" if classification else ""
    params = [classification] if classification else []
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            f"""SELECT id, session_id, classification, part_type, trace_class,
                       blake_signature, text_snippet, token_count,
                       tool_name, created_at, full_text
                FROM decision_traces {where}
                ORDER BY created_at DESC LIMIT %s""",
            params + [limit],
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "classification": r[2],
                "part_type": r[3],
                "trace_class": r[4],
                "blake_signature": _parse_vec(r[5]),
                "text_snippet": r[6] or "",
                "token_count": r[7],
                "tool_name": r[8],
                "created_at": str(r[9]) if r[9] else "",
                "full_text": r[10] or "",
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[DECISION_TRACE] Latest error: {exc}")
        return []


def count_by_classification() -> dict:
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT classification, COUNT(*) FROM decision_traces GROUP BY classification ORDER BY COUNT DESC"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except Exception as exc:
        print(f"[DECISION_TRACE] Count error: {exc}")
        return {}


def flush():
    _flush_batch()
