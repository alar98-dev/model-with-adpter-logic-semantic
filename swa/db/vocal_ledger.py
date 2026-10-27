"""Ledger vocal — persiste cada fala de Alifer de forma rastreável.

Fecha o ciclo que o `cortex/auditory_cortex.py` deixava aberto: antes, a
identidade vocal existia só como prefixo de texto (``[LOCUTOR: X]``) e o
``accuracy_score`` era descartado. Aqui a fala vira uma entidade com:

  • vínculo real de FK para `swa_identities` (quem falou) e para
    `blake_semantic_anchors` (o embedding 2560D indexável);
  • `tensor_snapshot` — retrato compacto do vetor (perfil de energia por banda),
    consultável em JSONB sem duplicar 2560 floats por linha;
  • uma linha por palavra, com deslocamento de caractere e janela temporal.

Sobre o tempo: o ASR deste projeto entrega a transcrição da fala inteira, sem
marcadores de palavra. Os tempos por palavra são portanto *derivados*
(interpolados por peso de caractere dentro da duração da fala) e ficam
marcados como tal em `timing_source`. Dia que o Whisper passar a emitir
`word_timestamps`, o mesmo campo recebe o valor real e nada mais precisa mudar.
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .models import Identity

# O SQLAlchemy é importado dentro das funções, não no topo. O
# cortex/auditory_cortex.py precisa só de `record_utterance_pg` e roda sob
# MemoryMax=512M — puxar o ORM no import custaria ~115MB residentes e causaria
# OOM kill. As funções puras (tokenização, snapshot) não custam nada.

# Velocidade de fala usada só como fallback quando não há duração alguma.
FALLBACK_WORDS_PER_MINUTE = 150.0

# O chunking acompanha a menor camada do K-Base (100 chars), para que cada
# linha de palavra seja rastreável até um chunk de corpus.
CHUNK_CHARS = 100

_WORD_RE = re.compile(r"\S+")


def tensor_snapshot(vec: Optional[Sequence[float]], bands: int = 16) -> dict:
    """Retrato compacto de um vetor: norma global + energia por banda.

    Escolhido em vez de copiar os 2560 floats: o vetor completo já vive em
    `blake_semantic_anchors.embedding_2560` (indexável por pgvector). O snapshot
    existe para o ledger responder "que forma tenía este tensor" em uma
    consulta JSONB barata.
    """
    if not vec:
        return {}

    n = len(vec)
    try:
        total_l2 = math.sqrt(sum(float(x) * float(x) for x in vec))
        mean_abs = sum(abs(float(x)) for x in vec) / n
    except (TypeError, ValueError):
        return {}

    width = max(1, n // bands)
    profile = []
    for i in range(bands):
        start = i * width
        end = n if i == bands - 1 else min(n, start + width)
        if start >= end:
            break
        chunk = [float(x) for x in vec[start:end]]
        profile.append(
            {
                "band": i,
                "l2": round(math.sqrt(sum(x * x for x in chunk)), 6),
                "mean_abs": round(sum(abs(x) for x in chunk) / len(chunk), 8),
            }
        )

    return {
        "dim": n,
        "l2": round(total_l2, 6),
        "mean_abs": round(mean_abs, 8),
        "bands": profile,
    }


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Quebra em palavras preservando o deslocamento de caractere de cada uma."""
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]


def _resolve_duration_ms(
    words: Sequence[tuple[str, int, int]], duration_ms: Optional[float]
) -> tuple[float, str]:
    """Devolve (duração_ms, origem). Nunca inventa origem 'asr'."""
    if duration_ms and duration_ms > 0:
        return float(duration_ms), "measured"

    estimated = (len(words) / FALLBACK_WORDS_PER_MINUTE) * 60_000.0
    return max(estimated, 1.0), "estimated"


def build_word_rows(
    text: str,
    duration_ms: Optional[float] = None,
    confidence: Optional[float] = None,
) -> tuple[list[dict], float, str, int]:
    """Monta as linhas de palavra interpolando o tempo por peso de caractere.

    Devolve (linhas, duração_ms, origem_do_tempo, nº de chunks).
    """
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return [], 0.0, "none", 0

    span_ms, timing_source = _resolve_duration_ms(tokens, duration_ms)
    total_chars = sum(len(surface) for surface, _, _ in tokens) or 1

    rows: list[dict] = []
    elapsed = 0.0
    chunk_index = 0
    chunk_chars = 0

    for ordinal, (surface, char_start, char_end) in enumerate(tokens):
        weight = len(surface) / total_chars
        t_start = elapsed
        t_end = elapsed + (span_ms * weight)
        elapsed = t_end

        if chunk_chars > 0 and chunk_chars + len(surface) > CHUNK_CHARS:
            chunk_index += 1
            chunk_chars = 0
        chunk_chars += len(surface) + 1

        rows.append(
            {
                "ordinal": ordinal,
                "surface": surface,
                "char_start": char_start,
                "char_end": char_end,
                "chunk_index": chunk_index,
                "t_start_ms": round(t_start, 3),
                "t_end_ms": round(t_end, 3),
                "confidence": confidence,
            }
        )

    return rows, round(span_ms, 3), timing_source, chunk_index + 1


def owner_nome() -> Optional[str]:
    """Nome do dono vem da configuração, nunca de um literal no código."""
    nome = os.environ.get("SWA_VOICE_OWNER", "").strip()
    return nome or None


def resolve_identity(db, speaker: Optional[str] = None, is_owner: bool = False) -> Optional[Identity]:
    """Resolve a Identity consultando o banco. Sem nome hardcoded.

    A ordem de tentativa é: nome exato do locutor → nome parcial → dono
    configurado → identidade com papel de administrador → única identidade
    ativa. Se nada casar, devolve None e a fala entra sem vínculo — que é
    preferível a inventar um dono.
    """
    from .models import Identity

    if speaker:
        found = db.query(Identity).filter(Identity.nome == speaker).first()
        if found:
            return found
        found = (
            db.query(Identity)
            .filter(Identity.nome.ilike(f"%{speaker}%"))
            .first()
        )
        if found:
            return found

    configured = owner_nome()
    if configured:
        found = db.query(Identity).filter(Identity.nome == configured).first()
        if found:
            return found

    if is_owner:
        admin = (
            db.query(Identity)
            .filter(Identity.papel == "admin", Identity.ativo.is_(True))
            .first()
        )
        if admin:
            return admin

    actives = db.query(Identity).filter(Identity.ativo.is_(True)).all()
    if len(actives) == 1:
        return actives[0]
    return None


def anchor_exists(db, anchor_id: Optional[int]) -> bool:
    """Confere no banco se a âncora vectorial realmente existe."""
    if not anchor_id:
        return False

    from sqlalchemy import text

    return bool(
        db.execute(
            text("SELECT EXISTS (SELECT 1 FROM blake_semantic_anchors WHERE id = :aid)"),
            {"aid": int(anchor_id)},
        ).scalar()
    )


def record_utterance(
    text: str,
    *,
    speaker: Optional[str] = None,
    is_owner: bool = False,
    similarity: float = 0.0,
    source: str = "MICROPHONE",
    captured_at: Optional[float] = None,
    duration_ms: Optional[float] = None,
    energy: Optional[float] = None,
    anchor_id: Optional[int] = None,
    embedding: Optional[Sequence[float]] = None,
    meta: Optional[dict] = None,
    min_similarity: float = 0.0,
) -> Optional[VocalUtterance]:
    """Persiste uma fala e suas palavras. Devolve a utterance ou None.

    Toda relação é conferida contra o banco antes do commit: a âncora
    vectorial precisa existir de fato, e a propriedade de "dono" vem do papel
    da Identity resolvida — não de um flag que o payload traz. Se o âncora
    não existir, a fala entra com `anchor_id` nulo em vez de apontar para o
    vazio.

    `min_similarity` descarta falas abaixo do limiar — o filtro que impede que
    ruído de ambiente entre no ledger como se fosse fala do dono.
    """
    text = (text or "").strip()
    if len(text) < 3:
        return None
    if float(similarity) < float(min_similarity):
        return None

    word_rows, span_ms, timing_source, n_chunks = build_word_rows(
        text, duration_ms, confidence=similarity or None
    )

    when = (
        datetime.fromtimestamp(captured_at, tz=timezone.utc).replace(tzinfo=None)
        if captured_at
        else datetime.utcnow()
    )

    from .models import VocalUtterance, VocalWord, session_scope

    with session_scope() as db:
        identity = resolve_identity(db, speaker, is_owner)

        # O vínculo vetorial só é gravado se a âncora existir no banco.
        linked_anchor = anchor_id if anchor_exists(db, anchor_id) else None

        # "Dono" é o que o banco diz sobre a Identity, não o que o payload afirmou.
        owner_flag = bool(identity and identity.papel == "admin" and identity.ativo)

        payload_meta = dict(meta or {})
        payload_meta.update(
            {
                "word_count": len(word_rows),
                "chunk_count": n_chunks,
                "char_count": len(text),
                "anchor_verified": linked_anchor is not None,
                "anchor_requested": anchor_id,
                "identity_resolved": identity.nome if identity else None,
            }
        )

        utt = VocalUtterance(
            identity_id=identity.id if identity else None,
            anchor_id=linked_anchor,
            text=text,
            source=source,
            captured_at=when,
            duration_ms=span_ms,
            similarity=float(similarity),
            is_owner=owner_flag,
            energy=energy,
            timing_source=timing_source,
            tensor_snapshot=tensor_snapshot(embedding),
            meta=payload_meta,
        )
        utt.words = [VocalWord(**row) for row in word_rows]
        db.add(utt)
        db.flush()
        utterance_id = utt.id

    return get_utterance(utterance_id)


def get_utterance(utterance_id: int) -> Optional["VocalUtterance"]:
    from sqlalchemy.orm import joinedload

    from .models import VocalUtterance, session_scope

    with session_scope() as db:
        return (
            db.query(VocalUtterance)
            .options(joinedload(VocalUtterance.identity), joinedload(VocalUtterance.words))
            .filter_by(id=utterance_id)
            .first()
        )


def iter_confirmed(limit: int = 50) -> list["VocalUtterance"]:
    """Falas confirmadas do dono, mais recentes primeiro."""
    from sqlalchemy.orm import joinedload

    from .models import VocalUtterance, session_scope

    with session_scope() as db:
        return (
            db.query(VocalUtterance)
            .options(joinedload(VocalUtterance.identity), joinedload(VocalUtterance.words))
            .filter(VocalUtterance.is_owner.is_(True))
            .order_by(VocalUtterance.captured_at.desc())
            .limit(limit)
            .all()
        )


def verify_integrity() -> dict:
    """Confere o ledger contra o estado real do banco e devolve o diagnóstico.

    Existe para não confiar na suposição: toda fala, âncora e palavra é
    reconciliada com o que o Postgres realmente tem.
    """
    from sqlalchemy import text

    from .models import Identity, VocalUtterance, VocalWord, session_scope

    report: dict = {
        "ok": True,
        "owner_configured": owner_nome(),
        "identities": [],
        "counts": {},
        "orphans": {},
    }

    with session_scope() as db:
        for ident in db.query(Identity).all():
            report["identities"].append(
                {
                    "id": str(ident.id),
                    "nome": ident.nome,
                    "papel": ident.papel,
                    "ativo": bool(ident.ativo),
                    "falas": db.query(VocalUtterance)
                    .filter_by(identity_id=ident.id)
                    .count(),
                }
            )

        report["counts"] = {
            "utterances": db.query(VocalUtterance).count(),
            "words": db.query(VocalWord).count(),
            "with_identity": db.query(VocalUtterance)
            .filter(VocalUtterance.identity_id.isnot(None))
            .count(),
            "with_anchor": db.query(VocalUtterance)
            .filter(VocalUtterance.anchor_id.isnot(None))
            .count(),
            "with_snapshot": db.query(VocalUtterance)
            .filter(VocalUtterance.tensor_snapshot != {})
            .count(),
        }

        # Falas cujo vínculo não sobrevive no banco.
        report["orphans"] = {
            "identity_missing": [
                r[0]
                for r in db.execute(
                    text(
                        "SELECT u.id FROM vocal_utterances u "
                        "LEFT JOIN swa_identities i ON i.id = u.identity_id "
                        "WHERE u.identity_id IS NOT NULL AND i.id IS NULL"
                    )
                ).all()
            ],
            "anchor_missing": [
                r[0]
                for r in db.execute(
                    text(
                        "SELECT u.id FROM vocal_utterances u "
                        "LEFT JOIN blake_semantic_anchors a ON a.id = u.anchor_id "
                        "WHERE u.anchor_id IS NOT NULL AND a.id IS NULL"
                    )
                ).all()
            ],
            "words_without_utterance": [
                r[0]
                for r in db.execute(
                    text(
                        "SELECT w.id FROM vocal_words w "
                        "LEFT JOIN vocal_utterances u ON u.id = w.utterance_id "
                        "WHERE u.id IS NULL"
                    )
                ).all()
            ],
        }

    report["ok"] = not any(report["orphans"].values())
    return report


# ─── Via de escrita sem ORM ────────────────────────────────────────────────
# O cortex/auditory_cortex.py roda sob MemoryMax=512M. Importar o SQLAlchemy
# ali custa ~115MB residentes (o pico subiu de 270MB para 384MB e disparou o
# OOM killer). Estas funções falam direto com o psycopg2 usando a conexão que o
# serviço já tem aberta — a escrita não custa memória extra e o ledger continua
# legível pelo ORM, que é a interface de consulta.

_RESOLVE_SQL = """
SELECT id, nome, papel, ativo FROM swa_identities
 WHERE nome = %(speaker)s
    OR nome ILIKE %(ilike)s
 ORDER BY (nome = %(speaker)s) DESC, nome
 LIMIT 1
"""

_RESOLVE_ADMIN_SQL = """
SELECT id, nome, papel, ativo FROM swa_identities
 WHERE papel = 'admin' AND ativo IS TRUE
 ORDER BY nome LIMIT 1
"""

_RESOLVE_SINGLE_SQL = """
SELECT id, nome, papel, ativo FROM swa_identities
 WHERE ativo IS TRUE ORDER BY nome LIMIT 1
"""


def _resolve_identity_pg(cur, speaker: Optional[str], is_owner: bool) -> Optional[tuple]:
    """Mesma política de `resolve_identity`, em SQL — sem carregar o ORM."""
    configured = owner_nome()

    for candidate in (speaker, configured):
        if not candidate:
            continue
        cur.execute(_RESOLVE_SQL, {"speaker": candidate, "ilike": f"%{candidate}%"})
        row = cur.fetchone()
        if row:
            return row

    if is_owner:
        cur.execute(_RESOLVE_ADMIN_SQL)
        row = cur.fetchone()
        if row:
            return row

    cur.execute(_RESOLVE_SINGLE_SQL)
    return cur.fetchone()


def _pack_f32(vec) -> Optional[bytes]:
    """Vetor -> bytes float32 little-endian (a forma que o PG devolve crua)."""
    if vec is None or len(vec) == 0:
        return None
    try:
        import numpy as _np
        return _np.asarray(vec, dtype=_np.float32).astype("<f4").tobytes()
    except Exception:
        return None


def band_profile(vec, bands: int = 16) -> Optional[list]:
    """Perfil de energia por banda, achatado em `bands` floats.

    Reaproveita `tensor_snapshot`, que já fatia o vetor em faixas com média
    absoluta. Achata para um vetor numérico para que os três níveis da
    hierarquia guardem a MESMA forma — assim dá para somar e agregar com
    numpy sem parsing. O conteúdo é derivado; a fonte primária continua
    sendo o tensor binário.
    """
    if vec is None or len(vec) == 0:
        return None
    try:
        bs = tensor_snapshot(vec, bands=bands).get("bands") or []
        if not bs:
            return None
        return [float(b.get("mean_abs", 0.0)) for b in bs]
    except Exception:
        return None


def record_utterance_pg(
    conn,
    text_: str,
    *,
    speaker: Optional[str] = None,
    is_owner: bool = False,
    similarity: float = 0.0,
    source: str = "MICROPHONE",
    captured_at: Optional[float] = None,
    energy: Optional[float] = None,
    anchor_id: Optional[int] = None,
    embedding: Optional[Sequence[float]] = None,
    meta: Optional[dict] = None,
    min_similarity: float = 0.0,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    duration_ms: Optional[float] = None,
    word_timestamps: Optional[Sequence[float]] = None,
    session_key: Optional[str] = None,
) -> Optional[int]:
    """Grava a fala e suas palavras via psycopg2. Devolve o id ou None.

    Mesmas garantias de `record_utterance`: âncora conferida antes do vínculo,
    propriedade de dono derivada do papel no banco, tempo marcado com a origem
    real.
    """
    text_ = (text_ or "").strip()
    if len(text_) < 3 or float(similarity) < float(min_similarity):
        return None

    word_rows, span_ms, timing_source, n_chunks = build_word_rows(
        text_, duration_ms, confidence=similarity or None
    )
    when = (
        datetime.fromtimestamp(captured_at, tz=timezone.utc).replace(tzinfo=None)
        if captured_at
        else datetime.utcnow()
    )

    import json as _json

    cur = conn.cursor()
    try:
        ident = _resolve_identity_pg(cur, speaker, is_owner)
        identity_id = ident[0] if ident else None
        owner_flag = bool(ident and ident[2] == "admin" and ident[3])

        linked_anchor = None
        if anchor_id:
            cur.execute(
                "SELECT 1 FROM blake_semantic_anchors WHERE id = %s", (int(anchor_id),)
            )
            if cur.fetchone():
                linked_anchor = int(anchor_id)

        payload_meta = dict(meta or {})
        payload_meta.update(
            {
                "word_count": len(word_rows),
                "chunk_count": n_chunks,
                "char_count": len(text_),
                "anchor_verified": linked_anchor is not None,
                "anchor_requested": anchor_id,
                "identity_resolved": ident[1] if ident else None,
            }
        )

        # --- conjunto (nível 3): sessão do dia, aberta sob demanda ---------
        seg_key = session_key or (f"live-{(when or datetime.utcnow()).strftime('%Y%m%d')}")
        cur.execute(
            "SELECT id FROM vocal_segments WHERE session_key = %s", (seg_key,)
        )
        row = cur.fetchone()
        if row:
            seg_id = row[0]
        else:
            cur.execute(
                """
                INSERT INTO vocal_segments
                  (identity_id, session_key, started_at, word_count, utterance_count)
                VALUES (%s,%s,%s,%s,0)
                ON CONFLICT (session_key) DO UPDATE SET identity_id = EXCLUDED.identity_id
                RETURNING id
                """,
                (identity_id, seg_key, (started_at or when), len(word_rows)),
            )
            seg_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO vocal_utterances
              (identity_id, anchor_id, segment_id, text, source, captured_at, duration_ms,
               similarity, is_owner, energy, timing_source, tensor_snapshot, meta,
               started_at, ended_at, text_hash, vector_acoustic, vector_energia, shadow)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,
                    %s,%s,%s,%s,%s,%s::jsonb)
            RETURNING id
            """,
            (
                identity_id, linked_anchor, seg_id, text_, source, when, span_ms,
                float(similarity), owner_flag, energy, timing_source,
                _json.dumps(tensor_snapshot(embedding)),
                _json.dumps(payload_meta),
                started_at, ended_at, text_[:80],
                _pack_f32(embedding), _pack_f32(band_profile(embedding)),
                _json.dumps({"source": source, "word_count": len(word_rows)}),
            ),
        )
        utterance_id = cur.fetchone()[0]

        if word_rows:
            # ts_abs por palavra quando o ASR os fornecer; senão a origem fica
            # marcada como estimada, para nunca se passar por instante real.
            ts_palavras = word_timestamps or []
            cur.executemany(
                """
                INSERT INTO vocal_words
                  (utterance_id, ordinal, surface, char_start, char_end,
                   chunk_index, t_start_ms, t_end_ms, confidence,
                   abs_timestamp, t_source, vector_energia)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                [
                    (
                        utterance_id, r["ordinal"], r["surface"], r["char_start"],
                        r["char_end"], r["chunk_index"], r["t_start_ms"],
                        r["t_end_ms"], r["confidence"],
                        ts_palavras[r["ordinal"]] if r["ordinal"] < len(ts_palavras) else None,
                        "real" if ts_palavras and r["ordinal"] < len(ts_palavras) else "estimado",
                        _pack_f32(band_profile(embedding)),
                    )
                    for r in word_rows
                ],
            )

        # fecha a contagem do conjunto para que a visão hierárquica não cresça
        cur.execute(
            """
            UPDATE vocal_segments s
               SET word_count = (SELECT count(*) FROM vocal_words w
                                  JOIN vocal_utterances u ON u.id = w.utterance_id
                                 WHERE u.segment_id = s.id),
                   utterance_count = (SELECT count(*) FROM vocal_utterances
                                       WHERE segment_id = s.id),
                   ended_at = COALESCE(s.ended_at, %s)
             WHERE s.id = %s
            """,
            (ended_at or when, seg_id),
        )
        conn.commit()
        return utterance_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


if __name__ == "__main__":
    import json as _json

    print(_json.dumps(verify_integrity(), indent=2, ensure_ascii=False, default=str))