"""
migrar_hierarquia.py — sobe acoustic_chunk_snapshots para a hierarquia de tensores
================================================================================
Os 263 chunks já carregam, por palavra, o que o padrão v2 exige:
`abs_timestamp` (instante real, não estimativa), `start_time_rel`/`end_time_rel`,
`energy_rms`, `z_score`, `mfcc_signature` (13 floats) e `latent_tensor_b64`
(o tensor binário). Este script apenas reorganiza isso em três níveis:

    vocal_segments  (frases agrupadas por sessão)
      └─ vocal_utterances  (uma por chunk, ligado por acoustic_chunk_id)
             └─ vocal_words  (uma por palavra, com instante real)

Chaves:
  * o vetor acústico vai como `bytea` float32 little-endian — é assim que o
    PostgreSQL o devolve sem parse, e o JSONB fica só como sombra legível;
  * a âncora de identidade continua sendo a FK para swa_identities. Chunks com
    locutor não cadastrado ('escritorio', 'Desconhecido') ficam de fora em vez
    de receber um dono inventado;
  * é idempotente: rodar de novo não duplica, porque utterance por chunk é
    único e as palavras são substituídas junto com a frase.

Uso:  ./venv/bin/python3 migrar_hierarquia.py [--dry-run] [--gap-min 20]
"""
import os
import sys
import json
import base64
import argparse
import datetime as dt

import numpy as np
import psycopg2
import psycopg2.extras

DSN = os.environ.get(
    "SWA_DSN",
    "host=localhost port=5432 dbname=swa_matrix user=alifer password=swa_alifer_2026",
)
BANDAS = 16
GAP_MIN = 20  # minutos sem fala que fecham uma sessão


def pack(vec) -> bytes | None:
    """float64 -> bytes float32 little-endian."""
    if vec is None or len(vec) == 0:
        return None
    return np.asarray(vec, dtype=np.float32).astype("<f4").tobytes()


def unpack_tensor_b64(b64) -> bytes | None:
    """O tensor binário que já vinha em base64 vira bytea cru."""
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def espectro_energia(rms: float) -> list[float] | None:
    """Perfil degenerado: a energia global vira 16 bandas iguais em log10.

    Só preenche a coluna para manter a forma do vetor idêntica em todos os
    níveis da hierarquia. A representação por banda de verdade vem do cálculo
    do pipeline vivo (leitor_sincronizado.perfil_tensorial).
    """
    if rms is None:
        return None
    return [float(np.log10(max(rms, 1e-10)))] * BANDAS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gap-min", type=int, default=GAP_MIN)
    args = ap.parse_args()

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT id, identity_id, speaker_identified, confidence_score,
               transcribed_text, duration_seconds, created_at, word_snapshots
          FROM acoustic_chunk_snapshots
         WHERE identity_id IS NOT NULL
           AND jsonb_array_length(word_snapshots) > 0
         ORDER BY created_at
    """)
    chunks = cur.fetchall()
    print(f"chunks elegíveis: {len(chunks)}")
    if not chunks:
        print("nada a migrar")
        return

    # ---- agrupar em sessões: uma gap de silêncio fecha o conjunto ----------
    grupos, atual = [], []
    for c in chunks:
        if atual:
            gap = (c["created_at"] - atual[-1]["created_at"]).total_seconds() / 60.0
            if gap > args.gap_min:
                grupos.append(atual)
                atual = []
        atual.append(c)
    if atual:
        grupos.append(atual)
    print(f"sessões detectadas: {len(grupos)} "
          f"(gap de {args.gap_min} min fecha um conjunto)")

    n_pal = n_fr = 0
    vistos = set()

    for gi, grupo in enumerate(grupos, 1):
        session_key = (f"mig-{grupo[0]['created_at'].strftime('%Y%m%dT%H%M%S')}"
                       f"-{grupo[-1]['created_at'].strftime('%H%M%S')}")
        ini = grupo[0]["created_at"]
        fim = grupo[-1]["created_at"] + dt.timedelta(
            seconds=float(grupo[-1]["duration_seconds"] or 0))

        if args.dry_run:
            print(f"  [dry] sessão {gi}: {session_key}  "
                  f"{len(grupo)} frases  {ini:%d/%m %H:%M}")
            continue

        cur.execute("""
            INSERT INTO vocal_segments
              (identity_id, session_key, started_at, ended_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_key) DO UPDATE
              SET ended_at = EXCLUDED.ended_at,
                  identity_id = EXCLUDED.identity_id
            RETURNING id
        """, (grupo[0]["identity_id"], session_key, ini, fim))
        seg_id = cur.fetchone()["id"]

        soma_palavras = 0
        vetores_frases = []
        for ch in grupo:
            cid = ch["id"]
            if cid in vistos:
                continue
            vistos.add(cid)

            palavras = [w for w in ch["word_snapshots"] if w.get("word")]
            palavras.sort(key=lambda w: w.get("word_index", 0))
            if not palavras:
                continue

            ts = [w.get("abs_timestamp") for w in palavras if w.get("abs_timestamp")]
            fala_ini = dt.datetime.fromtimestamp(min(ts)) if ts else ch["created_at"]
            fala_fim = dt.datetime.fromtimestamp(max(ts)) if ts else fim
            dur_ms = ((fala_fim - fala_ini).total_seconds() * 1000.0
                      if ts else (ch["duration_seconds"] or 0) * 1000.0)

            mfccs = [w["mfcc_signature"] for w in palavras if w.get("mfcc_signature")]
            media_mfcc = list(np.mean(np.asarray(mfccs, dtype=np.float64), axis=0)) if mfccs else None
            rmss = [w["energy_rms"] for w in palavras if w.get("energy_rms") is not None]
            rms_medio = float(np.mean(rmss)) if rmss else None
            if media_mfcc:
                vetores_frases.append(media_mfcc)

            cur.execute("""
                INSERT INTO vocal_utterances
                  (identity_id, segment_id, acoustic_chunk_id, text, source,
                   captured_at, similarity, is_owner, energy, duration_ms,
                   timing_source, started_at, ended_at, text_hash,
                   vector_acoustic, vector_energia, shadow)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (acoustic_chunk_id) WHERE acoustic_chunk_id IS NOT NULL
                DO UPDATE
                  SET segment_id      = EXCLUDED.segment_id,
                      started_at      = EXCLUDED.started_at,
                      ended_at        = EXCLUDED.ended_at,
                      duration_ms     = EXCLUDED.duration_ms,
                      similarity      = EXCLUDED.similarity,
                      vector_acoustic = EXCLUDED.vector_acoustic,
                      vector_energia  = EXCLUDED.vector_energia,
                      shadow          = EXCLUDED.shadow
                RETURNING id
            """, (
                ch["identity_id"], seg_id, cid, ch["transcribed_text"],
                "acoustic_chunk", ch["created_at"],
                float(ch["confidence_score"] or 0), True,
                rms_medio, dur_ms,
                "real" if ts else "estimado", fala_ini, fala_fim,
                ch["transcribed_text"][:80],
                pack(media_mfcc), pack(espectro_energia(rms_medio)),
                json.dumps({"speaker_identified": ch["speaker_identified"],
                            "n_palavras": len(palavras)}, ensure_ascii=False),
            ))
            utt_id = cur.fetchone()["id"]
            n_fr += 1
            soma_palavras += len(palavras)

            # substitui as palavras para manter a frase e suas palavras 1:1
            cur.execute("DELETE FROM vocal_words WHERE utterance_id = %s", (utt_id,))

            rows = []
            for i, w in enumerate(palavras):
                t_ini = (w.get("start_time_rel") or 0.0) * 1000.0
                t_fim = (w.get("end_time_rel") or 0.0) * 1000.0
                rows.append({
                    "utterance_id": utt_id, "ordinal": i, "surface": w.get("word", ""),
                    "chunk_index": i, "t_start_ms": t_ini, "t_end_ms": t_fim,
                    "abs_timestamp": w.get("abs_timestamp"),
                    # A origem do tempo vem do próprio snapshot. Snapshots
                    # antigos não têm `t_source` e foram cortados por divisão
                    # uniforme, então caem em "estimado" — ter `abs_timestamp`
                    # não torna o instante medido.
                    "t_source": w.get("t_source") or "estimado",
                    "confidence": w.get("z_score"), "energy_rms": w.get("energy_rms"),
                    "z": w.get("z_score"),
                    "vector_acoustic": pack(w.get("mfcc_signature")),
                    "vector_energia": pack(espectro_energia(w.get("energy_rms"))),
                    "tensor_acoustic": unpack_tensor_b64(w.get("latent_tensor_b64")),
                })
            if rows:
                psycopg2.extras.execute_batch(cur, """
                    INSERT INTO vocal_words
                      (utterance_id, ordinal, surface, chunk_index,
                       t_start_ms, t_end_ms, abs_timestamp, t_source,
                       confidence, energy_rms, z_score,
                       vector_acoustic, vector_energia, tensor_acoustic)
                    VALUES (%(utterance_id)s,%(ordinal)s,%(surface)s,%(chunk_index)s,
                            %(t_start_ms)s,%(t_end_ms)s,%(abs_timestamp)s,%(t_source)s,
                            %(confidence)s,%(energy_rms)s,%(z)s,
                            %(vector_acoustic)s,%(vector_energia)s,%(tensor_acoustic)s)
                """, rows)
                n_pal += len(rows)

        # ---- agregação do conjunto: herda a média das frases ---------------
        if vetores_frases:
            media = list(np.mean(np.asarray(vetores_frases), axis=0))
            cur.execute("""
                UPDATE vocal_segments
                   SET word_count = %s, utterance_count = %s,
                       vector_acoustic = %s, vector_energia = %s,
                       shadow = %s
                 WHERE id = %s
            """, (soma_palavras, len(vetores_frases), pack(media),
                  pack([0.0] * BANDAS),
                  json.dumps({"agregado": "media das frases",
                              "dim": len(media)}, ensure_ascii=False), seg_id))
        conn.commit()
        print(f"  sessão {gi}: {session_key}  "
              f"{len(vetores_frases)} frases · {soma_palavras} palavras")

    if args.dry_run:
        conn.rollback()
        print("\n🧪 dry-run: nada gravado.")
        return
    print(f"\n✅ {n_fr} frases e {n_pal} palavras escritas na hierarquia")
    conn.close()


if __name__ == "__main__":
    main()
