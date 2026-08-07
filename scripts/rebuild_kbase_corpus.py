#!/usr/bin/env python3
"""Reconstrói o corpus K-Base a partir das âncoras da Matriz (multicamada).

Fontes (por padrão): CODE_BLOCK, HUMAN_EXPLANATION, CONFIG_SETTING,
WEIGHT_KNOWLEDGE, TEST_CASE, ANOMALY_DYNAMIC, TEST, AEON_RESPONSE_*,
WIKIPEDIA_PT_* (categorias pequenas).

Uso:
  python3 scripts/rebuild_kbase_corpus.py            # corpus curado
  python3 scripts/rebuild_kbase_corpus.py --include-wiki   # + WIKI_KNOWLEDGE (334k, demorado)
  python3 scripts/rebuild_kbase_corpus.py --limit 100      # teste rápido

Resumível: fontes já presentes (source_path+layer) são puladas.
Embeddings: via serviço swa-embedder (http://127.0.0.1:21000), 640D+2560D.
"""
from __future__ import annotations

import argparse
import sys
import time

ROOT = "/home/alifer/swa-aeon"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import psycopg2

from swa.kbase import KBaseEngine, EmbedClient, LAYERS

CURATED = [
    "CODE_BLOCK",
    "HUMAN_EXPLANATION",
    "CONFIG_SETTING",
    "WEIGHT_KNOWLEDGE",
    "TEST_CASE",
    "ANOMALY_DYNAMIC",
    "TEST",
]


def fetch_sources(include_wiki: bool, limit: int | None, scope: str | None = None) -> list[tuple]:
    conn = psycopg2.connect("dbname=swa_matrix user=alifer", connect_timeout=5)
    try:
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 300000")
        conds = []
        if scope == "seed":
            groups = CURATED
            add_aeon_wiki = False
        elif scope:
            groups = [g.strip() for g in scope.split(",") if g.strip()]
            add_aeon_wiki = True
        else:
            groups = CURATED
            add_aeon_wiki = True
        if groups:
            placeholders = ",".join(["%s"] * len(groups))
            conds.append(f"concept_group IN ({placeholders})")
        if add_aeon_wiki:
            conds.append("concept_group LIKE 'AEON_RESPONSE_%%'")
            conds.append("concept_group LIKE 'WIKIPEDIA_PT_%%'")
        if include_wiki:
            conds.append("concept_group = 'WIKI_KNOWLEDGE'")
        where = " OR ".join(f"({c})" for c in conds)
        sql = f"""
            SELECT file_path, concept_group, raw_content, weight
            FROM blake_semantic_anchors
            WHERE raw_content IS NOT NULL AND length(raw_content) >= 40
              AND ({where})
            ORDER BY length(raw_content) DESC
        """
        params: list = []
        if groups:
            params = list(groups)
        if include_wiki:
            pass
        if limit:
            sql += " LIMIT %s"
            params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


def existing_sources() -> set[str]:
    conn = psycopg2.connect("dbname=swa_matrix user=alifer", connect_timeout=5)
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source_path FROM kbase_chunks")
        rows = {r[0] for r in cur.fetchall()}
        cur.close()
        return rows
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-wiki", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scope", type=str, default=None,
                    help="'seed' (curado sem wiki/auto-memória) ou grupos separados por vírgula")
    args = ap.parse_args()

    client = EmbedClient(timeout=600)
    engine = KBaseEngine(embed_client=client, enabled=True)
    if not client.ready():
        print("❌ Embedder indisponível em :21000 — inicie swa-embedder.")
        sys.exit(1)
    print("✅ Embedder pronto —", client._post("/health", {}) or client.base)
    if args.include_wiki:
        # health check é GET; apenas loga
        pass

    sources = fetch_sources(args.include_wiki, args.limit, args.scope)
    print(f"Fontes candidatas: {len(sources)}")
    done = existing_sources()
    print(f"Já no K-Base (source_path): {len(done)}")
    todo = [s for s in sources if s[0] not in done]
    print(f"A processar: {len(todo)}")
    if not todo:
        print("Nada a fazer.")
        return

    t0 = time.time()
    n_chunks = 0
    n_ok = 0
    n_skip = 0
    BATCH = 16
    todo.sort(key=lambda s: len(s[2] or ""), reverse=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        texts = [s[2][:8000] for s in batch]
        try:
            pairs = client.embed_batch_pairs(texts)
        except Exception:
            pairs = None
        if pairs is None:
            n_skip += len(batch)
            continue
        for (path, cg, content, weight), pair in zip(batch, pairs):
            if pair is None or pair[0] is None:
                n_skip += 1
                continue
            try:
                n = engine.ingest_source(path, cg, content[:8000], float(weight or 1.0), embedder=lambda _t, _p=pair: _p)
                n_chunks += n
                n_ok += 1
            except Exception:
                n_skip += 1
        done_proc = i + len(batch)
        if done_proc % 200 == 0 or done_proc >= len(todo):
            el = time.time() - t0
            print(f"  [{done_proc}/{len(todo)}] ok={n_ok} chunks={n_chunks} em {el:.0f}s (~{el/max(done_proc,1):.2f}s/fonte)")

    print(f"\nFIM: {n_ok} fontes ingeridas, {n_chunks} chunks (camadas {LAYERS}).")
    engine._refresh_stats()
    print("Estado K-Base:", {k: engine.last_status[k] for k in ("chunks", "collections")})


if __name__ == "__main__":
    main()
