#!/usr/bin/env python3
"""
Construtor da Block Allocation Table Semântica (vBAT) no manifold compartilhado.

Lê os vetores REAIS de aprendizado do pgvector (kbase_chunks 640D/2560D, blake
semantic_anchors 2560D, aeon_gravity_fields) e os projeta em blocos binários
contíguos float32 em /dev/shm/swa_vector_manifold (vectors_{dim}.bin + flags),
indexados por slot no interdim_index.json — acesso direto via mmap, sem TOAST.

Uso:
    python3 scripts/build_vbat.py                    # constrói 640D e 2560D
    python3 scripts/build_vbat.py --cap640 10000
    python3 scripts/build_vbat.py --bench 1000       # imprime bench DB vs mmap
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PG_DSN = os.environ.get("SWA_PG_DSN", "dbname=swa_matrix user=alifer host=localhost")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

DIM_640 = 640
DIM_2560 = 2560


def _parse_text(bruto: str) -> np.ndarray:
    return np.array([float(x) for x in bruto.strip("[]").split(",") if x], dtype=np.float32)


def _pad_dim(v: np.ndarray, dim: int) -> np.ndarray:
    if v.shape[0] == dim:
        return v
    if v.shape[0] > dim:
        return v[:dim]
    return np.pad(v, (0, dim - v.shape[0]))


def _ler_kbase(cur, col: str, dim: int, lim: int) -> tuple[list, list, list]:
    ids, vecs, labels = [], [], []
    cur.execute(
        f"SELECT id, {col}::text FROM kbase_chunks WHERE {col} IS NOT NULL LIMIT %s",
        (lim,),
    )
    for rid, bruto in cur.fetchall():
        ids.append(f"kbase:{rid}")
        vecs.append(_pad_dim(_parse_text(bruto), dim))
        labels.append(str(rid))
    return ids, vecs, labels


def _ler_blake(cur, lim: int, dim: int) -> tuple[list, list, list]:
    ids, vecs, labels = [], [], []
    cur.execute(
        "SELECT COALESCE(signature_hash, file_path), file_path, embedding_2560::text "
        "FROM blake_semantic_anchors WHERE embedding_2560 IS NOT NULL LIMIT %s",
        (lim,),
    )
    for sig, fp, bruto in cur.fetchall():
        ids.append(f"blake:{sig}")
        vecs.append(_pad_dim(_parse_text(bruto), dim))
        labels.append(fp)
    return ids, vecs, labels


def _ler_gravity(cur, lim: int, dim: int) -> tuple[list, list, list]:
    ids, vecs, labels = [], [], []
    cur.execute(
        "SELECT id, objective_vector::text FROM aeon_gravity_fields "
        "WHERE objective_vector IS NOT NULL LIMIT %s",
        (lim,),
    )
    for rid, bruto in cur.fetchall():
        ids.append(f"gravity:{rid}")
        vecs.append(_pad_dim(_parse_text(bruto), dim))
        labels.append(str(rid))
    return ids, vecs, labels


def _bench(conn, vm, bench: int, dim: int, db_sql: str, params: tuple) -> None:
    import psycopg2.extras

    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.execute(db_sql, params)
    linhas = cur.fetchall()

    t0 = time.perf_counter()
    for bruto, in linhas[:bench]:
        _ = _parse_text(bruto)
    t_db = time.perf_counter() - t0

    store = vm.open_vbat(dim)
    if store is None or store.shape[0] == 0:
        print("  (store vBAT vazio — benchmark mmap pulado)")
        return
    n = store.shape[0]
    idx = np.random.RandomState(7).choice(n, min(bench, n), replace=False)
    t1 = time.perf_counter()
    for i in idx:
        _ = store[int(i)]
    t_mmap = time.perf_counter() - t1

    print(
        f"  {dim:>5}D | DB-parsing ({min(bench, len(linhas))}): {t_db * 1000:8.2f} ms | "
        f"mmap-read ({len(idx)}): {t_mmap * 1000:8.2f} ms | "
        f"speedup: {t_db / max(t_mmap, 1e-12):7.1f}x"
    )
    cur.close()


def main():
    ap = argparse.ArgumentParser(description="Build da vBAT (manifold binário mmap)")
    ap.add_argument("--cap640", type=int, default=5000, help="blocos 640D de kbase_chunks")
    ap.add_argument("--cap2560", type=int, default=3000, help="blocos 2560D (kbase + blake)")
    ap.add_argument("--cap-gravity", type=int, default=830, help="blocos gravity fields")
    ap.add_argument("--bench", type=int, default=0, help="benchmark DB vs mmap (n)")
    args = ap.parse_args()

    from swa.vector_manifold import VectorManifold

    vm = VectorManifold()
    t0 = time.time()

    import psycopg2

    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")

    print("▄▄▄▄ BUILD vBAT — BLOCOS SEMÂNTICOS CONTÍGUOS (mmap) ▄▄▄▄")
    print(f"dsn: {PG_DSN.split()[0]}\n")

    ids640, vecs640, lab640 = _ler_kbase(cur, "embedding_640", DIM_640, args.cap640)
    gid, gvc, glab = _ler_gravity(cur, args.cap_gravity, DIM_640)
    ids640 += gid
    vecs640 += gvc
    lab640 += glab

    ids2560, vecs2560, lab2560 = _ler_kbase(cur, "embedding_2560", DIM_2560, args.cap2560)
    bid, bvc, blab = _ler_blake(cur, args.cap2560, DIM_2560)
    ids2560 += bid
    vecs2560 += bvc
    lab2560 += blab

    report = []
    if vecs640:
        mat640 = np.stack(vecs640)
        r640 = vm.write_vbat(DIM_640, mat640, ids640, "kbase_chunks+aeon_gravity_fields", lab640)
        report.append(("640D (layer0)", r640))
        print(f"  [640D] {r640['blocks']:>6} blocos  {r640['total_bytes']/1024:8.1f} KiB  "
              f"frozen={r640['state_dist']['1']} drift={r640['state_dist']['0']} decay={r640['state_dist']['2']}")
    if vecs2560:
        mat2560 = np.stack(vecs2560)
        r2560 = vm.write_vbat(DIM_2560, mat2560, ids2560, "kbase_chunks+blake_anchors", lab2560)
        report.append(("2560D", r2560))
        print(f"  [2560D] {r2560['blocks']:>6} blocos  {r2560['total_bytes']/1024:8.1f} KiB  "
              f"frozen={r2560['state_dist']['1']} drift={r2560['state_dist']['0']} decay={r2560['state_dist']['2']}")

    print("\n—— Índice de slots (amostra) ———————————————————————————")
    index = vm._load_id_index()
    vbat_i = index.get("vbat", {})
    for dim in ("640", "2560"):
        blocos = vbat_i.get(dim, {})
        if not blocos:
            continue
        amostra = list(blocos.items())[:3]
        for vid, e in amostra:
            print(f"  [{dim}] {vid:<70} slot={e['slot']:<6} state={e['state']}")

    if args.bench:
        print("\n—— Benchmark: pgvector(TOAST) vs mmap —————————————————————")
        _bench(conn, vm, args.bench, DIM_640,
               "SELECT embedding_640::text FROM kbase_chunks WHERE embedding_640 IS NOT NULL LIMIT %s",
               (args.cap640,))
        _bench(conn, vm, args.bench, DIM_2560,
               "SELECT embedding_2560::text FROM blake_semantic_anchors WHERE embedding_2560 IS NOT NULL LIMIT %s",
               (args.cap2560,))

    cur.close()
    conn.close()

    status = vm.vbat_status()
    print("\n—— Manifest vBAT ————————————————————————————————————————")
    print(json_str := __import__("json").dumps(status, ensure_ascii=False, indent=2))

    verif = vm.vbat_vector(ids640[0], DIM_640) if ids640 else None
    if verif is not None:
        print(f"\nVerificação slot 0: kbase:{ids640[0].split(':')[1]}")
        print(f"  norma L2 = {float(np.linalg.norm(verif)):.6f} (esperado ~1.0)")
        print(f"  via mmap em {time.time() - t0:.2f}s (total do build)")

    print("\nDone.")


if __name__ == "__main__":
    main()