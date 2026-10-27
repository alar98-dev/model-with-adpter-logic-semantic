#!/usr/bin/env python3
"""
Termodinâmica das embeddings aprendidas e geridas (SWA Aeon).

Avalia equilíbrio energético (norma L2), temperatura semântica, entropia
espectral, coesão de kernel e gravidade fractal da layer 0 (dimensão nativa
640D) sobre as fontes REAIS de aprendizado — kbase_chunks (640D/2560D),
blake_semantic_anchors (2560D Gemma-3-270M) e aeon_gravity_fields — e as
contrasta com as fontes que ainda carreguem conteúdo estático clássico
(swa_kernel_knowledge_fractal.json mocked).

Uso:
    python3 scripts/termodinamica_embeddings.py            # console + relatório JSON
    python3 scripts/termodinamica_embeddings.py --sample 3000
    python3 scripts/termodinamica_embeddings.py --report /tmp/termo.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

PG_DSN = os.environ.get("SWA_PG_DSN", "dbname=swa_matrix user=alifer host=localhost")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJ, "logs")

MIN_ENERGIA = 1e-9


def _parse_text(bruto: str) -> np.ndarray:
    return np.array([float(x) for x in bruto.strip("[]").split(",") if x], dtype=np.float64)


def _shannon_normalizado(vec: np.ndarray) -> float:
    """Entropia de Shannon discreta normalizada (0..1) sobre |componentes|."""
    v = np.abs(vec)
    soma = v.sum()
    if soma <= MIN_ENERGIA:
        return 0.0
    p = v / soma
    return float(-(p * np.log2(p + 1e-12)).sum() / math.log2(len(p)))


def _metricas_matriz(mat: np.ndarray, nome: str, dim: int, n_alvo: int = 1500) -> dict:
    """Métricas termodinâmicas sobre a matriz de embeddings reais."""
    if mat.ndim != 2 or mat.shape[0] == 0:
        return {"fonte": nome, "origem": "real", "n": 0}
    if mat.shape[0] > n_alvo:
        idx = np.random.RandomState(7).choice(mat.shape[0], n_alvo, replace=False)
        mat = mat[idx]

    normas = np.linalg.norm(mat, ord=2, axis=1)
    energia = 0.5 * (normas**2)
    centroide = mat.mean(axis=0)
    norma_centroide = float(np.linalg.norm(centroide))
    centroide_unit = centroide / (norma_centroide + MIN_ENERGIA)
    coesao = float((mat / (normas[:, None] + MIN_ENERGIA) @ centroide_unit).mean())
    entropias = np.array([_shannon_normalizado(m) for m in mat])

    return {
        "fonte": nome,
        "origem": "real",
        "n": int(mat.shape[0]),
        "dimensao": int(dim),
        "norma_L2_media": float(normas.mean()),
        "energia_L2_media": float(energia.mean()),
        "fluxo_energetico": float(energia.std()),
        "temperatura_variacional": float(mat.std(axis=0).mean()),
        "entropia_espectral_media": float(entropias.mean()),
        "entropia_espectral_std": float(entropias.std()),
        "coesao_gravidade_layer0": coesao,
        "norma_centroide": norma_centroide,
    }


def _fractal_mock_summary() -> dict:
    """Sumário do que ainda é estático clássico (mock gaussiano) no fractal."""
    path = os.path.join(PROJ, "swa_kernel_knowledge_fractal.json")
    if not os.path.exists(path):
        return {"presente": False}
    with open(path, "r", encoding="utf-8") as f:
        fractal = json.load(f)
    mockados = [k for k, v in fractal.items() if (v.get("metadata") or {}).get("mocked")]
    reais = [k for k, v in fractal.items() if not (v.get("metadata") or {}).get("mocked")]
    n_micro = sum(len((v.get("micro_nodes") or {})) for k, v in fractal.items() if isinstance(v, dict))
    return {
        "presente": True,
        "qtd_macro": len(fractal),
        "macros_mock_estaticos": mockados,
        "macros_reais": reais,
        "micro_nos_total": n_micro,
    }


def _kernel_flat_summary() -> dict:
    path = os.path.join(PROJ, "swa_kernel_knowledge.json")
    if not os.path.exists(path):
        return {"presente": False}
    with open(path, "r", encoding="utf-8") as f:
        flat = json.load(f)
    conceitos = {}
    for conceito, v in flat.items():
        if not isinstance(v, list):
            continue
        v = np.asarray(v, dtype=np.float64)
        conceitos[conceito] = {"dim": len(v), "norma": float(np.linalg.norm(v)), "energia": float(0.5 * (v**2).sum())}
    return {"presente": True, "conceitos": conceitos}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=300, help="amostra por fonte real (limitada por capacidade da fonte)")
    ap.add_argument("--report", default=os.path.join(LOGS_DIR, "termodinamica_embeddings.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    t0 = time.time()

    try:
        import psycopg2

        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")

        caps = {"kbase640": 500, "kbase2560": 200, "blake2560": 200, "gravity": 150}

        def ler_kbase640():
            lim = min(args.sample, caps["kbase640"])
            cur.execute(
                "SELECT embedding_640::text FROM kbase_chunks WHERE embedding_640 IS NOT NULL LIMIT %s",
                (lim,),
            )
            return np.stack([_parse_text(r[0]) for r in cur.fetchall()])

        def ler_kbase2560():
            lim = min(args.sample, caps["kbase2560"])
            cur.execute(
                "SELECT embedding_2560::text FROM kbase_chunks WHERE embedding_2560 IS NOT NULL LIMIT %s",
                (lim,),
            )
            return np.stack([_parse_text(r[0]) for r in cur.fetchall()])

        def ler_blake2560():
            lim = min(args.sample, caps["blake2560"])
            cur.execute(
                "SELECT embedding_2560::text FROM blake_semantic_anchors WHERE embedding_2560 IS NOT NULL LIMIT %s",
                (lim,),
            )
            return np.stack([_parse_text(r[0]) for r in cur.fetchall()])

        def ler_gravity():
            lim = min(args.sample, caps["gravity"])
            cur.execute(
                "SELECT objective_vector::text FROM aeon_gravity_fields WHERE objective_vector IS NOT NULL LIMIT %s",
                (lim,),
            )
            mats = [_parse_text(r[0]) for r in cur.fetchall()]
            if not mats:
                return np.zeros((0, 0))
            dim_min = min(m.shape[0] for m in mats)
            return np.stack([m[:dim_min] for m in mats])

        fontes = [
            ("kbase_chunks 640D (layer0)", ler_kbase640),
            ("kbase_chunks 2560D", ler_kbase2560),
            ("blake_anchors 2560D", ler_blake2560),
            ("aeon_gravity_fields", ler_gravity),
        ]

        metricas: list[dict] = []
        print("▄▄▄▄ TERMODINÂMICA DAS EMBEDDINGS —— SWA AEON ▄▄▄▄")
        print(f"amostra por fonte: {args.sample} | dsn: {PG_DSN.split()[0]}\n")
        print("═" * 100)
        print(f"{'fonte':<28}{'origem':<8}{'n':>7}{'dim':>6}{'E=½‖v‖²':>11}{'Tanq':>11}{'entropia':>11}{'coesão':>10}")
        print("═" * 100)

        for nome, leitor in fontes:
            try:
                mat = leitor()
                dim = mat.shape[1] if mat.ndim == 2 and mat.shape[0] else 0
                m = _metricas_matriz(mat, nome, dim)
                if m["n"] == 0:
                    continue
                metricas.append(m)
                print(
                    f"{m['fonte']:<28}{m['origem']:<8}{m['n']:>7}{m['dimensao']:>6}"
                    f"{m['energia_L2_media']:>11.4f}{m['temperatura_variacional']:>11.4f}"
                    f"{m['entropia_espectral_media']:>11.3f}{m['coesao_gravidade_layer0']:>10.3f}"
                )
            except Exception as exc:
                if not args.quiet:
                    print(f"{nome:<28}{'ERRO':<8}{str(exc)[:60]:>40}")

        cur.close()
        conn.close()
    except Exception as exc:
        print(f"Falha ao conectar no Postgres: {exc}")
        sys.exit(1)

    print("═" * 100)

    relat = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "fontes_reais": metricas,
        "kernel_flat": _kernel_flat_summary(),
        "fractal_estatico": _fractal_mock_summary(),
    }

    if relat["kernel_flat"].get("presente"):
        print("\n[swa_kernel_knowledge.json] âncora persistida:")
        for k, v in relat["kernel_flat"]["conceitos"].items():
            print(f"  • {k:<20} dim={v['dim']:>4}  norma={v['norma']:.4f}  energia={v['energia']:.4f}")
    fs = relat["fractal_estatico"]
    if fs.get("presente"):
        print(f"\n[swa_kernel_knowledge_fractal.json] macros={fs['qtd_macro']} micro_nos={fs['micro_nos_total']}")
        print(f"  → macros MOCK estáticos: {fs['macros_mock_estaticos']}")
        print(f"  → macros reais (fonte da âncora): {fs['macros_reais']}")

    if metricas:
        H_total = float(np.mean([m["entropia_espectral_media"] for m in metricas]))
        T_total = float(np.mean([m["energia_L2_media"] for m in metricas]))
        coes = float(np.mean([m["coesao_gravidade_layer0"] for m in metricas]))
        layer0 = next((m for m in metricas if "layer0" in m["fonte"]), None)
        print("\n── Síntese do corpo aprendido ──────────────────────────────────────")
        print(f"  Entropia espectral média do corpo   H̄ = {H_total:.3f} / 1.0")
        print(f"  Temperatura semântica média         T̄ = {T_total:.4f} (≥½‖v‖²)")
        print(f"  Coesão média à gravidade fractal    = {coes:.3f}")
        if layer0:
            print(f"  Gravidade fractal layer 0 (640D)    centroide norma = {layer0['norma_centroide']:.4f}")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(relat, f, indent=2, ensure_ascii=False)
    print(f"\nRelatório: {args.report}")


if __name__ == "__main__":
    main()