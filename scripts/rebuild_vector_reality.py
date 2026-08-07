#!/usr/bin/env python3
"""
Reescreve a árvore de realidade (tree folder) no manifold vetorial compartilhado.

Uso:
  python3 scripts/rebuild_vector_reality.py
  python3 scripts/rebuild_vector_reality.py --roots /services,/home/alifer/.cursor/projects/.../agent-transcripts
  python3 scripts/rebuild_vector_reality.py --pg-only   # só Postgres
  python3 scripts/rebuild_vector_reality.py --shm-only  # só índice SHM (rápido)

Saída:
  /dev/shm/swa_vector_manifold/manifest.json
  /dev/shm/swa_vector_manifold/reality_tree.jsonl
  /dev/shm/swa_vector_manifold/interdim_index.json
  logs/structural_origins.jsonl (append por arquivo no bridge)
  blake_semantic_anchors no Postgres (pgvector 2560D)
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="Rebuild SWA vector reality manifold")
    parser.add_argument(
        "--roots",
        default="",
        help="Raízes comma-separated (default: política workspace_policy)",
    )
    parser.add_argument("--pg-only", action="store_true", help="Sincroniza só pgvector")
    parser.add_argument("--shm-only", action="store_true", help="Só índice SHM, sem Postgres")
    parser.add_argument("--dry-run", action="store_true", help="Só conta arquivos, não grava")
    parser.add_argument(
        "--skip-if-real",
        action="store_true",
        help="Sai com 0 se manifest já tem embedding_source gemma-3-270m_proj2560 (boot systemd)",
    )
    args = parser.parse_args()

    from swa.workspace_policy import get_watch_roots
    from swa.vector_manifold import VectorManifold, SHM_MANIFOLD_DIR

    if args.skip_if_real:
        existing = VectorManifold.load_manifest()
        if existing and existing.get("embedding_source") == "gemma-3-270m_proj2560":
            print("✅ [MANIFOLD] Índice SHM já com embeddings reais — skip.")
            return

    roots = [r.strip() for r in args.roots.split(",") if r.strip()] if args.roots else get_watch_roots()

    print("=" * 72)
    print("SWA — REESCRITA DA REALIDADE VETORIAL")
    print("=" * 72)
    print("Raízes:")
    for r in roots:
        print(f"  • {r}")
    print()

    manifold = VectorManifold()
    nodes = manifold.collect_tree(roots)
    print(f"Nós indexáveis: {len(nodes)}")

    if args.dry_run:
        from collections import Counter
        zones = Counter(n.zone for n in nodes)
        for z, c in zones.most_common():
            print(f"  [{z}] {c}")
        return

    if args.shm_only:
        manifest = manifold.write_shared_manifold(nodes, roots)
        print(f"SHM: {SHM_MANIFOLD_DIR} — {manifest['node_count']} nós")
        return

    if args.pg_only:
        from swa.quantum_bridge import QuantumPhaseBridge
        manifold.bridge = QuantumPhaseBridge()
        n = manifold.sync_to_pgvector(nodes)
        print(f"pgvector: {n} âncoras")
        return

    manifest = manifold.rebuild_reality(roots)
    print()
    print("Fonte embedding:", manifest.get("embedding_source"))
    print("Zonas:", manifest.get("zones"))
    print(f"Manifest: {SHM_MANIFOLD_DIR}/manifest.json")
    if manifest.get("embedding_source") != "gemma-3-270m_proj2560":
        print("⚠️ AVISO: rebuild não usou Gemma 270M. Verifique RAM e SWA_REAL_EMBEDDINGS=1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
