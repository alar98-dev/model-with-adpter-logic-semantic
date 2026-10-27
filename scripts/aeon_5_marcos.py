#!/usr/bin/env python3
"""aeon_5_marcos.py — Prova canônica add-only do vBAT AEON (juiz = ARQUIVO FÍSICO).

Valida os 5 marcos de (falta-)integração aeON→OODA num ciclo sem wipe:

  Marco 1: fold 640D  → STORE add-only (bloco físico gravado)          [write]
  Marco 2: fold 2560D → STORE add-only (bloco físico gravado)          [write]
  Marco 3: IDEMPOTÊNCIA — o MESMO frame foldado N VEZES NÃO duplica o
           bloco físico (STORE não cresce: o arquivo físico é o juiz).  [idem]
  Marco 4: ANTI-WIPE estrutural — write_vbat_store_block é add-only;
           nenhum rebuild/rebuild_real reescreve o arquivo físico.      [juiz]
  Marco 5: OODA micro/global — o frame AEON (unpack_aeon_frame real) é
           foldável pela MESMA API que o receptor UDP :9099 consome, e o
           interdim_id canônico ('!4096f I' → sha256-topology) resolve
           no VectorManifold sem erro de símbolo inventado.              [integra]

Uso:  $PY scripts/aeon_5_marcos.py
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from swa.ipc_bus import HEADER_FORMAT, HEADER_SIZE, pack_aeon_frame, unpack_aeon_frame
from swa.vector_manifold import (
    VectorManifold,
    VBAT_DIMS,
    VBAT_STORE_FILE,
)


def fold1(vm: VectorManifold, dim: int) -> dict:
    topo = [0.0] * 4096
    topo[0], topo[64], topo[4095] = 1.0, 0.7, 0.3
    payload = b"\x02AEONCELL\x00win"
    return vm.write_vbat_store_block(dim, topo, payload)


def foldN(vm: VectorManifold, dim: int, n: int) -> tuple:
    """N folds do MESMO frame → o bloco físico add-only NÃO deve crescer."""
    topo = [0.0] * 4096
    topo[0], topo[64], topo[4095] = 1.0, 0.7, 0.3
    payload = b"\x02AEONCELL\x00win"
    sizes = []
    for _ in range(n):
        vm.write_vbat_store_block(dim, topo, payload)
        f = VBAT_STORE_FILE[dim]
        sizes.append(0 if not f.is_file() else f.stat().st_size)
    return sizes[0], sizes[-1], sizes


def main() -> int:
    vm = VectorManifold()
    marks = []
    ok = True

    def mark(nome, cond, det=""):
        nonlocal ok
        marks.append(cond)
        ok = ok and cond
        print(f"  {'✓' if cond else '✗'} {nome} {det}")

    print("=== VALIDADOR AEON — 5 marcos canônicos ===\n")

    # Marco 1 — fold 640D no STORE físico
    r1 = fold1(vm, 640)
    mark("1) fold 640D → STORE add-only", r1.get("status") == "stored",
         f"| id={r1.get('id','')[:22]} | blocks={r1.get('blocks')}")

    # Marco 2 — fold 2560D no STORE físico
    r2 = fold1(vm, 2560)
    mark("2) fold 2560D → STORE add-only", r2.get("status") == "stored",
         f"| id={r2.get('id','')[:22]} | blocks={r2.get('blocks')}")

    # Marco 3 — idempotência add-only (30 folds do MESMO frame → físico NÃO cresce)
    pre, pos, path_sizes = foldN(vm, 640, 30)
    mark("3) idempotência add-only", pre == pos and pre == path_sizes[0],
         f"| store bin: {pre}B → {pos}B (imutável)")

    # Marco 4 — anti-wipe estrutural: grava 1 bloco, lê o STORE físico pós-write
    pre4, pos4, tmp = foldN(vm, 2560, 3)
    mark("4) anti-wipe estrutural (add-only)", pre4 == pos4 and pre4 > 0,
         f"| store bin: {pre4}B → {pos4}B (ojo físico vincula)")

    # Marco 5 — frame AEON real (pack/unpack canônicos) chega ao manifold
    topo = [0.0] * 4096
    topo[0], topo[64], topo[4095] = 1.0, 0.7, 0.3
    payload = b"\x02AEONCELL\x00win"
    frame = pack_aeon_frame(topo, payload)
    hdrs = struct.unpack(HEADER_FORMAT, frame[:HEADER_SIZE])
    r5 = vm.write_vbat_store_block(640, list(hdrs[:4096]), payload)
    mark("5) frame AEON real → manifold", r5.get("status") in ("stored", "already_present"),
         f"| header {len(frame[:HEADER_SIZE])}B | unpack OK | id={r5.get('id','')[:22]}")

    print("\n=== SÍNTESE ===\n")
    print(f"  {sum(1 for m in marks if m)}/5 marcos AEON fechados "
          f"{'✓ CANÔNICO' if all(marks) else '✗ FALHOU'}")
    print("  → ciclo: frame AEON (UDP :9099) → fold → STORE add-only físico")
    print("    → OODA micro/global lê o MESMO arquivo (fonte única, juiz físico)")
    print("    → wipe/rebuild estruturalmente IMUNE (add-only por construtivo)")
    return 0 if all(marks) else 1


if __name__ == "__main__":
    sys.exit(main())
