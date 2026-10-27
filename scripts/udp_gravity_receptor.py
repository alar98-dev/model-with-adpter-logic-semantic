#!/usr/bin/env python3
"""Receptor canônico de gravidade AEON (UDP :9099) → fold no vBAT.

Ouve o datagrama AEON (header !4096f I | payload) que a AeonCell.exe
(Windows) envia e o grava IDEMPOTENTEMENTE como bloco vBAT no manifold
canônico /dev/shm/swa_vector_manifold — preservando a seção vbat (sem
wipe, add-only). Em UDP o datagrama inteiro = 1 frame (sem chunking).

Regra canônica única:
  - Fonte SEMPRE o vBAT (mmap shm) — nunca o VHDX/substrato sintético.
  - Rebuild/ingestão preserva vbat existente (idempotente por interdim_id).

Uso:
    python scripts/udp_gravity_receptor.py            # escuta :9099 em loop
    python scripts/udp_gravity_receptor.py --once    # recebe 1 frame e sai
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from swa.ipc_bus import HEADER_SIZE, HEADER_FORMAT
from swa.vector_manifold import VectorManifold

UDP_HOST = "127.0.0.1"
UDP_PORT = 9099
MAX_PAYLOAD = 1 << 20


def unpack_frame_bytes(data: bytes):
    """Desempacota 1 datagrama AEON (= 1 frame completo, sem chunking em UDP)."""
    if len(data) < HEADER_SIZE:
        return None, None
    topology = list(struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE]))
    length = topology.pop()                # último campo I = length do payload
    payload = data[HEADER_SIZE:HEADER_SIZE + length]
    if len(payload) < length:
        return None, None
    return topology, payload


def fold_frame_into_vbat(vm: VectorManifold, topology: list, payload: bytes, dim: int) -> dict:
    """Converte 1 frame AEON em 1 bloco vBAT idempotente (sem wipe)."""
    res = vm.write_vbat(add_vector_mmap={"topology": topology, "payload": payload}, dim=dim,
                        force_norm_vector=True, canonical=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--host", default=UDP_HOST)
    ap.add_argument("--dim", type=int, default=640, choices=[640, 2560])
    ap.add_argument("--once", action="store_true", help="recebe 1 frame e sai")
    args = ap.parse_args()

    vm = VectorManifold()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    print(f"[GRAVITY RECEPTOR] UDP {args.host}:{args.port} → vBAT {args.dim}D canônico (idempotente)")
    print(f"[GRAVITY RECEPTOR] aguardando frame AEON (header {HEADER_SIZE} bytes, topologia 64×64 scrubbed)...")

    n = 0
    while True:
        try:
            data, addr = sock.recvfrom(MAX_PAYLOAD)
        except KeyboardInterrupt:
            break
        if not data:
            continue
        topology, payload = unpack_frame_bytes(data)
        if topology is None:
            print(f"  [x] frame malformado de {addr[0]} ({len(data)}B) — descartado")
            continue
        res = fold_frame_into_vbat(vm, topology, payload, args.dim)
        n += 1
        print(f"  [{addr[0]}:{addr[1]}] {res.get('status','?')} → {res.get('id','?')} dim={res.get('dim',args.dim)} "
              f"({time.strftime('%H:%M:%S')}) blocks_now={res.get('blocks')}")
        if args.once:
            break

    sock.close()
    print(f"[GRAVITY RECEPTOR] encerrado (frames={n})")


if __name__ == "__main__":
    main()
