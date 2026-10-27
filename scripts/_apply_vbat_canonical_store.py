#!/usr/bin/env python3
"""Aplica a trava vBAT CANÔNICA DEFINITIVA no swa/vector_manifold.py.

Objetivo (contrato do usuário):
  - vBAT tem FONTE ÚNICA num STORE dedicado (add-only, archivado em forma de
    bloco: topology 4096f + payload), fora de interdim_index.json.
  - write_shared_manifold/spell_rebase NUNCA apaga o vBAT: o índice SHM só
    ESPELHA o store (fingerprint sha256 + dims). O vBAT vive no STORE.
  - Rebuild/rebuild_real NUNCA toca o STORE (é add-only).
  - Idempotência por interdim_id (bloco já presente => skip, sem duplicar).

Sem rewrite do código de rebuild: apenas injeta os helpers + faz
write_shared_manifold FONTEAR o vBAT do store dedicado.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "swa" / "vector_manifold.py"


def apply() -> None:
    src = TARGET.read_text(encoding="utf-8")

    # 1) constantes do STORE dedicado (fora do alcance de qualquer rebuild)
    if "VBAT_STORE_DIR" not in src:
        src = src.replace(
            'VBAT_DIMS = (640, 2560)',
            'VBAT_DIMS = (640, 2560)\n'
            '# ---- vBAT STORE CANÔNICO (fonte única; ADD-ONLY; NUNCA tocado por rebuild) ----\n'
            'VBAT_STORE_DIR = SHM_MANIFOLD_DIR / "vbat_store"\n'
            'VBAT_STORE_FILE = {d: VBAT_STORE_DIR / f"vbat_{d}.bin" for d in VBAT_DIMS}\n'
            'VBAT_STORE_MANIFEST = VBAT_STORE_DIR / "vbat_manifest.json"\n',
        )

    # 2) helpers idempotentes de persistência no store (injetados antes de _load_id_index)
    if "def _vbat_store_entry_count" not in src:
        src = src.replace(
            '    def _load_id_index(self) -> dict:',
            '    # ---- helpers vBAT STORE canônico (add-only, idempotentes) ----\n'
            '    def _vbat_store_entry_count(self, dim: int) -> int:\n'
            '        p = VBAT_STORE_FILE[dim]\n'
            '        return 0 if not p.is_file() else p.stat().st_size // (4 * 4096 + 4)\n'
            '\n'
            '    def _vbat_has_interdim(self, dim: int, interdim_id: str) -> bool:\n'
            '        try:\n'
            '            mans = json.loads(VBAT_STORE_MANIFEST.read_text(encoding="utf-8"))\n'
            '        except Exception:\n'
            '            return False\n'
            '        return any(m.get("interdim_id") == interdim_id for m in mans)\n'
            '\n'
            '    def write_vbat_store_block(self, dim: int, topology: list, payload: bytes) -> dict:\n'
            '        """Grava 1 bloco no STORE dedicado — add-only, idempotente por interdim_id.\n'
            '\n'
            '        Este é o ÚNICO write do vBAT: nunca existe wipe aqui (append de 1 frame\n'
            '        canônico), e nenhum rebuild/rebase reescreve o arquivo.\n'
            '        """\n'
            '        VBAT_STORE_DIR.mkdir(parents=True, exist_ok=True)\n'
            '        interdim_id = "vbat:" + hashlib.sha256(repr(topology[:64]).encode()).hexdigest()[:24]\n'
            '        if self._vbat_has_interdim(dim, interdim_id):\n'
            '            return {"status": "already_present", "id": interdim_id, "dim": dim,\n'
            '                    "blocks": self._vbat_store_entry_count(dim)}\n'
            '        with open(VBAT_STORE_FILE[dim], "ab") as f:\n'
            '            f.write(struct.pack("!4096f I", *topology, len(payload)))\n'
            '            f.write(payload)\n'
            '        mans = []\n'
            '        if VBAT_STORE_MANIFEST.is_file():\n'
            '            try:\n'
            '                mans = json.loads(VBAT_STORE_MANIFEST.read_text(encoding="utf-8"))\n'
            '            except Exception:\n'
            '                mans = []\n'
            '        mans.append({"interdim_id": interdim_id, "dim": dim, "blocks": len(mans) + 1,\n'
            '                     "sha256": hashlib.sha256(bytes(repr(topology[:64]), "utf-8")).hexdigest()[:16],\n'
            '                     "at": time.time()})\n'
            '        VBAT_STORE_MANIFEST.write_text(json.dumps(mans, indent=2), encoding="utf-8")\n'
            '        return {"status": "stored", "id": interdim_id, "dim": dim,\n'
            '                "blocks": self._vbat_store_entry_count(dim)}\n'
            '\n'
            '    def _load_id_index(self) -> dict:',
        )

    # 3) write_shared_manifold: FONTE do vBAT == STORE dedicado (espelho do manifesto),
    #    nunca do índice (que rebuild pode sobrescrever).
    src = src.replace(
        '        vbat_man = (self.load_manifest() or {}).get("vbat") or {}',
        '        vbat_man = (self.load_manifest() or {}).get("vbat") or {}',
    )
    if "VBAT_STORE_MANIFEST" not in src:
        src = src.replace(
            '        vbat_man = (self.load_manifest() or {}).get("vbat") or {}\n'
            '        if vbat_man:\n'
            '            manifest["vbat"] = vbat_man',
            '        # --- vBAT canônico: FONTE ÚNICA = STORE dedicado (add-only). ---\n'
            '        # O índice SHM é só um espelho (fingerprint). Rebuild nunca apaga isto.\n'
            '        store_vbat = {}\n'
            '        if VBAT_STORE_MANIFEST.is_file():\n'
            '            try:\n'
            '                store_vbat = json.loads(VBAT_STORE_MANIFEST.read_text(encoding="utf-8"))\n'
            '            except Exception:\n'
            '                store_vbat = {}\n'
            '        if store_vbat:\n'
            '            manifest["vbat"] = {"_store_manifest": store_vbat,\n'
            '                                "blocks": sum(m.get("blocks", 1) for m in store_vbat),\n'
            '                                "source": "canonical_vbat_store"}\n'
            '        elif vbat_man:\n'
            '            manifest["vbat"] = vbat_man',
        )

    TARGET.write_text(src, encoding="utf-8")
    print(f"[CANÔNICO] trava vBAT store aplicada em {TARGET}")


if __name__ == "__main__":
    apply()
