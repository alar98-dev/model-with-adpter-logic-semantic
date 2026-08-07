#!/usr/bin/env python3
"""Gera baseline BLAKE3 — mesmo algoritmo usado pelo collector.py."""
import os
import json
import sys

SWA_ROOT = "/services/model-with-adpter-logic-semantic"
OUTPUT = os.environ.get("OBELISK_BASELINE", "/etc/swa/obelisk_baseline.json")

FILES = [
    "cortex/obelisk/model.py",
    "cortex/obelisk/train.py",
    "cortex/obelisk/blake/collector.py",
    "cortex/obelisk/blake/tensor.py",
    "cortex/obelisk/blake/classifier.py",
    "cortex/obelisk/main.py",
    "cortex/obelisk/daemon.py",
    "cortex/obelisk/dataset.py",
    "swa/blake_tools.py",
    "swa/blake_shaper.py",
    "swa/vector_manifold.py",
    "swa/hardware_state.py",
    "swa/eventual_state.py",
    "swa/flux_coupling.py",
    "swa/topological_db.py",
]


def _blake3_file(path: str) -> str:
    try:
        import blake3 as _b3
        hasher = _b3.blake3()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        try:
            import hashlib
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""


def _sha256_file(path: str) -> str:
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def main():
    baseline_blake3 = {}
    baseline_sha256 = {}
    for rel in FILES:
        full = os.path.join(SWA_ROOT, rel)
        if os.path.exists(full):
            b3 = _blake3_file(full)
            s256 = _sha256_file(full)
            baseline_blake3[rel] = b3
            baseline_sha256[rel] = s256
            status = f"OK  {rel}  →  blake3={b3[:16]}... sha256={s256[:16]}..."
        else:
            baseline_blake3[rel] = ""
            baseline_sha256[rel] = ""
            status = f"--- {rel}  →  (nao encontrado)"
        print(status)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(baseline_blake3, f, indent=2)
    os.chmod(OUTPUT, 0o444)
    sha256_output = os.environ.get("OBELISK_BASELINE_SHA256", OUTPUT.replace(".json", ".sha256.json"))
    with open(sha256_output, "w") as f:
        json.dump(baseline_sha256, f, indent=2)
    os.chmod(sha256_output, 0o444)
    print(f"\nBaseline (blake3): {OUTPUT} ({len(baseline_blake3)} arquivos)")
    print(f"Baseline (sha256): {sha256_output} ({len(baseline_sha256)} arquivos)")


if __name__ == "__main__":
    main()
