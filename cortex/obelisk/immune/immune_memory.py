"""Memória imunológica — registro persistente de todas as classificações.

Cada ciclo de classificação salva um arquivo JSON em
/var/lib/obelisk/immune_memory/ com timestamp, tensor, label, severidade.

Utilizado por:
  - dataset.py para incluir amostras reais no treino
  - immune_query.py para busca por similaridade de tensor
"""

import os
import json
import glob
import time

MEMORY_DIR = os.environ.get("OBELISK_IMMUNE_MEMORY", "/var/lib/obelisk/immune_memory")


def _ensure_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)


def save_classification(entry: dict):
    _ensure_dir()
    ts = entry.get("timestamp", time.time())
    label = entry.get("label", "nominal")
    filename = os.path.join(MEMORY_DIR, f"{ts:.0f}_{label}.json")
    try:
        with open(filename, "w") as f:
            json.dump(entry, f, indent=2)
    except Exception:
        pass


def load_recent(n: int = 500) -> list[dict]:
    _ensure_dir()
    files = sorted(glob.glob(os.path.join(MEMORY_DIR, "*.json")))
    files = files[-n:]
    entries = []
    for f in files:
        try:
            with open(f) as fp:
                entries.append(json.load(fp))
        except (json.JSONDecodeError, OSError):
            pass
    return entries


def query_similar(tensor: list[float], top_k: int = 5) -> list[dict]:
    entries = load_recent(1000)
    scored = []
    for e in entries:
        et = e.get("gravity_tensor", e.get("tensor", []))
        if not et or len(et) < 6:
            continue
        sim = 1.0 - sum(abs(a - b) for a, b in zip(tensor, et)) / 6.0
        scored.append((sim, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


def count_by_label(hours: int = 24) -> dict:
    cutoff = time.time() - hours * 3600
    entries = load_recent(10000)
    counts = {}
    for e in entries:
        ts = e.get("timestamp", 0)
        if ts < cutoff:
            continue
        label = e.get("label", "nominal")
        counts[label] = counts.get(label, 0) + 1
    return counts


def latest_tensor() -> list[float] | None:
    entries = load_recent(1)
    if entries:
        return entries[0].get("gravity_tensor", entries[0].get("tensor"))
    return None
