"""Coletor do espaço Blake — extrai telemetria de todos os subespaços de gravidade.

Fontes consultadas (sem importar swa.*):
  1. BLAKE3 file signatures → escaneia arquivos-chave, computa hash
  2. SHM Vector Manifold   → /dev/shm/swa_vector_manifold/
  3. Kernel knowledge      → swa_kernel_knowledge.json (pesos 640D)
  4. HW state (fallback)   → /dev/shm/swa_hw_state.bin ou /dev/sde
  5. Topological DB        → acessa PostgreSQL matriz_vetorial diretamente
  6. Structural origins    → swa/logs/structural_origins.jsonl (últimos N blocos)

Self baseline: arquivos são comparados contra /etc/swa/obelisk_baseline.json.
Se o hash atual difere do baseline → conta como mutação (hash_entropy).
"""

import os
import json
import hashlib
import struct
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASELINE_PATH = os.environ.get("OBELISK_BASELINE", "/etc/swa/obelisk_baseline.json")

BLAKE_SOURCES = {
    "model": os.path.join(PROJECT_ROOT, "cortex/obelisk/model.py"),
    "train": os.path.join(PROJECT_ROOT, "cortex/obelisk/train.py"),
    "kernel_knowledge": os.path.join(PROJECT_ROOT, "swa_kernel_knowledge.json"),
    "blake_tools": os.path.join(PROJECT_ROOT, "swa/blake_tools.py"),
    "blake_shaper": os.path.join(PROJECT_ROOT, "swa/blake_shaper.py"),
    "vector_manifold": os.path.join(PROJECT_ROOT, "swa/vector_manifold.py"),
    "hardware_state": os.path.join(PROJECT_ROOT, "swa/hardware_state.py"),
    "eventual_state": os.path.join(PROJECT_ROOT, "swa/eventual_state.py"),
    "flux_coupling": os.path.join(PROJECT_ROOT, "swa/flux_coupling.py"),
    "topological_db": os.path.join(PROJECT_ROOT, "swa/topological_db.py"),
}

SHM_MANIFOLD_DIR = "/dev/shm/swa_vector_manifold"
HW_STATE_FILE = "/dev/shm/swa_hw_state.bin"
STRUCTURAL_LOG = os.path.join(PROJECT_ROOT, "swa/logs/structural_origins.jsonl")

_BASELINE_CACHE = None


def _load_baseline() -> dict:
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE
    try:
        with open(BASELINE_PATH) as f:
            _BASELINE_CACHE = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _BASELINE_CACHE = {}
    return _BASELINE_CACHE


def _blake3_file(path: str) -> str:
    import blake3 as _b3
    try:
        hasher = _b3.blake3()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def collect_file_signatures() -> dict:
    baseline = _load_baseline()
    sigs = {}
    mismatches = 0
    for name, path in BLAKE_SOURCES.items():
        if os.path.exists(path):
            sigs[name] = _blake3_file(path)
        else:
            sigs[name] = ""
        rel = os.path.relpath(path, PROJECT_ROOT)
        ref_hash = baseline.get(rel)
        current = sigs[name]
        if ref_hash and current and current != ref_hash:
            mismatches += 1
    sigs["_mismatch_count"] = mismatches
    sigs["_total"] = len(BLAKE_SOURCES)
    return sigs


def collect_shm_manifold() -> dict:
    result = {"exists": os.path.isdir(SHM_MANIFOLD_DIR), "nodes": 0, "files": {}}
    if not result["exists"]:
        return result
    try:
        for entry in os.listdir(SHM_MANIFOLD_DIR):
            fpath = os.path.join(SHM_MANIFOLD_DIR, entry)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                result["files"][entry] = {
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
                if entry == "manifest.json":
                    with open(fpath) as f:
                        manifest = json.load(f)
                        result["nodes"] = len(manifest) if isinstance(manifest, list) else 0
        return result
    except Exception:
        return result


def collect_hw_state() -> dict:
    result = {"device_exists": os.path.exists("/dev/sde"), "fallback_exists": os.path.exists(HW_STATE_FILE)}
    if result["fallback_exists"]:
        try:
            stat = os.stat(HW_STATE_FILE)
            result["fallback_bytes"] = stat.st_size
            result["fallback_mtime"] = stat.st_mtime
        except Exception:
            pass
    return result


def collect_structural_origins(max_lines: int = 50) -> dict:
    result = {"path": STRUCTURAL_LOG, "exists": os.path.exists(STRUCTURAL_LOG), "entries": 0, "concepts": {}}
    if not result["exists"]:
        return result
    try:
        concepts = {}
        with open(STRUCTURAL_LOG) as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    d = json.loads(line)
                    c = d.get("concept", "?")
                    concepts[c] = concepts.get(c, 0) + 1
                except json.JSONDecodeError:
                    pass
        result["entries"] = sum(concepts.values())
        result["concepts"] = concepts
        return result
    except Exception:
        return result


def collect_kernel_knowledge() -> dict:
    path = os.path.join(PROJECT_ROOT, "swa_kernel_knowledge.json")
    result = {"exists": os.path.exists(path), "keys": 0, "dimensions": {}}
    if not result["exists"]:
        return result
    try:
        with open(path) as f:
            data = json.load(f)
        result["keys"] = len(data)
        dims = {}
        for k, v in data.items():
            if isinstance(v, list):
                dims[k] = len(v)
        result["dimensions"] = dims
        return result
    except Exception:
        return result


def collect_all() -> dict:
    from cortex.obelisk.blake.opencode_tensor import collect_opencode_space
    opencode = collect_opencode_space()
    return {
        "timestamp": time.time(),
        "file_signatures": collect_file_signatures(),
        "shm_manifold": collect_shm_manifold(),
        "hw_state": collect_hw_state(),
        "structural_origins": collect_structural_origins(),
        "kernel_knowledge": collect_kernel_knowledge(),
        "opencode_space": opencode,
    }


def sha256_fingerprint(telemetry: dict) -> str:
    return hashlib.sha256(json.dumps(telemetry, sort_keys=True).encode()).hexdigest()[:16]


def space_signature(telemetry: dict) -> float:
    """Indicador de 'gravidade' — quão diferente o espaço está da referência.

    Quanto maior, mais o espaço blake está se distorcendo (anomalia gravitacional).
    Usa mismatch_count do self baseline + subespaços offline.
    """
    score = 0.0
    sigs = telemetry.get("file_signatures", {})
    mismatches = sigs.get("_mismatch_count", 0)
    total = sigs.get("_total", 1)
    score += (mismatches / total) * 5.0
    for k, v in sigs.items():
        if k.startswith("_"):
            continue
        if not v:
            score += 1.0
    shm = telemetry.get("shm_manifold", {})
    if not shm.get("exists"):
        score += 3.0
    hw = telemetry.get("hw_state", {})
    if not hw.get("device_exists") and not hw.get("fallback_exists"):
        score += 2.0
    return score
