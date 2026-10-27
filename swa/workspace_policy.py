"""
Política de varredura do workspace — raiz global Debian por zonas seguras.
"""
from __future__ import annotations

import os
from pathlib import Path

from swa.paths import PROJECT_ROOT

# Zonas Debian (sem varrer /home inteiro — evita EMFILE no .vscode-server)
DEBIAN_GLOBAL_ZONES = (
    "/services",
    "/opt",
)

# Nata da sabedoria: histórico de conversas Cursor Agent neste projeto
AGENT_WISDOM_ROOT = Path(
    "/home/alifer/.cursor/projects/services-model-with-adpter-logic-semantic/agent-transcripts"
)

# Projetos opcionais sob home (adicione via SWA_HOME_ZONES=/path1,/path2)
DEFAULT_HOME_ZONES = (
    str(AGENT_WISDOM_ROOT),
)

PROJECT_CORE_ZONES = (
    "cortex",
    "swa",
    "sensors",
    "prompts",
    "actuators",
    "scripts",
)

ANOMALOUS_PREFIXES = (
    "/proc", "/sys", "/dev", "/run", "/tmp", "/var/run",
    "/var/lib/docker", "/snap", "/boot", "/lost+found",
    "/usr", "/var", "/lib", "/lib64",
)

# Segmentos de caminho proibidos (IDE, caches, histórico local do editor)
FORBIDDEN_PATH_SEGMENTS = (
    ".vscode-server",
    ".cursor-server",
    "/History/",
    "/.history/",
    "site-packages",
    "node_modules",
    "/.npm/",
    "/.cache/",
    "/.local/share/",
    "/Caches/",
)

EXCLUDE_DIR_NAMES = frozenset({
    ".git", "__pycache__", "venv", ".venv", "models", ".kilo",
    "input_frames", "logs", "deployments", "node_modules", "dist", "build",
    "target", "out", "bin", "obj", ".next", ".nuxt", ".idea", ".vscode",
    "site-packages", "wheels", "cache", "Caches", "AppData",
    "docs", "tests", "test", "testing", "mcp_gateway", "mcp",
    "History", ".vscode-server", ".cursor-server", ".npm", ".cache",
})

SKIP_FILE_NAMES = frozenset({
    "swa_Aeon.log", "swa_PENSAMENTO_PURO.log", "server.log", "sonhador.log",
    "speaker_signatures.json", "aeon_sensory.log", "conftest.py",
})

SKIP_FILE_PREFIXES = (
    "benchmark_", "SWA_EVENTUAL", "swa_chaos", "test_", "smoke_",
)

SKIP_FILE_SUFFIXES = ("_test.py",)

EXT_CODE = (".py", ".sql")
EXT_CONFIG = (".json", ".txt", ".env", ".cfg")
EXT_WISDOM = (".jsonl", ".md")
EXT_DOCS_ALLOWED_DIRS = frozenset({"prompts"})

# Limite de arquivos rastreados em RAM pelo watcher
MAX_TRACKED_FILES = int(os.environ.get("SWA_MAX_TRACKED_FILES", "400"))


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _home_extra_zones() -> list[str]:
    extra = os.environ.get("SWA_HOME_ZONES", "").strip()
    if extra:
        return [p.strip() for p in extra.split(",") if p.strip()]
    return [z for z in DEFAULT_HOME_ZONES if os.path.isdir(z)]


def get_watch_roots() -> list[str]:
    custom = os.environ.get("SWA_WATCH_ROOTS", "").strip()
    if custom:
        return [p.strip() for p in custom.split(",") if p.strip()]

    if _env_truthy("SWA_PROJECT_ONLY"):
        return [str(PROJECT_ROOT / z) for z in PROJECT_CORE_ZONES if (PROJECT_ROOT / z).is_dir()]

    roots: list[str] = []
    for z in DEBIAN_GLOBAL_ZONES:
        if os.path.isdir(z):
            roots.append(z)
    roots.extend(_home_extra_zones())
    if _env_truthy("SWA_INCLUDE_ETC", default=False) and os.path.isdir("/etc"):
        roots.append("/etc")
    for z in PROJECT_CORE_ZONES:
        p = PROJECT_ROOT / z
        if p.is_dir():
            roots.append(str(p))
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out or [str(PROJECT_ROOT)]


def describe_watch_policy() -> str:
    roots = get_watch_roots()
    mode = "PROJECT_ONLY" if _env_truthy("SWA_PROJECT_ONLY") else "RAIZ_GLOBAL_DEBIAN"
    wisdom = "sim" if any("agent-transcripts" in r for r in roots) else "não"
    return (
        f"🌐 [STRUCTURAL SYNC] Modo {mode} — {len(roots)} zonas | "
        f"sabedoria agentes: {wisdom} | max_arquivos: {MAX_TRACKED_FILES}"
    )


def _path_under_anomalous(path: str) -> bool:
    p = path.replace("\\", "/")
    if any(p.startswith(prefix) for prefix in ANOMALOUS_PREFIXES):
        return True
    if any(seg in p for seg in FORBIDDEN_PATH_SEGMENTS):
        return True
    return False


def should_skip_dir(dirname: str) -> bool:
    if dirname in EXCLUDE_DIR_NAMES:
        return True
    if dirname.startswith("."):
        return True
    if dirname.endswith(".orig"):
        return True
    return False


def is_agent_wisdom_path(filepath: str) -> bool:
    return "agent-transcripts" in filepath.replace("\\", "/")


def should_index_file(filepath: str) -> bool:
    if _path_under_anomalous(filepath):
        return False

    p = Path(filepath)
    name = p.name
    if name in SKIP_FILE_NAMES:
        return False
    if any(name.startswith(pref) for pref in SKIP_FILE_PREFIXES):
        return False
    if any(name.endswith(suf) for suf in SKIP_FILE_SUFFIXES):
        return False

    suffix = p.suffix.lower()
    parts = set(p.parts)

    if is_agent_wisdom_path(filepath) and suffix in EXT_WISDOM:
        return True

    if suffix in EXT_CODE:
        return True

    if suffix == ".md":
        if "docs" in parts:
            return False
        parent_names = {x.name for x in p.parents}
        if parent_names & EXT_DOCS_ALLOWED_DIRS:
            return True
        if "swa" in parts and "cortex" not in parts:
            return True
        return False

    if suffix in EXT_CONFIG:
        if "prompts" in parts or "swa" in parts:
            return True
        return False

    return False


def get_crawler_roots() -> list[str]:
    if os.environ.get("SWA_CRAWLER_EXTRA", "").strip():
        extra = [x.strip() for x in os.environ["SWA_CRAWLER_EXTRA"].split(",") if x.strip()]
        return get_watch_roots() + extra
    return get_watch_roots()


def classify_zone(filepath: str) -> str:
    """Rótulo de zona para o manifold (árvore de realidade)."""
    p = filepath.replace("\\", "/")
    if is_agent_wisdom_path(p):
        return "agent_wisdom"
    if str(PROJECT_ROOT) in p:
        rel = p.split(str(PROJECT_ROOT).rstrip("/"), 1)[-1].lstrip("/")
        top = rel.split("/")[0] if rel else "root"
        return top or "project"
    if p.startswith("/services"):
        return "services"
    if p.startswith("/opt"):
        return "opt"
    if p.startswith("/etc"):
        return "etc"
    return "debian"


def iter_indexable_files(roots: list[str] | None = None, max_per_root: int = 5000):
    """Varre zonas permitidas e produz caminhos indexáveis (generator)."""
    if roots is None:
        roots = get_watch_roots()
    for root in roots:
        if not os.path.isdir(root) or _path_under_anomalous(root):
            continue
        count = 0
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            if _path_under_anomalous(dirpath):
                dirnames.clear()
                continue
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
            for name in filenames:
                if count >= max_per_root:
                    break
                filepath = os.path.join(dirpath, name)
                if not should_index_file(filepath):
                    continue
                try:
                    if os.path.getsize(filepath) > 200 * 1024:
                        continue
                except OSError:
                    continue
                count += 1
                yield filepath
