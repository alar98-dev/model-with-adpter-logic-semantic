#!/usr/bin/env python3
"""
Mapa objetivo do workspace SWA no WSL + snapshot de métricas operacionais.
Uso: python3 scripts/map_wsl_workspace.py [--json logs/workspace_map.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

try:
    from swa.workspace_policy import (
        get_watch_roots,
        describe_watch_policy,
        DEBIAN_GLOBAL_ZONES,
        AGENT_WISDOM_ROOT,
    )
except ImportError:
    get_watch_roots = lambda: ["/services/model-with-adpter-logic-semantic"]
    describe_watch_policy = lambda: "(workspace_policy indisponível)"
    DEBIAN_GLOBAL_ZONES = ("/services",)
    AGENT_WISDOM_ROOT = Path("/home/alifer/.cursor/projects/services-model-with-adpter-logic-semantic/agent-transcripts")

ROOT = Path("/services/model-with-adpter-logic-semantic")
SHM_DRIFT = Path("/dev/shm/brain_map_vhd")
ORIGINS_LOG = ROOT / "logs/structural_origins.jsonl"

# Árvore lógica (o que importa para o Aeon — não é scan recursivo cego)
WORKSPACE_ZONES = {
    "cortex": "Loop principal (run.py), órgãos, auditivo, obelisk",
    "swa": "Gemma, bridge pgvector, LoRA, orquestração, DB controllers",
    "sensors": "Watcher mmap 640D, subcortex 270M (serviço separado)",
    "mcp": "Templates systemd MCP",
    "mcp_gateway": "Servidor MCP SSE :8000",
    "scripts": "Utilitários (este mapa, setup pgvector, testes)",
    "prompts": "Persona master JSON → Postgres",
    "models": "Pesos Gemma (excluído do watcher)",
    "logs": "Telemetria estrutural, origens interdimensionais",
    "deployments": "swa-rest.service (template; geralmente inativo)",
    "actuators": "Homeostase CPU/RAM (system_motor)",
}

EXCLUDE_SCAN = {
    ".git", "__pycache__", "venv", ".venv", "models", "node_modules",
    "logs", "input_frames", ".kilo",
}
NOISE_FILES = {
    "swa_Aeon.log", "swa_PENSAMENTO_PURO.log", "server.log", "sonhador.log",
    "speaker_signatures.json", "aeon_sensory.log",
}


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"(erro: {e})"


def service_status() -> dict:
    units = [
        "swa-auto.service",
        "swa-subcortex.service",
        "swa-mcp.service",
        "swa-auditory.service",
        "swa-obelisk.service",
        "swa-rest.service",
        "postgresql@17-main.service",
    ]
    out = {}
    for u in units:
        out[u] = _run(["systemctl", "is-active", u])
    return out


def latest_drift() -> dict | None:
    if not SHM_DRIFT.is_dir():
        return None
    maps = sorted(SHM_DRIFT.glob("drift_map_*.json"), key=lambda p: p.stat().st_mtime)
    if not maps:
        return None
    latest = maps[-1]
    try:
        with open(latest, encoding="utf-8") as f:
            data = json.load(f)
        return {"file": str(latest), "metrics": data, "mtime": latest.stat().st_mtime}
    except Exception:
        return None


def count_tree(root: Path, max_files: int = 8000) -> dict:
    """Contagem leve por extensão (sem listar tudo)."""
    counts: dict[str, int] = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_SCAN]
        for name in filenames:
            if name in NOISE_FILES:
                continue
            total += 1
            ext = Path(name).suffix.lower() or "(sem ext)"
            counts[ext] = counts.get(ext, 0) + 1
            if total >= max_files:
                break
        if total >= max_files:
            break
    return {"files_scanned_cap": total, "by_extension": counts}


def priority_paths() -> list[str]:
    """Arquivos que devem ancorar o raciocínio (núcleo SWA)."""
    patterns = [
        "cortex/run.py",
        "swa/gemma_integration.py",
        "swa/quantum_bridge.py",
        "swa/orchestrator.py",
        "swa/broadcast_runtime.py",
        "prompts/swa_master_persona.json",
        "sensors/directory_watcher.py",
        "cortex/organs/brain_mapper.py",
    ]
    existing = []
    for rel in patterns:
        p = ROOT / rel
        if p.is_file():
            existing.append(str(p))
    return existing


def ram_snapshot() -> dict:
    try:
        import psutil
        m = psutil.virtual_memory()
        return {
            "total_gb": round(m.total / (1024**3), 2),
            "available_gb": round(m.available / (1024**3), 2),
            "percent_used": m.percent,
        }
    except Exception:
        return {}


def build_report() -> dict:
    drift = latest_drift()
    metrics = (drift or {}).get("metrics") or {}
    l0_13 = metrics.get("L0_to_L13")
    l13_24 = metrics.get("L13_to_L24")

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "objective": describe_watch_policy(),
        "debian_global_zones": list(DEBIAN_GLOBAL_ZONES),
        "agent_wisdom_root": str(AGENT_WISDOM_ROOT),
        "active_watch_roots": get_watch_roots(),
        "services": service_status(),
        "workspace_zones": WORKSPACE_ZONES,
        "priority_anchor_files": priority_paths(),
        "tree_scan": count_tree(ROOT),
        "ram": ram_snapshot(),
        "topological_drift": {
            "latest": drift,
            "interpretation": {
                "L0_to_L13": l0_13,
                "L13_to_L24": l13_24,
                "note": (
                    "L13→L24 costuma ficar >0.9 em todo forward pass (camadas vizinhas). "
                    "Recompensa só faz sentido se L0→L13 for menor (processamento profundo distinto da entrada)."
                ),
            },
        },
        "monitoring": {
            "aeon_log": str(ROOT / "swa_Aeon.log"),
            "structural_log": str(ROOT / "logs/structural_geometry.log"),
            "origins_log": str(ORIGINS_LOG),
            "commands": [
                "journalctl -u swa-auto.service -f",
                "python3 swa_monitor.py",
                "python3 scripts/map_wsl_workspace.py",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Mapa WSL + métricas SWA")
    parser.add_argument("--json", default="", help="Salvar JSON (ex: logs/workspace_map.json)")
    args = parser.parse_args()

    report = build_report()

    print("=" * 72)
    print("SWA — MAPA DO WORKSPACE (WSL)")
    print("=" * 72)
    print(f"Raiz: {report['root']}")
    print(f"\nObjetivo:\n  {report['objective']}\n")

    print("Serviços:")
    for unit, state in report["services"].items():
        print(f"  {unit:40} {state}")

    print("\nZonas lógicas:")
    for zone, desc in report["workspace_zones"].items():
        p = ROOT / zone
        mark = "✓" if p.exists() else "—"
        print(f"  [{mark}] {zone:16} {desc}")

    print("\nÂncoras prioritárias (núcleo):")
    for p in report["priority_anchor_files"]:
        print(f"  • {p}")

    td = report["topological_drift"]
    if td.get("latest"):
        m = td["latest"]["metrics"]
        print("\nDrift topológico (último SHM):")
        for k, v in sorted(m.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print("\nDrift topológico: (nenhum drift_map em /dev/shm/brain_map_vhd)")

    print(f"\nRAM: {report.get('ram')}")
    print(f"\nExtensões no repo (amostra): {report['tree_scan']['by_extension']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON salvo em: {out}")


if __name__ == "__main__":
    main()
