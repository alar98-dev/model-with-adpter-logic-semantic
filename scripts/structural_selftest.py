#!/usr/bin/env python3
"""
SWA Structural Self-Test — ciclo ativo de auto-teste estrutural.
Valida integridade de todos os subsistemas críticos antes do boot.

Uso:
  python3 scripts/structural_selftest.py           # teste completo
  python3 scripts/structural_selftest.py --quick    # só infra
  python3 scripts/structural_selftest.py --watch    # loop contínuo (watchdog)
  python3 scripts/structural_selftest.py --json     # output JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import socket
import subprocess
import importlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
sys.path.insert(0, str(ROOT))


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class TestResult:
    name: str
    status: Status
    latency_ms: float = 0.0
    detail: str = ""
    critical: bool = False


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return all(r.status != Status.FAIL for r in self.results if r.critical)

    @property
    def summary(self) -> str:
        counts = {}
        for r in self.results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return " ".join(f"{v}{'↑' if k == 'ok' else '⚠' if k == 'warn' else '✗' if k == 'fail' else '○'}" for k, v in counts.items())

    def add(self, result: TestResult):
        self.results.append(result)


def _t(fn, *a, **kw):
    t0 = time.time()
    try:
        val = fn(*a, **kw)
        ms = (time.time() - t0) * 1000
        return val, ms
    except Exception as e:
        ms = (time.time() - t0) * 1000
        return e, ms


# ═══════════════════════════════════════════════════════════
#  TESTES DE INFRAESTRUTURA
# ═══════════════════════════════════════════════════════════

def test_redis(suite: TestSuite):
    """Redis: ping + pub/sub funcional."""
    try:
        import redis
        r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True, socket_connect_timeout=3)
        _, ms = _t(r.ping)
        assert _t(r.ping)[0] is True
        suite.add(TestResult("redis", Status.OK, ms, "ping OK", critical=True))
    except Exception as e:
        suite.add(TestResult("redis", Status.FAIL, detail=f"{type(e).__name__}: {e}", critical=True))


def test_postgres(suite: TestSuite):
    """PostgreSQL: conexão + tabela blake_semantic_anchors."""
    try:
        import psycopg2
        t0 = time.time()
        conn = psycopg2.connect("dbname=swa_matrix user=alifer connect_timeout=5")
        ms = (time.time() - t0) * 1000
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM blake_semantic_anchors")
        count = cur.fetchone()[0]
        conn.close()
        suite.add(TestResult("postgres", Status.OK, ms, f"conectado, {count} âncoras", critical=True))
    except Exception as e:
        suite.add(TestResult("postgres", Status.FAIL, detail=str(e), critical=True))


def test_shm_manifold(suite: TestSuite):
    """SHM Manifold: manifest.json válido."""
    manifest_path = Path("/dev/shm/swa_vector_manifold/manifest.json")
    if not manifest_path.exists():
        suite.add(TestResult("shm_manifold", Status.WARN, detail="manifest não existe"))
        return
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        nodes = data.get("node_count", 0)
        if nodes < 10:
            suite.add(TestResult("shm_manifold", Status.WARN, detail=f"só {nodes} nós"))
        else:
            suite.add(TestResult("shm_manifold", Status.OK, detail=f"{nodes} nós indexados"))
    except Exception as e:
        suite.add(TestResult("shm_manifold", Status.FAIL, detail=str(e), critical=True))


def test_shm_llm_context(suite: TestSuite):
    """SHM LLM Context: contexto ativo do TokenGuardian."""
    ctx_path = Path("/dev/shm/aeon_llm_context.json")
    if not ctx_path.exists():
        suite.add(TestResult("shm_llm_context", Status.WARN, detail="arquivo não existe"))
        return
    try:
        stat = os.stat(ctx_path)
        age_min = (time.time() - stat.st_mtime) / 60
        if age_min > 30:
            suite.add(TestResult("shm_llm_context", Status.WARN, detail=f"última atualização há {age_min:.0f}min"))
        else:
            suite.add(TestResult("shm_llm_context", Status.OK, detail=f"ativo, atualizado há {age_min:.1f}min"))
    except Exception as e:
        suite.add(TestResult("shm_llm_context", Status.WARN, detail=str(e)))


def test_aeon_socket(suite: TestSuite):
    """Unix Socket: /var/run/aeon/synapse.sock."""
    sock_path = Path("/var/run/aeon/synapse.sock")
    if not sock_path.exists():
        suite.add(TestResult("aeon_socket", Status.WARN, detail="socket não existe"))
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        _, ms = _t(s.connect, str(sock_path))
        s.close()
        suite.add(TestResult("aeon_socket", Status.OK, ms, "conectável"))
    except Exception as e:
        suite.add(TestResult("aeon_socket", Status.WARN, detail=str(e)))


def test_vhdx_mount(suite: TestSuite):
    """SWA_ROOT.vhdx: loop device ativo."""
    try:
        result = subprocess.run(["losetup", "-a"], capture_output=True, text=True, timeout=5)
        if "SWA_ROOT" in result.stdout or "/dev/loop" in result.stdout:
            suite.add(TestResult("vhdx_mount", Status.OK, detail="loop device ativo"))
        else:
            suite.add(TestResult("vhdx_mount", Status.WARN, detail="nenhum loop detectado"))
    except Exception as e:
        suite.add(TestResult("vhdx_mount", Status.WARN, detail=str(e)))


# ═══════════════════════════════════════════════════════════
#  TESTES DE MÓDULOS PYTHON
# ═══════════════════════════════════════════════════════════

SWA_MODULES = [
    ("swa.ipc_bus", "pack_aeon_frame"),
    ("swa.shm_guard", "ShmGuard"),
    ("swa.grid_registry", "GridRegistry"),
    ("swa.aeon_logger", "append_aeon"),
]

SWA_HEAVY_MODULES = [
    ("swa.semantic_bus", "SemanticBus"),
    ("swa.autonomy_pulse", "AutonomyPulse"),
    ("swa.geosense", "Geosense"),
    ("swa.token_guardian", "TokenGuardian"),
    ("swa.vector_manifold", "VectorManifold"),
    ("swa.reality_embedder", "get_reality_embedder"),
]

def _test_module_list(modules, timeout_per=5):
    failed = []
    ok_count = 0
    for mod_name, attr in modules:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {mod_name}" + (f"; getattr(__import__('{mod_name}', fromlist=['{attr}']), '{attr}')" if attr else "")],
                capture_output=True, text=True, timeout=timeout_per
            )
            if proc.returncode == 0:
                ok_count += 1
            else:
                err = proc.stderr.strip().splitlines()[-1] if proc.stderr else "unknown"
                failed.append(f"{mod_name}: {err[:60]}")
        except subprocess.TimeoutExpired:
            failed.append(f"{mod_name}: timeout")
        except Exception as e:
            failed.append(f"{mod_name}: {e}")
    return ok_count, failed

def test_swa_modules(suite: TestSuite):
    """Módulos SWA leves: importabilidade (timeout 5s)."""
    ok_count, failed = _test_module_list(SWA_MODULES, timeout_per=5)
    if failed:
        suite.add(TestResult("swa_modules", Status.WARN,
                             detail=f"{ok_count}/{len(SWA_MODULES)} OK, falhas: {'; '.join(failed[:3])}"))
    else:
        suite.add(TestResult("swa_modules", Status.OK,
                             detail=f"{ok_count}/{len(SWA_MODULES)} módulos leves OK"))

def test_swa_heavy_modules(suite: TestSuite):
    """Módulos SWA pesados: importabilidade (timeout 15s, podem puxar torch)."""
    ok_count, failed = _test_module_list(SWA_HEAVY_MODULES, timeout_per=15)
    if failed:
        suite.add(TestResult("swa_heavy_modules", Status.WARN,
                             detail=f"{ok_count}/{len(SWA_HEAVY_MODULES)} OK, falhas: {'; '.join(failed[:3])}"))
    else:
        suite.add(TestResult("swa_heavy_modules", Status.OK,
                             detail=f"{ok_count}/{len(SWA_HEAVY_MODULES)} módulos pesados OK"))


# ═══════════════════════════════════════════════════════════
#  TESTES DE SERVIÇOS EXTERNOS
# ═══════════════════════════════════════════════════════════

def test_systemd_services(suite: TestSuite):
    """Serviços SWA: quais estão ativos."""
    critical_svcs = ["redis-server", "postgresql@17-main"]
    swa_svcs = ["swa-subcortex", "swa-mcp", "swa-obelisk", "swa-manifold", "swa-wiki", "swa-tts", "swa-dream"]
    all_svcs = critical_svcs + swa_svcs
    dead = []
    active = []
    for svc in all_svcs:
        try:
            r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True, timeout=3)
            if r.stdout.strip() == "active":
                active.append(svc)
            else:
                dead.append(svc)
        except:
            dead.append(svc)
    if any(s in dead for s in critical_svcs):
        suite.add(TestResult("systemd_services", Status.FAIL,
                             detail=f"MORTOS: {', '.join(dead)}", critical=True))
    elif dead:
        suite.add(TestResult("systemd_services", Status.WARN,
                             detail=f"{len(active)} ativos, {len(dead)} inativos: {', '.join(dead)}"))
    else:
        suite.add(TestResult("systemd_services", Status.OK,
                             detail=f"{len(active)} serviços ativos"))


# ═══════════════════════════════════════════════════════════
#  TESTES DE REDE/PORTAS
# ═══════════════════════════════════════════════════════════

def test_listening_ports(suite: TestSuite):
    """Portas TCP: verificar serviços essenciais."""
    expected = {
        6379: "redis",
        5432: "postgres",
        8000: "mcp-gateway",
        9300: "tts-server",
    }
    try:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        listening = set()
        for line in result.stdout.splitlines():
            for port in expected:
                if f":{port}" in line:
                    listening.add(port)
        missing = set(expected.keys()) - listening
        if missing:
            names = [expected[p] for p in missing]
            suite.add(TestResult("listening_ports", Status.WARN,
                                 detail=f"portas ausentes: {', '.join(names)}"))
        else:
            suite.add(TestResult("listening_ports", Status.OK,
                                 detail=f"{len(listening)}/{len(expected)} portas ativas"))
    except Exception as e:
        suite.add(TestResult("listening_ports", Status.WARN, detail=str(e)))


# ═══════════════════════════════════════════════════════════
#  TESTES DE INTEGRIDADE DE CÓDIGO
# ═══════════════════════════════════════════════════════════

def test_syntax_integrity(suite: TestSuite):
    """Integridade: verificar syntax dos módulos críticos."""
    critical_files = [
        "cortex/run.py",
        "swa/semantic_bus.py",
        "swa/autonomy_pulse.py",
        "swa/token_guardian.py",
        "swa/vector_manifold.py",
    ]
    failed = []
    for f in critical_files:
        path = ROOT / f
        if not path.exists():
            failed.append(f"{f}: não existe")
            continue
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import py_compile; py_compile.compile('{path}', doraise=True)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                failed.append(f"{f}: {result.stderr.strip()[:80]}")
        except Exception as e:
            failed.append(f"{f}: {e}")
    if failed:
        suite.add(TestResult("syntax_integrity", Status.FAIL,
                             detail="; ".join(failed), critical=True))
    else:
        suite.add(TestResult("syntax_integrity", Status.OK,
                             detail=f"{len(critical_files)} arquivos OK"))


def test_hardcoded_ips(suite: TestSuite):
    """IPs hardcoded: verificar ausência de 172.22."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "172.22", str(ROOT / "swa"), "--include=*.py"],
            capture_output=True, text=True, timeout=10
        )
        hits = [l for l in result.stdout.strip().splitlines() if l and "WSL_REDIS_HOST" not in l]
        if hits:
            suite.add(TestResult("hardcoded_ips", Status.WARN,
                                 detail=f"{len(hits)} IPs hardcoded restantes"))
        else:
            suite.add(TestResult("hardcoded_ips", Status.OK, detail="limpo"))
    except Exception as e:
        suite.add(TestResult("hardcoded_ips", Status.WARN, detail=str(e)))


# ═══════════════════════════════════════════════════════════
#  TESTES DE RECURSOS
# ═══════════════════════════════════════════════════════════

def test_system_resources(suite: TestSuite):
    """Recursos: RAM + CPU disponíveis."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)
        load1, _, _ = psutil.getloadavg()
        issues = []
        if ram.percent > 85:
            issues.append(f"RAM {ram.percent}%")
        if cpu > 90:
            issues.append(f"CPU {cpu}%")
        if load1 > 10:
            issues.append(f"load {load1:.1f}")
        if issues:
            suite.add(TestResult("system_resources", Status.WARN, detail=", ".join(issues)))
        else:
            free_gb = ram.available / (1024**3)
            suite.add(TestResult("system_resources", Status.OK,
                                 detail=f"RAM {ram.percent}% ({free_gb:.1f}GB livre), CPU {cpu}%, load {load1:.1f}"))
    except Exception as e:
        suite.add(TestResult("system_resources", Status.WARN, detail=str(e)))


# ═══════════════════════════════════════════════════════════
#  ORQUESTRADOR
# ═══════════════════════════════════════════════════════════

ALL_TESTS = [
    # Infra (crítico)
    test_redis,
    test_postgres,
    test_vhdx_mount,
    # SHM
    test_shm_manifold,
    test_shm_llm_context,
    test_aeon_socket,
    # Serviços
    test_systemd_services,
    test_listening_ports,
    # Código
    test_syntax_integrity,
    test_hardcoded_ips,
    test_swa_modules,
    test_swa_heavy_modules,
    # Recursos
    test_system_resources,
]

QUICK_TESTS = [
    test_redis,
    test_postgres,
    test_systemd_services,
    test_system_resources,
]


def run_suite(tests=None, json_output=False) -> TestSuite:
    suite = TestSuite(started_at=time.time())
    for test_fn in (tests or ALL_TESTS):
        test_fn(suite)
    suite.finished_at = time.time()

    if json_output:
        print(json.dumps({
            "ok": suite.ok,
            "latency_ms": round((suite.finished_at - suite.started_at) * 1000),
            "tests": [asdict(r) for r in suite.results],
        }, indent=2))
    else:
        total_ms = (suite.finished_at - suite.started_at) * 1000
        icon = "✅" if suite.ok else "❌"
        print(f"\n{icon} SWA SELF-TEST {suite.summary} ({total_ms:.0f}ms)")
        print("=" * 60)
        for r in suite.results:
            s_icon = {"ok": "✓", "warn": "⚠", "fail": "✗", "skip": "○"}[r.status.value]
            crit = " [CRÍTICO]" if r.critical else ""
            lat = f" ({r.latency_ms:.0f}ms)" if r.latency_ms > 0 else ""
            print(f"  {s_icon} {r.name}{crit}{lat}: {r.detail}")
        print("=" * 60)

    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWA Structural Self-Test")
    parser.add_argument("--quick", action="store_true", help="Testes rápidos (infra only)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--watch", action="store_true", help="Loop contínuo (30s)")
    parser.add_argument("--interval", type=int, default=30, help="Intervalo do watch (segundos)")
    args = parser.parse_args()

    if args.watch:
        print("🔄 SWA Self-Test Watch — Ctrl+C para parar")
        while True:
            tests = QUICK_TESTS if args.quick else ALL_TESTS
            suite = run_suite(tests, args.json)
            if not suite.ok:
                print("🚨 SELF-TEST FALHOU — intervenção necessária")
            time.sleep(args.interval)
    else:
        tests = QUICK_TESTS if args.quick else ALL_TESTS
        suite = run_suite(tests, args.json)
        sys.exit(0 if suite.ok else 1)
