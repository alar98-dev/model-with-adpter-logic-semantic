"""
Adapters para ciclos existentes do SWA.

Cada adapter encapsula um ciclo legado e o adapta à interface do CycleDomain.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import threading
import urllib.request
from typing import Any

ROOT = os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic")
sys.path.insert(0, ROOT)

from swa.cycle_domain.models import (
    Cycle,
    CycleConfig,
    Phase,
    ResourceType,
)

logger = logging.getLogger("swa.cycle.adapters")


# ═══════════════════════════════════════════════════════════
#  ADAPTER: DREAM CYCLE (rem_dream.py legado)
# ═══════════════════════════════════════════════════════════

class DreamCycleAdapter:
    """Adapter para o ciclo REM de treino adaptativo."""

    def __init__(self):
        self.config = CycleConfig(
            name="dream",
            interval_seconds=600,
            max_duration_seconds=60,
            max_iterations=50,
            priority=2,
            resource_requirements=(ResourceType.TORCH, ResourceType.PG),
            auto_restart=True,
            max_errors=3,
        )
        self._adapter = None
        self._kernel = None
        self._device = None

    def _setup(self):
        import torch
        from swa.adapter import SemanticWarpAdapter

        self._device = torch.device("cpu")
        self._adapter = SemanticWarpAdapter(d=256).to(self._device)

    def execute(self, context: Any):
        import torch
        import torch.optim as optim
        import psycopg2

        if self._adapter is None:
            self._setup()

        t0 = time.time()
        opt = optim.Adam(self._adapter.parameters(), lr=1e-4)

        conn = psycopg2.connect("dbname=swa_matrix user=alifer connect_timeout=5")
        cur = conn.cursor()
        cur.execute("SELECT raw_content FROM blake_semantic_anchors LIMIT 50")
        samples = [row[0] for row in cur.fetchall() if row[0]]
        conn.close()

        if not samples:
            return

        epochs = 0
        deadline = t0 + 55

        while time.time() < deadline and epochs < 50:
            for sample in samples[:10]:
                try:
                    text_tensor = torch.randn(1, 256, device=self._device)
                    target = torch.randn(1, 256, device=self._device)
                    output = self._adapter(text_tensor)
                    loss = torch.nn.functional.mse_loss(output, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                except Exception:
                    continue
            epochs += 1

        context.set_metric("dream_epochs", epochs)
        context.set_metric("dream_duration_s", time.time() - t0)

    def create_cycle(self) -> Cycle:
        cycle = Cycle(self.config)
        cycle.add_phase(Phase("dream_train", self.execute, timeout=60.0))
        return cycle


# ═══════════════════════════════════════════════════════════
#  ADAPTER: CORTEX CYCLE (subcortex.py legado)
# ═══════════════════════════════════════════════════════════

class CortexCycleAdapter:
    """Adapter para o ciclo de inferência do cortex."""

    def __init__(self):
        self.config = CycleConfig(
            name="cortex",
            interval_seconds=30,
            max_duration_seconds=5,
            max_iterations=10000,
            priority=1,
            resource_requirements=(ResourceType.REDIS,),
            auto_restart=True,
            max_errors=5,
        )

    def execute(self, context: Any):
        import redis
        import socket

        r = redis.Redis(
            host="127.0.0.1", port=6379, decode_responses=True, socket_connect_timeout=2
        )
        grid_status = r.get("grid_status") or "{}"

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = json.dumps({
            "type": "telemetry",
            "source": "cortex_cycle",
            "timestamp": time.time(),
        }).encode()
        s.sendto(payload, ("127.0.0.1", 20005))
        s.close()

        context.set_metric("grid_status_len", len(grid_status))

    def create_cycle(self) -> Cycle:
        cycle = Cycle(self.config)
        cycle.add_phase(Phase("cortex_infer", self.execute, timeout=5.0))
        return cycle


# ═══════════════════════════════════════════════════════════
#  ADAPTER: HEALTH CYCLE (auto-teste estrutural)
# ═══════════════════════════════════════════════════════════

class HealthCycleAdapter:
    """Adapter para health checks do sistema."""

    def __init__(self):
        self.config = CycleConfig(
            name="health",
            interval_seconds=60,
            max_duration_seconds=2,
            max_iterations=100000,
            priority=0,  # Crítico
            resource_requirements=(ResourceType.REDIS, ResourceType.PG),
            auto_restart=True,
            max_errors=10,
        )

    def execute(self, context: Any):
        import psutil
        import redis
        import psycopg2

        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        if cpu > 90 or ram.percent > 85:
            raise ResourceWarning(f"CPU {cpu}%, RAM {ram.percent}%")

        r = redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=2)
        if not r.ping():
            raise ServiceUnavailable("Redis")

        conn = psycopg2.connect("dbname=swa_matrix user=alifer connect_timeout=3")
        conn.close()

        context.set_metric("cpu_percent", cpu)
        context.set_metric("ram_percent", ram.percent)
        context.set_metric("ram_mb", ram.used / (1024 * 1024))

    def create_cycle(self) -> Cycle:
        cycle = Cycle(self.config)
        cycle.add_phase(Phase("health_check", self.execute, timeout=2.0))
        return cycle


class ResourceWarning(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


# ═══════════════════════════════════════════════════════════
#  ADAPTER: WIKI CYCLE (wiki_crawler.py legado)
# ═══════════════════════════════════════════════════════════

class WikiCycleAdapter:
    """Adapter para o ciclo de crawling da Wikipedia."""

    def __init__(self):
        self.config = CycleConfig(
            name="wiki",
            interval_seconds=300,
            max_duration_seconds=15,
            max_iterations=10000,
            priority=3,  # Baixa prioridade
            resource_requirements=(),
            auto_restart=True,
            max_errors=5,
        )

    def execute(self, context: Any):
        url = "https://pt.wikipedia.org/api/rest_v1/page/random/summary"
        req = urllib.request.Request(url, headers={"User-Agent": "AeonCycle/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        title = data.get("title", "")
        extract = data.get("extract", "")[:500]

        context.set_metric("wiki_title", title)
        context.set_metric("wiki_chars", len(extract))

    def create_cycle(self) -> Cycle:
        cycle = Cycle(self.config)
        cycle.add_phase(Phase("wiki_fetch", self.execute, timeout=15.0))
        return cycle


# ═══════════════════════════════════════════════════════════
#  ADAPTER: OODA CYCLE (orchestrator.py legado)
# ═══════════════════════════════════════════════════════════

class OODACycleAdapter:
    """Adapter para o ciclo OODA principal."""

    def __init__(self):
        self.config = CycleConfig(
            name="ooda",
            interval_seconds=1.5,
            max_duration_seconds=10,
            max_iterations=100000,
            priority=0,  # Crítico
            resource_requirements=(ResourceType.REDIS, ResourceType.PG),
            auto_restart=True,
            max_errors=3,
        )
        self._orchestrator = None

    def _get_orchestrator(self):
        try:
            from swa.orchestrator import AutonomousOrchestrator
            return AutonomousOrchestrator()
        except ImportError:
            return None

    def execute(self, context: Any):
        if self._orchestrator is None:
            self._orchestrator = self._get_orchestrator()

        if self._orchestrator:
            self._orchestrator._run_ooda_cycle()
        else:
            logger.warning("OODA orchestrator not available")

    def create_cycle(self) -> Cycle:
        cycle = Cycle(self.config)
        cycle.add_phase(Phase("ooda_run", self.execute, timeout=10.0))
        return cycle
