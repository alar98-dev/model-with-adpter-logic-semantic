#!/usr/bin/env python3
"""
SWA Unified Cycle V2 — DDD Controller para ciclos eventuais.

Substitui: rem_dream.py, subcortex.py, wiki_crawler.py como processos separados.
Usa a arquitetura DDD com CycleOrchestrator, EventBus e ResourceManager.

Uso:
    python3 scripts/unified_cycle_v2.py              # modo normal
    python3 scripts/unified_cycle_v2.py --dry-run    # simula sem rodar
    python3 scripts/unified_cycle_v2.py --profile    # mostra perfil de consumo
    python3 scripts/unified_cycle_v2.py --status     # mostra estado atual
    python3 scripts/unified_cycle_v2.py --stats      # mostra estatísticas
"""
from __future__ import annotations

import os
import sys
import time
import signal
import json
import logging
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
sys.path.insert(0, str(ROOT))

from swa.cycle_domain import (
    CycleOrchestrator,
    Cycle,
    CycleConfig,
    Phase,
    ResourceType,
    EventBus,
    ResourceManager,
    SharedCycleState,
)
from swa.cycle_domain.adapters import (
    DreamCycleAdapter,
    CortexCycleAdapter,
    HealthCycleAdapter,
    WikiCycleAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("swa.cycle.v2")


def create_orchestrator(dry_run: bool = False) -> CycleOrchestrator:
    """Factory principal - cria o orquestrador com todos os ciclos."""
    event_bus = EventBus()
    resource_manager = ResourceManager()
    shared_state = SharedCycleState()

    orchestrator = CycleOrchestrator(
        event_bus=event_bus,
        resource_manager=resource_manager,
        shared_state=shared_state,
    )

    if not dry_run:
        dream_adapter = DreamCycleAdapter()
        cortex_adapter = CortexCycleAdapter()
        health_adapter = HealthCycleAdapter()
        wiki_adapter = WikiCycleAdapter()

        orchestrator.register_cycle(dream_adapter.create_cycle())
        orchestrator.register_cycle(cortex_adapter.create_cycle())
        orchestrator.register_cycle(health_adapter.create_cycle())
        orchestrator.register_cycle(wiki_adapter.create_cycle())

    return orchestrator


def signal_handler(sig, frame):
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SWA Unified Cycle V2 — DDD Controller")
    parser.add_argument("--dry-run", action="store_true", help="Simula sem rodar")
    parser.add_argument("--profile", action="store_true", help="Mostra perfil de consumo")
    parser.add_argument("--status", action="store_true", help="Mostra estado atual")
    parser.add_argument("--stats", action="store_true", help="Mostra estatísticas")
    args = parser.parse_args()

    if args.profile:
        print("SWA Unified Cycle V2 — DDD Controller")
        print("=" * 60)
        print("  DREAM:  256D adapter, 50 epochs max, 60s/ciclo")
        print("  CORTEX: Redis ping + UDP telemetry, 30s/ciclo")
        print("  HEALTH: Redis + PG + recursos, 60s/ciclo")
        print("  WIKI:   1 artigo Wikipedia, 5min/ciclo")
        print("=" * 60)
        print("  RAM estimada: ~200MB (1 modelo compartilhado)")
        print("  CPU estimada: ~5-15% (throttled)")
        print("  vs atual: 3 processos separados, ~9.5GB RAM, ~578% CPU")
        print("=" * 60)
        print("  Arquitetura: DDD CycleDomain")
        print("  Componentes: EventBus + ResourceManager + SharedState")
        return

    if args.status:
        shared_state = SharedCycleState()
        state = shared_state.get_all()
        if state:
            print("SWA Cycle State (SHM):")
            print(json.dumps(state, indent=2, default=str))
        else:
            print("Nenhum estado encontrado")
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    orchestrator = create_orchestrator(dry_run=args.dry_run)
    orchestrator.start()

    print("🔄 SWA Unified Cycle V2 — DDD Controller")
    print(f"   Ciclos registrados: {len(orchestrator._cycles)}")
    print("   Ctrl+C para parar\n")

    try:
        while True:
            if args.stats:
                stats = orchestrator.get_stats()
                print(json.dumps(stats, indent=2, default=str))
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
        print("\n✅ Unified cycle V2 encerrado")


if __name__ == "__main__":
    main()
