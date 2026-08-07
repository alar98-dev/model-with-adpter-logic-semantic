"""
SWA Cycle Domain — Modelos de domínio para ciclos eventuais.

Uso:
    from swa.cycle_domain import CycleOrchestrator, Cycle, CycleConfig, Phase

    # Criar orquestrador
    orchestrator = CycleOrchestrator()

    # Criar ciclo
    config = CycleConfig(
        name="dream",
        interval_seconds=600,
        max_duration_seconds=60,
        priority=2,
        resource_requirements=(ResourceType.TORCH, ResourceType.PG),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("dream", dream_handler))

    # Registrar e iniciar
    orchestrator.register_cycle(cycle)
    orchestrator.start()
"""
from .models import (
    CycleState,
    PhaseState,
    ResourceType,
    CycleConfig,
    PhaseResult,
    Heartbeat,
    ResourceToken,
    CycleStarted,
    CycleCompleted,
    CyclePaused,
    CycleError,
    PhaseCompleted,
    ResourceAcquired,
    ResourceReleased,
    HeartbeatEmitted,
    SystemWarning,
    Phase,
    Cycle,
)
from .event_bus import EventBus, EventFilter, Subscription
from .resource_manager import ResourceManager, ResourcePool
from .shared_state import SharedCycleState
from .orchestrator import CycleOrchestrator, CycleContext

__all__ = [
    "CycleState",
    "PhaseState",
    "ResourceType",
    "CycleConfig",
    "PhaseResult",
    "Heartbeat",
    "ResourceToken",
    "CycleStarted",
    "CycleCompleted",
    "CyclePaused",
    "CycleError",
    "PhaseCompleted",
    "ResourceAcquired",
    "ResourceReleased",
    "HeartbeatEmitted",
    "SystemWarning",
    "Phase",
    "Cycle",
    "EventBus",
    "EventFilter",
    "Subscription",
    "ResourceManager",
    "ResourcePool",
    "SharedCycleState",
    "CycleOrchestrator",
    "CycleContext",
]
