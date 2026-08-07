"""
Conftest para testes do CycleDomain.

Fixtures compartilhadas para todos os testes.
"""
import os
import sys
import time
import json
import tempfile
import threading
import pytest

ROOT = os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic")
sys.path.insert(0, ROOT)

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


@pytest.fixture
def event_bus():
    bus = EventBus()
    bus.start()
    yield bus
    bus.stop()


@pytest.fixture
def resource_manager():
    return ResourceManager()


@pytest.fixture
def shared_state(tmp_path):
    state_file = tmp_path / "test_state.json"
    return SharedCycleState(str(state_file))


@pytest.fixture
def orchestrator(event_bus, resource_manager, shared_state):
    return CycleOrchestrator(
        event_bus=event_bus,
        resource_manager=resource_manager,
        shared_state=shared_state,
    )


@pytest.fixture
def simple_cycle():
    def handler(context):
        context.set_metric("executed", True)

    config = CycleConfig(
        name="test_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=2,
        resource_requirements=(),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("test_phase", handler, timeout=5.0))
    return cycle


@pytest.fixture
def failing_cycle():
    def handler(context):
        raise ValueError("Test error")

    config = CycleConfig(
        name="failing_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=2,
        resource_requirements=(),
        max_errors=3,
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("failing_phase", handler, timeout=5.0))
    return cycle


@pytest.fixture
def slow_cycle():
    def handler(context):
        time.sleep(0.1)

    config = CycleConfig(
        name="slow_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=2,
        resource_requirements=(),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("slow_phase", handler, timeout=1.0))
    return cycle


@pytest.fixture
def timeout_cycle():
    def handler(context):
        time.sleep(10)  # Maior que o timeout

    config = CycleConfig(
        name="timeout_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=2,
        resource_requirements=(),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("timeout_phase", handler, timeout=0.1))
    return cycle


@pytest.fixture
def resource_heavy_cycle():
    def handler(context):
        context.set_metric("resource_used", True)

    config = CycleConfig(
        name="resource_heavy",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=2,
        resource_requirements=(ResourceType.TORCH, ResourceType.REDIS),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("resource_phase", handler, timeout=5.0))
    return cycle


@pytest.fixture
def priority_critical_cycle():
    def handler(context):
        context.set_metric("critical", True)

    config = CycleConfig(
        name="critical_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=0,  # Crítico
        resource_requirements=(),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("critical_phase", handler, timeout=5.0))
    return cycle


@pytest.fixture
def priority_low_cycle():
    def handler(context):
        context.set_metric("low", True)

    config = CycleConfig(
        name="low_cycle",
        interval_seconds=1,
        max_duration_seconds=5,
        priority=3,  # Baixa
        resource_requirements=(),
    )
    cycle = Cycle(config)
    cycle.add_phase(Phase("low_phase", handler, timeout=5.0))
    return cycle
