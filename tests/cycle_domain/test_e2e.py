"""
Testes end-to-end para o CycleOrchestrator.

Cobre: fluxo completo de ciclo, scheduling, prioridades, eventos.
"""
import pytest
import time
import threading
from swa.cycle_domain import (
    CycleOrchestrator,
    Cycle,
    CycleConfig,
    Phase,
    ResourceType,
    EventBus,
    ResourceManager,
    SharedCycleState,
    PhaseCompleted,
    HeartbeatEmitted,
)


# ═══════════════════════════════════════════════════════════
#  END-TO-END: FLUXO COMPLETO
# ═══════════════════════════════════════════════════════════

class TestEndToEndFlow:
    def test_single_cycle_execution(self, orchestrator, simple_cycle):
        orchestrator.register_cycle(simple_cycle)
        orchestrator.start()
        time.sleep(2)
        orchestrator.stop()

        assert simple_cycle.iterations >= 1

    def test_multiple_cycles_execution(self, orchestrator):
        cycles = []
        for i in range(3):
            def handler(ctx, idx=i):
                ctx.set_metric("idx", idx)

            config = CycleConfig(
                name=f"cycle_{i}",
                interval_seconds=0.5,
                resource_requirements=(),
                priority=i,
            )
            cycle = Cycle(config)
            cycle.add_phase(Phase(f"phase_{i}", handler))
            cycles.append(cycle)
            orchestrator.register_cycle(cycle)

        orchestrator.start()
        time.sleep(3)
        orchestrator.stop()

        for cycle in cycles:
            assert cycle.iterations >= 1

    def test_priority_ordering(self, orchestrator):
        execution_order = []

        def critical_handler(ctx):
            execution_order.append("critical")

        def low_handler(ctx):
            execution_order.append("low")

        critical_config = CycleConfig(
            name="critical", interval_seconds=0.1, resource_requirements=(), priority=0
        )
        low_config = CycleConfig(
            name="low", interval_seconds=0.1, resource_requirements=(), priority=3
        )

        critical_cycle = Cycle(critical_config)
        critical_cycle.add_phase(Phase("p", critical_handler))

        low_cycle = Cycle(low_config)
        low_cycle.add_phase(Phase("p", low_handler))

        orchestrator.register_cycle(low_cycle)
        orchestrator.register_cycle(critical_cycle)

        orchestrator.start()
        time.sleep(1)
        orchestrator.stop()

        assert execution_order[0] == "critical"


# ═══════════════════════════════════════════════════════════
#  END-TO-END: EVENTOS
# ═══════════════════════════════════════════════════════════

class TestEndToEndEvents:
    def test_events_emitted(self, orchestrator, simple_cycle):
        events = []

        def collect(event):
            events.append(type(event).__name__)

        orchestrator._event_bus.subscribe("PhaseCompleted", collect)
        orchestrator._event_bus.subscribe("HeartbeatEmitted", collect)

        orchestrator.register_cycle(simple_cycle)
        orchestrator.start()
        time.sleep(1)
        orchestrator.stop()

        assert "PhaseCompleted" in events
        assert "HeartbeatEmitted" in events

    def test_cycle_completed_event(self, orchestrator, simple_cycle):
        completed = []

        def on_completed(event):
            completed.append(event)

        orchestrator._event_bus.subscribe("CycleCompleted", on_completed)
        orchestrator.register_cycle(simple_cycle)
        orchestrator.start()
        time.sleep(2)
        orchestrator.stop()

        assert len(completed) >= 1
        assert completed[0].cycle_name == "test_cycle"


# ═══════════════════════════════════════════════════════════
#  END-TO-END: SHARED STATE
# ═══════════════════════════════════════════════════════════

class TestEndToEndSharedState:
    def test_state_updated(self, orchestrator, simple_cycle):
        orchestrator.register_cycle(simple_cycle)
        orchestrator.start()
        time.sleep(2)
        orchestrator.stop()

        state = orchestrator._shared_state.get_cycle("test_cycle")
        assert "iterations" in state
        assert state["iterations"] >= 1

    def test_state_persistence(self, tmp_path):
        state_file = tmp_path / "e2e_state.json"
        shared_state = SharedCycleState(str(state_file))

        def handler(ctx):
            ctx.set_metric("persisted", True)

        config = CycleConfig(
            name="persist_test", interval_seconds=0.5, resource_requirements=()
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("p", handler))

        orchestrator = CycleOrchestrator(shared_state=shared_state)
        orchestrator.register_cycle(cycle)
        orchestrator.start()
        time.sleep(1)
        orchestrator.stop()

        state = shared_state.get_cycle("persist_test")
        assert "iterations" in state


# ═══════════════════════════════════════════════════════════
#  END-TO-END: RESOURCE MANAGEMENT
# ═══════════════════════════════════════════════════════════

class TestEndToEndResources:
    def test_resource_acquisition_and_release(self, orchestrator):
        resources_used = []

        def handler(ctx):
            resources_used.append(list(ctx.resources.keys()))

        config = CycleConfig(
            name="resource_test",
            interval_seconds=1,
            resource_requirements=(ResourceType.REDIS,),
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("p", handler))

        orchestrator.register_cycle(cycle)
        orchestrator.start()
        time.sleep(2)
        orchestrator.stop()

        assert len(resources_used) >= 1
        assert ResourceType.REDIS in resources_used[0]
