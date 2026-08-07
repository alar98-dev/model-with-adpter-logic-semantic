"""
Testes de performance e estresse para o CycleDomain.

Cobre: throughput, latência, uso de memória, concorrência extrema.
"""
import pytest
import time
import threading
import os
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


# ═══════════════════════════════════════════════════════════
#  PERFORMANCE: THROUGHPUT
# ═══════════════════════════════════════════════════════════

class TestThroughput:
    def test_cycles_per_second(self, orchestrator):
        count = []

        def handler(ctx):
            count.append(1)

        config = CycleConfig(
            name="throughput", interval_seconds=0.01, resource_requirements=()
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("p", handler))

        orchestrator.register_cycle(cycle)
        start = time.time()
        orchestrator.start()
        time.sleep(2)
        orchestrator.stop()
        elapsed = time.time() - start

        cps = len(count) / elapsed
        assert cps > 50  # At least 50 cycles per second

    def test_event_bus_throughput(self, event_bus):
        count = []

        def handler(event):
            count.append(1)

        event_bus.subscribe("PhaseCompleted", handler)
        start = time.time()

        for i in range(1000):
            event_bus.emit(PhaseCompleted(
                cycle_id=str(i), cycle_name="t", phase="p", duration_ms=1, success=True
            ))

        time.sleep(2)
        elapsed = time.time() - start
        eps = len(count) / elapsed

        assert eps > 200  # At least 200 events per second


# ═══════════════════════════════════════════════════════════
#  PERFORMANCE: LATENCY
# ═══════════════════════════════════════════════════════════

class TestLatency:
    def test_cycle_execution_latency(self):
        latencies = []

        def handler(ctx):
            latencies.append(time.time())

        config = CycleConfig(
            name="latency", interval_seconds=0.01, resource_requirements=()
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("p", handler))

        for _ in range(100):
            cycle.run(None)

        assert len(latencies) == 100

    def test_resource_acquisition_latency(self, resource_manager):
        latencies = []

        for _ in range(100):
            start = time.time()
            tokens = resource_manager.acquire((ResourceType.REDIS,), "latency_test")
            latencies.append(time.time() - start)
            resource_manager.release(tokens, "latency_test")

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.01  # Less than 10ms average


# ═══════════════════════════════════════════════════════════
#  PERFORMANCE: MEMORY
# ═══════════════════════════════════════════════════════════

class TestMemory:
    def test_shared_state_memory_growth(self, shared_state):
        initial_size = os.path.getsize(shared_state._path) if shared_state._path.exists() else 0

        for i in range(1000):
            shared_state.update_cycle(f"cycle_{i}", {"data": "x" * 100})

        final_size = os.path.getsize(shared_state._path)
        growth = final_size - initial_size

        # Should be less than 1MB for 1000 cycles
        assert growth < 1_000_000

    def test_event_bus_memory(self, event_bus):
        initial_stats = event_bus.get_stats()

        for i in range(1000):
            event_bus.emit(PhaseCompleted(
                cycle_id=str(i), cycle_name="t", phase="p", duration_ms=1, success=True
            ))

        time.sleep(1)
        final_stats = event_bus.get_stats()
        assert final_stats["queue_size"] < 100


# ═══════════════════════════════════════════════════════════
#  STRESS: CONCURRENT OPERATIONS
# ═══════════════════════════════════════════════════════════

class TestStress:
    def test_concurrent_cycle_registration(self):
        orchestrator = CycleOrchestrator()
        errors = []

        def register_cycle(idx):
            try:
                config = CycleConfig(
                    name=f"stress_{idx}",
                    interval_seconds=1,
                    resource_requirements=(),
                )
                cycle = Cycle(config)
                cycle.add_phase(Phase("p", lambda ctx: None))
                orchestrator.register_cycle(cycle)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_cycle, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert len(orchestrator._cycles) == 100

    def test_concurrent_shared_state_updates(self, shared_state):
        errors = []

        def update(idx):
            try:
                for i in range(100):
                    shared_state.update_cycle(f"cycle_{idx}", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []

    def test_high_frequency_events(self, event_bus):
        received = []

        def handler(event):
            received.append(1)

        event_bus.subscribe("PhaseCompleted", handler)

        for i in range(5000):
            event_bus.emit(PhaseCompleted(
                cycle_id=str(i), cycle_name="t", phase="p", duration_ms=1, success=True
            ))

        time.sleep(3)
        assert len(received) >= 4000  # At least 80% delivered
