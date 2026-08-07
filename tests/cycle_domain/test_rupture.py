"""
Testes de ruptura (breakpoint/boundary) para o CycleDomain.

Cobre: resource exhaustion, deadlock detection, timeout cascades,
error propagation, state corruption, concurrent mutations.
"""
import pytest
import time
import threading
import os
from swa.cycle_domain import (
    CycleOrchestrator,
    Cycle,
    CycleConfig,
    CycleState,
    Phase,
    ResourceType,
    EventBus,
    ResourceManager,
    SharedCycleState,
)


# ═══════════════════════════════════════════════════════════
#  RUPTURA: RESOURCE EXHAUSTION
# ═══════════════════════════════════════════════════════════

class TestResourceExhaustion:
    def test_all_resources_consumed(self, resource_manager):
        tokens = resource_manager.acquire((ResourceType.TORCH,), "c1")
        assert ResourceType.TORCH not in resource_manager.available()
        resource_manager.release(tokens, "c1")
        assert ResourceType.TORCH in resource_manager.available()

    def test_acquire_nonexistent_resource(self, resource_manager):
        tokens = resource_manager.acquire((ResourceType.TORCH,), "preempt")
        with pytest.raises(TimeoutError):
            resource_manager.acquire(
                (ResourceType.TORCH, ResourceType.REDIS),
                "c1",
                timeout=0.01,
            )
        resource_manager.release(tokens, "preempt")

    def test_release_already_released(self, resource_manager):
        tokens = resource_manager.acquire((ResourceType.REDIS,), "c1")
        resource_manager.release(tokens, "c1")
        resource_manager.release(tokens, "c1")  # Double release

    def test_acquire_under_contention(self, resource_manager):
        results = []
        errors = []

        def acquire_and_hold():
            try:
                tokens = resource_manager.acquire(
                    (ResourceType.TORCH,), "contention", timeout=2.0
                )
                results.append(tokens)
                time.sleep(0.1)
                resource_manager.release(tokens, "contention")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_and_hold) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 10
        assert errors == []


# ═══════════════════════════════════════════════════════════
#  RUPTURA: TIMEOUT CASCADES
# ═══════════════════════════════════════════════════════════

class TestTimeoutCascades:
    def test_phase_timeout_doesnt_kill_cycle(self, timeout_cycle):
        result = timeout_cycle.run(None)
        assert result.success is False
        assert timeout_cycle.state.value == "error"

    def test_cycle_recovers_after_timeout(self, timeout_cycle):
        timeout_cycle.run(None)
        timeout_cycle.consecutive_errors = 0
        timeout_cycle.state = CycleState.IDLE
        result = timeout_cycle.run(None)
        assert result.success is False  # Still times out

    def test_timeout_in_one_phase_stops_execution(self):
        def fast_phase(ctx):
            pass

        def slow_phase(ctx):
            time.sleep(10)

        config = CycleConfig(
            name="multi_phase",
            interval_seconds=1,
            resource_requirements=(),
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("fast", fast_phase, timeout=1.0))
        cycle.add_phase(Phase("slow", slow_phase, timeout=0.1))

        result = cycle.run(None)
        assert result.success is False
        assert result.phase_name == "slow"


# ═══════════════════════════════════════════════════════════
#  RUPTURA: ERROR PROPAGATION
# ═══════════════════════════════════════════════════════════

class TestErrorPropagation:
    def test_error_in_handler_stops_cycle(self):
        def error_handler(ctx):
            raise RuntimeError("boom")

        config = CycleConfig(name="err", interval_seconds=1, resource_requirements=())
        cycle = Cycle(config)
        cycle.add_phase(Phase("err", error_handler))

        result = cycle.run(None)
        assert result.success is False
        assert "boom" in result.error

    def test_error_count_increments(self, failing_cycle):
        failing_cycle.run(None)
        assert failing_cycle.errors == 1
        failing_cycle.run(None)
        assert failing_cycle.errors == 2

    def test_max_errors_stops_cycle(self, failing_cycle):
        for _ in range(3):
            failing_cycle.run(None)
        assert not failing_cycle.can_run(set())

    def test_error_resets_on_success(self):
        success_count = []

        def handler(ctx):
            success_count.append(True)

        config = CycleConfig(
            name="mixed", interval_seconds=1, resource_requirements=()
        )
        cycle = Cycle(config)
        cycle.add_phase(Phase("p", handler))

        cycle.state = CycleState.ERROR
        cycle.consecutive_errors = 3

        cycle.run(None)
        assert cycle.consecutive_errors == 0


# ═══════════════════════════════════════════════════════════
#  RUPTURA: STATE CORRUPTION
# ═══════════════════════════════════════════════════════════

class TestStateCorruption:
    def test_concurrent_state_mutations(self):
        config = CycleConfig(
            name="corrupt_test", interval_seconds=1, resource_requirements=()
        )
        cycle = Cycle(config)
        mutations = []

        def mutate():
            for _ in range(100):
                cycle.iterations += 1
                mutations.append(cycle.iterations)

        threads = [threading.Thread(target=mutate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert cycle.iterations == 1000

    def test_shared_state_concurrent_writes(self, shared_state):
        errors = []

        def write_data(cycle_name):
            try:
                for i in range(50):
                    shared_state.update_cycle(cycle_name, {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_data, args=(f"cycle_{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []


# ═══════════════════════════════════════════════════════════
#  RUPTURA: DEADLOCK DETECTION
# ═══════════════════════════════════════════════════════════

class TestDeadlockDetection:
    def test_acquire_timeout_prevents_deadlock(self, resource_manager):
        tokens1 = resource_manager.acquire((ResourceType.TORCH,), "c1")
        start = time.time()
        with pytest.raises(TimeoutError):
            resource_manager.acquire((ResourceType.TORCH,), "c2", timeout=0.1)
        elapsed = time.time() - start
        assert elapsed < 1.0
        resource_manager.release(tokens1, "c1")

    def test_force_release_breaks_deadlock(self, resource_manager):
        resource_manager.acquire((ResourceType.TORCH,), "c1")
        resource_manager.force_release("c1")
        tokens2 = resource_manager.acquire((ResourceType.TORCH,), "c2")
        assert ResourceType.TORCH in tokens2
        resource_manager.release(tokens2, "c2")


# ═══════════════════════════════════════════════════════════
#  RUPTURA: ORCHESTRATOR EDGE CASES
# ═══════════════════════════════════════════════════════════

class TestOrchestratorRupture:
    def test_start_stop_rapid(self, orchestrator):
        for _ in range(5):
            orchestrator.start()
            time.sleep(0.01)
            orchestrator.stop()

    def test_register_after_start(self, orchestrator, simple_cycle):
        orchestrator.start()
        orchestrator.register_cycle(simple_cycle)
        time.sleep(2.0)
        assert simple_cycle.iterations > 0
        orchestrator.stop()

    def test_unregister_during_execution(self, orchestrator, slow_cycle):
        orchestrator.start()
        orchestrator.register_cycle(slow_cycle)
        time.sleep(0.2)
        orchestrator.unregister_cycle(slow_cycle.id)
        orchestrator.stop()

    def test_cycle_error_doesnt_stop_orchestrator(self, orchestrator, failing_cycle):
        orchestrator.register_cycle(failing_cycle)
        orchestrator.start()
        time.sleep(1)
        assert orchestrator._running
        orchestrator.stop()

    def test_stats_availability(self, orchestrator):
        stats = orchestrator.get_stats()
        assert "running" in stats
        assert "cycle_count" in stats


# ═══════════════════════════════════════════════════════════
#  RUPTURA: MEMORY AND RESOURCE LEAKS
# ═══════════════════════════════════════════════════════════

class TestResourceLeaks:
    def test_resource_release_on_cycle_error(self, orchestrator, failing_cycle):
        resource_manager = orchestrator._resource_manager
        initial_available = resource_manager.available()

        orchestrator.register_cycle(failing_cycle)
        orchestrator.start()
        time.sleep(1)
        orchestrator.stop()

        final_available = resource_manager.available()
        assert initial_available == final_available

    def test_shared_state_cleanup(self, shared_state):
        shared_state.update_cycle("temp", {"data": "x"})
        assert shared_state.get_cycle("temp") != {}
        shared_state.remove_cycle("temp")
        assert shared_state.get_cycle("temp") == {}
