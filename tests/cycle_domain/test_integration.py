"""
Testes de integração para EventBus, ResourceManager e SharedState.

Cobre: fluxos de dados entre componentes, sincronização, edge cases.
"""
import pytest
import time
import threading
import queue
from swa.cycle_domain import (
    EventBus,
    EventFilter,
    ResourceManager,
    ResourceType,
    SharedCycleState,
    Cycle,
    CycleConfig,
    Phase,
    PhaseCompleted,
    HeartbeatEmitted,
    SystemWarning,
)


# ═══════════════════════════════════════════════════════════
#  EVENTBUS
# ═══════════════════════════════════════════════════════════

class TestEventBus:
    def test_emit_and_receive(self, event_bus):
        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe("PhaseCompleted", handler)
        event_bus.emit(PhaseCompleted(
            cycle_id="123", cycle_name="test", phase="p1", duration_ms=10, success=True
        ))

        time.sleep(0.2)
        assert len(received) == 1
        assert received[0].cycle_name == "test"

    def test_multiple_subscribers(self, event_bus):
        received1 = []
        received2 = []

        def handler1(event):
            received1.append(event)

        def handler2(event):
            received2.append(event)

        event_bus.subscribe("PhaseCompleted", handler1)
        event_bus.subscribe("PhaseCompleted", handler2)
        event_bus.emit(PhaseCompleted(
            cycle_id="1", cycle_name="t", phase="p", duration_ms=1, success=True
        ))

        time.sleep(0.2)
        assert len(received1) == 1
        assert len(received2) == 1

    def test_filter_by_event_type(self, event_bus):
        received = []

        def handler(event):
            received.append(event)

        filt = EventFilter(event_types={"PhaseCompleted"})
        event_bus.subscribe("PhaseCompleted", handler, filt)
        event_bus.emit(SystemWarning(warning_type="test", message="test"))

        time.sleep(0.2)
        assert len(received) == 0

    def test_filter_by_cycle_name(self, event_bus):
        received = []

        def handler(event):
            received.append(event)

        filt = EventFilter(cycle_names={"target_cycle"})
        event_bus.subscribe("PhaseCompleted", handler, filt)
        event_bus.emit(PhaseCompleted(
            cycle_id="1", cycle_name="other_cycle", phase="p", duration_ms=1, success=True
        ))
        event_bus.emit(PhaseCompleted(
            cycle_id="2", cycle_name="target_cycle", phase="p", duration_ms=1, success=True
        ))

        time.sleep(0.2)
        assert len(received) == 1
        assert received[0].cycle_name == "target_cycle"

    def test_unsubscribe(self, event_bus):
        received = []

        def handler(event):
            received.append(event)

        sub_id = event_bus.subscribe("PhaseCompleted", handler)
        event_bus.emit(PhaseCompleted(
            cycle_id="1", cycle_name="t", phase="p", duration_ms=1, success=True
        ))
        time.sleep(0.1)

        event_bus.unsubscribe("PhaseCompleted", sub_id)
        event_bus.emit(PhaseCompleted(
            cycle_id="2", cycle_name="t", phase="p", duration_ms=1, success=True
        ))
        time.sleep(0.1)

        assert len(received) == 1

    def test_handler_error_does_not_block(self, event_bus):
        received = []

        def bad_handler(event):
            raise ValueError("error")

        def good_handler(event):
            received.append(event)

        event_bus.subscribe("PhaseCompleted", bad_handler)
        event_bus.subscribe("PhaseCompleted", good_handler)
        event_bus.emit(PhaseCompleted(
            cycle_id="1", cycle_name="t", phase="p", duration_ms=1, success=True
        ))

        time.sleep(0.2)
        assert len(received) == 1

    def test_stats(self, event_bus):
        event_bus.emit(PhaseCompleted(
            cycle_id="1", cycle_name="t", phase="p", duration_ms=1, success=True
        ))
        time.sleep(0.1)
        stats = event_bus.get_stats()
        assert stats["running"] is True
        assert stats["event_count"] >= 1


# ═══════════════════════════════════════════════════════════
#  RESOURCEMANAGER
# ═══════════════════════════════════════════════════════════

class TestResourceManager:
    def test_available_resources(self, resource_manager):
        available = resource_manager.available()
        assert ResourceType.TORCH in available
        assert ResourceType.REDIS in available

    def test_acquire_and_release(self, resource_manager):
        tokens = resource_manager.acquire(
            (ResourceType.REDIS,), "cycle_1"
        )
        assert ResourceType.REDIS in tokens
        resource_manager.release(tokens, "cycle_1")

    def test_acquire_multiple_resources(self, resource_manager):
        tokens = resource_manager.acquire(
            (ResourceType.REDIS, ResourceType.PG), "cycle_1"
        )
        assert len(tokens) == 2
        resource_manager.release(tokens, "cycle_1")

    def test_acquire_timeout(self, resource_manager):
        tokens1 = resource_manager.acquire(
            (ResourceType.TORCH,), "cycle_1"
        )
        with pytest.raises(TimeoutError):
            resource_manager.acquire(
                (ResourceType.TORCH,), "cycle_2", timeout=0.1
            )
        resource_manager.release(tokens1, "cycle_1")

    def test_acquire_context_manager(self, resource_manager):
        with resource_manager.acquire_context(
            (ResourceType.REDIS,), "cycle_1"
        ) as tokens:
            assert ResourceType.REDIS in tokens

    def test_force_release(self, resource_manager):
        resource_manager.acquire((ResourceType.REDIS,), "cycle_1")
        resource_manager.force_release("cycle_1")

    def test_stats(self, resource_manager):
        stats = resource_manager.get_stats()
        assert "pools" in stats
        assert "torch" in stats["pools"]


# ═══════════════════════════════════════════════════════════
#  SHAREDCYCLESTATE
# ═══════════════════════════════════════════════════════════

class TestSharedCycleState:
    def test_update_and_get(self, shared_state):
        shared_state.update_cycle("dream", {"iterations": 5})
        data = shared_state.get_cycle("dream")
        assert data["iterations"] == 5

    def test_get_all(self, shared_state):
        shared_state.update_cycle("dream", {"iterations": 5})
        shared_state.update_cycle("cortex", {"iterations": 10})
        all_data = shared_state.get_all()
        assert "dream" in all_data
        assert "cortex" in all_data

    def test_remove_cycle(self, shared_state):
        shared_state.update_cycle("dream", {"iterations": 5})
        shared_state.remove_cycle("dream")
        assert shared_state.get_cycle("dream") == {}

    def test_clear(self, shared_state):
        shared_state.update_cycle("dream", {"iterations": 5})
        shared_state.clear()
        assert shared_state.get_all() == {}

    def test_persistence(self, tmp_path):
        state_file = tmp_path / "persist.json"
        state1 = SharedCycleState(str(state_file))
        state1.update_cycle("dream", {"iterations": 5})

        state2 = SharedCycleState(str(state_file))
        data = state2.get_cycle("dream")
        assert data["iterations"] == 5

    def test_stale_cycles(self, shared_state):
        shared_state.update_cycle("old_cycle", {"iterations": 1})
        shared_state._state["old_cycle"]["last_update"] = time.time() - 600
        stale = shared_state.get_stale_cycles(max_age_seconds=300)
        assert "old_cycle" in stale

    def test_watcher_notification(self, shared_state):
        notifications = []

        def watcher(cycle_name, data):
            notifications.append((cycle_name, data))

        shared_state.add_watcher(watcher)
        shared_state.update_and_notify("dream", {"iterations": 1})

        assert len(notifications) == 1
        assert notifications[0][0] == "dream"

    def test_thread_safety(self, shared_state):
        errors = []

        def update_cycle(name):
            try:
                for i in range(100):
                    shared_state.update_cycle(name, {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_cycle, args=(f"cycle_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert len(shared_state.get_all()) == 10
