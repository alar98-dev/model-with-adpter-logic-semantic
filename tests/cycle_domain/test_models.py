"""
Testes unitários dos domain models do CycleDomain.

Cobre: Value Objects, Entities, Enums, Domain Events.
"""
import pytest
import time
import threading
from dataclasses import FrozenInstanceError
from swa.cycle_domain import (
    Cycle,
    CycleConfig,
    CycleState,
    Phase,
    PhaseState,
    PhaseResult,
    ResourceType,
    Heartbeat,
    ResourceToken,
    CycleStarted,
    CycleCompleted,
    CyclePaused,
    CycleError,
    PhaseCompleted,
    HeartbeatEmitted,
    SystemWarning,
)


# ═══════════════════════════════════════════════════════════
#  CYCLECONFIG (Value Object)
# ═══════════════════════════════════════════════════════════

class TestCycleConfig:
    def test_create_valid_config(self):
        config = CycleConfig(
            name="test",
            interval_seconds=10.0,
            max_duration_seconds=60.0,
            priority=2,
            resource_requirements=(ResourceType.REDIS,),
        )
        assert config.name == "test"
        assert config.interval_seconds == 10.0
        assert config.priority == 2

    def test_config_is_immutable(self):
        config = CycleConfig(name="test", interval_seconds=10.0)
        with pytest.raises(FrozenInstanceError):
            config.name = "changed"

    def test_config_invalid_interval(self):
        with pytest.raises(ValueError):
            CycleConfig(name="test", interval_seconds=-1)

    def test_config_invalid_duration(self):
        with pytest.raises(ValueError):
            CycleConfig(name="test", interval_seconds=1, max_duration_seconds=0)

    def test_config_defaults(self):
        config = CycleConfig(name="test", interval_seconds=1.0)
        assert config.max_duration_seconds == 60.0
        assert config.priority == 2
        assert config.auto_restart is True
        assert config.max_errors == 5


# ═══════════════════════════════════════════════════════════
#  PHASERESULT (Value Object)
# ═══════════════════════════════════════════════════════════

class TestPhaseResult:
    def test_create_success_result(self):
        result = PhaseResult(
            phase_name="test",
            success=True,
            duration_ms=100.0,
        )
        assert result.success is True
        assert result.error is None

    def test_create_failure_result(self):
        result = PhaseResult(
            phase_name="test",
            success=False,
            duration_ms=100.0,
            error="timeout",
        )
        assert result.success is False
        assert result.error == "timeout"

    def test_result_is_immutable(self):
        result = PhaseResult(phase_name="test", success=True, duration_ms=100.0)
        with pytest.raises(FrozenInstanceError):
            result.success = False

    def test_result_to_dict(self):
        result = PhaseResult(
            phase_name="test",
            success=True,
            duration_ms=100.0,
            metrics=(("key", 42.0),),
        )
        d = result.to_dict()
        assert d["phase_name"] == "test"
        assert d["success"] is True
        assert d["metrics"]["key"] == 42.0


# ═══════════════════════════════════════════════════════════
#  PHASE (Entity)
# ═══════════════════════════════════════════════════════════

class TestPhase:
    def test_create_phase(self):
        def handler(ctx):
            pass

        phase = Phase("test", handler, timeout=5.0)
        assert phase.name == "test"
        assert phase.state == PhaseState.PENDING

    def test_phase_execute_success(self):
        executed = []

        def handler(ctx):
            executed.append(True)

        phase = Phase("test", handler)
        result = phase.execute(None)

        assert result.success is True
        assert phase.state == PhaseState.COMPLETED
        assert executed == [True]

    def test_phase_execute_failure(self):
        def handler(ctx):
            raise ValueError("error")

        phase = Phase("test", handler)
        result = phase.execute(None)

        assert result.success is False
        assert result.error == "error"
        assert phase.state == PhaseState.FAILED

    def test_phase_execute_timeout(self):
        def handler(ctx):
            time.sleep(10)

        phase = Phase("test", handler, timeout=0.1)
        result = phase.execute(None)

        assert result.success is False
        assert result.error.startswith("timeout")
        assert phase.state == PhaseState.TIMEOUT

    def test_phase_stores_last_result(self):
        def handler(ctx):
            pass

        phase = Phase("test", handler)
        result = phase.execute(None)

        assert phase.last_result is result
        assert phase.last_result.success is True


# ═══════════════════════════════════════════════════════════
#  CYCLE (Entity)
# ═══════════════════════════════════════════════════════════

class TestCycle:
    def test_create_cycle(self, simple_cycle):
        assert simple_cycle.name == "test_cycle"
        assert simple_cycle.state == CycleState.IDLE
        assert simple_cycle.iterations == 0

    def test_cycle_run_success(self, simple_cycle):
        ctx = type('MockCtx', (), {'set_metric': lambda s, k, v: None})()
        result = simple_cycle.run(ctx)

        assert result.success is True
        assert simple_cycle.iterations == 1
        assert simple_cycle.state == CycleState.IDLE

    def test_cycle_run_failure(self, failing_cycle):
        result = failing_cycle.run(None)

        assert result.success is False
        assert failing_cycle.errors == 1
        assert failing_cycle.state == CycleState.ERROR

    def test_cycle_consecutive_errors(self, failing_cycle):
        for _ in range(3):
            failing_cycle.run(None)

        assert failing_cycle.consecutive_errors == 3

    def test_cycle_max_errors_prevents_run(self, failing_cycle):
        for _ in range(3):
            failing_cycle.run(None)

        assert not failing_cycle.can_run(set())

    def test_cycle_can_run(self, simple_cycle):
        assert simple_cycle.can_run(set())

    def test_cycle_can_run_needs_resources(self, resource_heavy_cycle):
        assert not resource_heavy_cycle.can_run(set())
        assert resource_heavy_cycle.can_run({ResourceType.TORCH, ResourceType.REDIS})

    def test_cycle_pause_resume(self, simple_cycle):
        simple_cycle.state = CycleState.RUNNING
        simple_cycle.pause()
        assert simple_cycle.state == CycleState.PAUSED

        simple_cycle.resume()
        assert simple_cycle.state == CycleState.IDLE

    def test_cycle_stop(self, simple_cycle):
        simple_cycle.stop()
        assert simple_cycle.state == CycleState.STOPPED

    def test_cycle_to_dict(self, simple_cycle):
        d = simple_cycle.to_dict()
        assert d["name"] == "test_cycle"
        assert d["state"] == "idle"
        assert d["iterations"] == 0


# ═══════════════════════════════════════════════════════════
#  DOMAIN EVENTS
# ═══════════════════════════════════════════════════════════

class TestDomainEvents:
    def test_cycle_started(self):
        event = CycleStarted(cycle_id="123", cycle_name="test")
        assert event.cycle_id == "123"
        assert event.timestamp > 0

    def test_cycle_completed(self):
        event = CycleCompleted(
            cycle_id="123", cycle_name="test", iterations=5, duration_ms=100.0
        )
        assert event.iterations == 5

    def test_cycle_error(self):
        event = CycleError(
            cycle_id="123", cycle_name="test", error="fail", phase="test_phase"
        )
        assert event.error == "fail"

    def test_heartbeat(self):
        hb = Heartbeat(
            timestamp=time.time(),
            cycle_name="test",
            state=CycleState.IDLE,
            phase="idle",
            iterations=0,
            errors=0,
        )
        assert hb.to_dict()["cycle_name"] == "test"

    def test_system_warning(self):
        warn = SystemWarning(
            warning_type="resource_timeout", message="test", severity="high"
        )
        assert warn.severity == "high"


# ═══════════════════════════════════════════════════════════
#  THREAD SAFETY
# ═══════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_cycle_concurrent_access(self):
        config = CycleConfig(
            name="thread_test", interval_seconds=1, resource_requirements=()
        )
        cycle = Cycle(config)
        errors = []

        def run_cycle():
            try:
                for _ in range(10):
                    cycle.run(None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cycle) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert cycle.iterations == 50
