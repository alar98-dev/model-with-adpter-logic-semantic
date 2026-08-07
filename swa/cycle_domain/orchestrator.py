"""
Cycle Orchestrator — Aggregate Root para ciclos eventuais.

Orquestra múltiplos ciclos com:
- Scheduling assíncrono por prioridade
- Gerenciamento de recursos com semáforos
- Eventos de ciclo via EventBus
- Estado compartilhado via SHM
- Detecção de deadlock e timeout
"""
from __future__ import annotations

import os
import time
import signal
import threading
import traceback
import logging
from typing import Any, Callable, Optional
from collections import defaultdict

from .models import (
    Cycle,
    CycleConfig,
    CycleState,
    Phase,
    PhaseResult,
    Heartbeat,
    ResourceType,
    CycleStarted,
    CycleCompleted,
    CycleError,
    PhaseCompleted,
    HeartbeatEmitted,
    SystemWarning,
)
from .event_bus import EventBus
from .resource_manager import ResourceManager
from .shared_state import SharedCycleState

logger = logging.getLogger("swa.cycle.orchestrator")


class CycleContext:
    """Contexto executado durante um ciclo."""

    def __init__(
        self,
        cycle_id: str,
        cycle_name: str,
        resource_tokens: dict[ResourceType, Any],
        shared_state: SharedCycleState,
        event_bus: EventBus,
    ):
        self.cycle_id = cycle_id
        self.cycle_name = cycle_name
        self.resources = resource_tokens
        self.shared_state = shared_state
        self.event_bus = event_bus
        self.start_time = time.time()
        self.metrics: dict[str, Any] = {}

    def set_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def get_metric(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def emit_event(self, event: Any, priority: int = 2):
        self.event_bus.emit(event, priority)

    def update_state(self, metrics: dict[str, Any]):
        self.shared_state.update_and_notify(self.cycle_name, metrics)


class CycleOrchestrator:
    """Aggregate Root - orquestra todos os ciclos."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        resource_manager: Optional[ResourceManager] = None,
        shared_state: Optional[SharedCycleState] = None,
    ):
        self._cycles: dict[str, Cycle] = {}
        self._event_bus = event_bus or EventBus()
        self._resource_manager = resource_manager or ResourceManager()
        self._shared_state = shared_state or SharedCycleState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cycle_count = 0
        self._error_count = 0
        self._heartbeat_count = 0
        self._start_time: Optional[float] = None

    def register_cycle(self, cycle: Cycle) -> str:
        with self._lock:
            self._cycles[cycle.id] = cycle
            self._cycle_count += 1
            logger.info(f"Registered cycle: {cycle.name} ({cycle.id})")
            return cycle.id

    def unregister_cycle(self, cycle_id: str):
        with self._lock:
            cycle = self._cycles.pop(cycle_id, None)
            if cycle:
                cycle.stop()
                self._resource_manager.force_release(cycle_id)
                logger.info(f"Unregistered cycle: {cycle.name}")

    def get_cycle(self, cycle_id: str) -> Optional[Cycle]:
        return self._cycles.get(cycle_id)

    def get_cycle_by_name(self, name: str) -> Optional[Cycle]:
        for cycle in self._cycles.values():
            if cycle.name == name:
                return cycle
        return None

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._event_bus.start()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="CycleOrchestrator"
        )
        self._thread.start()
        logger.info("CycleOrchestrator started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        for cycle in self._cycles.values():
            cycle.stop()
        self._event_bus.stop()
        logger.info("CycleOrchestrator stopped")

    def pause_cycle(self, cycle_id: str):
        cycle = self._cycles.get(cycle_id)
        if cycle:
            cycle.pause()
            self._event_bus.emit(CyclePaused(
                cycle_id=cycle_id,
                cycle_name=cycle.name,
                reason="manual_pause",
            ))

    def resume_cycle(self, cycle_id: str):
        cycle = self._cycles.get(cycle_id)
        if cycle:
            cycle.resume()

    def _run_loop(self):
        while self._running:
            try:
                self._execute_ready_cycles()
                self._emit_heartbeat()
                self._check_stale_cycles()
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Orchestrator loop error: {e}")
                traceback.print_exc()
                self._error_count += 1
                time.sleep(5.0)

    def _execute_ready_cycles(self):
        available = self._resource_manager.available()

        ready = []
        with self._lock:
            for cycle in self._cycles.values():
                if cycle.can_run(available):
                    ready.append(cycle)

        ready.sort(key=lambda c: c.config.priority)

        for cycle in ready:
            try:
                self._execute_cycle(cycle)
            except Exception as e:
                logger.error(f"Error executing cycle {cycle.name}: {e}")
                self._error_count += 1

    def _execute_cycle(self, cycle: Cycle):
        self._event_bus.emit(CycleStarted(
            cycle_id=cycle.id,
            cycle_name=cycle.name,
        ))

        try:
            tokens = self._resource_manager.acquire(
                cycle.config.resource_requirements,
                cycle.id,
                timeout=5.0,
            )
        except TimeoutError:
            logger.warning(f"Failed to acquire resources for {cycle.name}")
            self._event_bus.emit(SystemWarning(
                warning_type="resource_timeout",
                message=f"Failed to acquire resources for {cycle.name}",
                severity="medium",
            ))
            return

        try:
            context = CycleContext(
                cycle_id=cycle.id,
                cycle_name=cycle.name,
                resource_tokens=tokens,
                shared_state=self._shared_state,
                event_bus=self._event_bus,
            )

            result = cycle.run(context)

            for rt, token in tokens.items():
                self._event_bus.emit(PhaseCompleted(
                    cycle_id=cycle.id,
                    cycle_name=cycle.name,
                    phase=f"resource_{rt.value}",
                    duration_ms=token.age_seconds() * 1000,
                    success=True,
                ))

            if result.success:
                self._event_bus.emit(CycleCompleted(
                    cycle_id=cycle.id,
                    cycle_name=cycle.name,
                    iterations=cycle.iterations,
                    duration_ms=result.duration_ms,
                ))
            else:
                self._event_bus.emit(CycleError(
                    cycle_id=cycle.id,
                    cycle_name=cycle.name,
                    error=result.error or "unknown",
                    phase=result.phase_name,
                ))

        finally:
            self._resource_manager.release(tokens, cycle.id)

    def _emit_heartbeat(self):
        self._heartbeat_count += 1
        for cycle in self._cycles.values():
            heartbeat = Heartbeat(
                timestamp=time.time(),
                cycle_name=cycle.name,
                state=cycle.state,
                phase=cycle.last_result.phase_name if cycle.last_result else "idle",
                iterations=cycle.iterations,
                errors=cycle.errors,
            )
            self._event_bus.emit(HeartbeatEmitted(heartbeat=heartbeat), priority=3)

            self._shared_state.update_and_notify(
                cycle.name,
                {
                    "state": cycle.state.value,
                    "iterations": cycle.iterations,
                    "errors": cycle.errors,
                    "last_run": cycle.last_run,
                },
            )

    def _check_stale_cycles(self):
        stale = self._shared_state.get_stale_cycles(max_age_seconds=300.0)
        for cycle_name in stale:
            self._event_bus.emit(SystemWarning(
                warning_type="stale_cycle",
                message=f"Cycle {cycle_name} has not updated in 5 minutes",
                severity="low",
            ))

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "cycle_count": self._cycle_count,
            "error_count": self._error_count,
            "heartbeat_count": self._heartbeat_count,
            "resource_manager": self._resource_manager.get_stats(),
            "event_bus": self._event_bus.get_stats(),
            "cycles": {
                cycle.id: cycle.to_dict() for cycle in self._cycles.values()
            },
        }
