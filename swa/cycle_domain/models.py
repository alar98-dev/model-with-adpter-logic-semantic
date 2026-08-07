"""
SWA Cycle Domain — Modelos de domínio para ciclos eventuais.

Bounded Contexts:
  - CycleDomain: Configuração e execução de ciclos
  - StateDomain: Estado compartilhado via SHM
  - ResourceDomain: Gerenciamento de recursos compartilhados
"""
from __future__ import annotations

import time
import json
import threading
import traceback
import queue
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4
from pathlib import Path


# ═══════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════

class CycleState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class PhaseState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ResourceType(Enum):
    TORCH = "torch"
    REDIS = "redis"
    PG = "pg"
    SHM = "shm"
    CPU = "cpu"


# ═══════════════════════════════════════════════════════════
#  VALUE OBJECTS (Imutáveis)
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CycleConfig:
    """Configuração de um ciclo - imutável após criação."""
    name: str
    interval_seconds: float
    max_duration_seconds: float = 60.0
    max_iterations: int = 1000
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    resource_requirements: tuple[ResourceType, ...] = ()
    auto_restart: bool = True
    max_errors: int = 5

    def __post_init__(self):
        if self.interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {self.interval_seconds}")
        if self.max_duration_seconds <= 0:
            raise ValueError(f"max_duration_seconds must be > 0, got {self.max_duration_seconds}")


@dataclass(frozen=True)
class PhaseResult:
    """Resultado de uma fase - imutável."""
    phase_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    metrics: tuple[tuple[str, float],...] = ()

    def to_dict(self) -> dict:
        return {
            "phase_name": self.phase_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class Heartbeat:
    """Heartbeat do ciclo - imutável."""
    timestamp: float
    cycle_name: str
    state: CycleState
    phase: str
    iterations: int
    errors: int
    cpu_percent: float = 0.0
    ram_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResourceToken:
    """Token de recurso adquirido - imutável."""
    resource_type: ResourceType
    acquired_at: float
    cycle_id: str

    def age_seconds(self) -> float:
        return time.time() - self.acquired_at


# ═══════════════════════════════════════════════════════════
#  DOMAIN EVENTS
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CycleStarted:
    cycle_id: str
    cycle_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CycleCompleted:
    cycle_id: str
    cycle_name: str
    iterations: int
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CyclePaused:
    cycle_id: str
    cycle_name: str
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CycleError:
    cycle_id: str
    cycle_name: str
    error: str
    phase: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PhaseCompleted:
    cycle_id: str
    cycle_name: str
    phase: str
    duration_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ResourceAcquired:
    resource_type: ResourceType
    cycle_id: str
    cycle_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ResourceReleased:
    resource_type: ResourceType
    cycle_id: str
    cycle_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class HeartbeatEmitted:
    heartbeat: Heartbeat
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SystemWarning:
    warning_type: str
    message: str
    severity: str = "medium"
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════
#  DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════

class Phase:
    """Entidade de fase do ciclo."""

    def __init__(self, name: str, handler: Callable[[Any], Any], timeout: float = 30.0):
        self.name = name
        self.handler = handler
        self.timeout = timeout
        self.state = PhaseState.PENDING
        self.last_result: Optional[PhaseResult] = None

    def execute(self, context: Any) -> PhaseResult:
        self.state = PhaseState.RUNNING
        start = time.time()
        result_holder = [None]

        def _run():
            try:
                self.handler(context)
                result_holder[0] = ("ok", None)
            except Exception as e:
                result_holder[0] = ("error", str(e))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        elapsed_ms = (time.time() - start) * 1000

        if t.is_alive():
            self.state = PhaseState.TIMEOUT
            result = PhaseResult(
                phase_name=self.name,
                success=False,
                duration_ms=elapsed_ms,
                error=f"timeout after {self.timeout}s",
            )
            self.last_result = result
            return result

        status, error = result_holder[0] or ("error", "handler returned no result")
        if status == "ok":
            self.state = PhaseState.COMPLETED
            result = PhaseResult(phase_name=self.name, success=True, duration_ms=elapsed_ms)
        else:
            self.state = PhaseState.FAILED
            result = PhaseResult(phase_name=self.name, success=False, duration_ms=elapsed_ms, error=error)

        self.last_result = result
        return result


class Cycle:
    """Entidade raiz de um ciclo."""

    def __init__(self, config: CycleConfig):
        self.id = str(uuid4())
        self.config = config
        self.state = CycleState.IDLE
        self.iterations = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.last_run: Optional[float] = None
        self.last_result: Optional[PhaseResult] = None
        self._phases: list[Phase] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.name

    def add_phase(self, phase: Phase):
        with self._lock:
            self._phases.append(phase)

    def can_run(self, available_resources: set[ResourceType]) -> bool:
        with self._lock:
            if self.state not in (CycleState.IDLE, CycleState.ERROR):
                return False
            if self.consecutive_errors >= self.config.max_errors:
                return False
            return set(self.config.resource_requirements).issubset(available_resources)

    def run(self, context: Any) -> PhaseResult:
        with self._lock:
            self.state = CycleState.RUNNING

        start = time.time()

        try:
            for phase in self._phases:
                result = phase.execute(context)
                if not result.success:
                    with self._lock:
                        self.state = CycleState.ERROR
                        self.errors += 1
                        self.consecutive_errors += 1
                        self.last_result = result
                    return result

            duration = (time.time() - start) * 1000
            with self._lock:
                self.iterations += 1
                self.last_run = time.time()
                self.state = CycleState.IDLE
                self.consecutive_errors = 0
                self.last_result = PhaseResult(
                    phase_name="complete",
                    success=True,
                    duration_ms=duration,
                )
            return self.last_result

        except Exception as e:
            duration = (time.time() - start) * 1000
            with self._lock:
                self.state = CycleState.ERROR
                self.errors += 1
                self.consecutive_errors += 1
                self.last_result = PhaseResult(
                    phase_name="error",
                    success=False,
                    duration_ms=duration,
                    error=str(e),
                )
            return self.last_result

    def pause(self):
        with self._lock:
            if self.state == CycleState.RUNNING:
                self.state = CycleState.PAUSED

    def resume(self):
        with self._lock:
            if self.state == CycleState.PAUSED:
                self.state = CycleState.IDLE

    def stop(self):
        with self._lock:
            self.state = CycleState.STOPPED

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "name": self.config.name,
                "state": self.state.value,
                "iterations": self.iterations,
                "errors": self.errors,
                "consecutive_errors": self.consecutive_errors,
                "last_run": self.last_run,
                "last_result": self.last_result.to_dict() if self.last_result else None,
                "phases": [p.name for p in self._phases],
            }
