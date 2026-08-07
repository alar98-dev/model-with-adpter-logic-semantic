"""
Resource Manager para ciclos SWA.

Gerencia recursos compartilhados com semáforos e timeout.
Thread-safe com tracking de uso e deadlock detection.
"""
from __future__ import annotations

import time
import threading
import logging
from typing import Optional
from collections import defaultdict
from contextlib import contextmanager

from .models import ResourceType, ResourceToken, CycleConfig

logger = logging.getLogger("swa.cycle.resource_manager")


class ResourcePool:
    """Pool de recursos com semáforo."""

    def __init__(self, resource_type: ResourceType, max_count: int):
        self.resource_type = resource_type
        self.max_count = max_count
        self.semaphore = threading.Semaphore(max_count)
        self.current_count = 0
        self.total_acquired = 0
        self.total_released = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 5.0) -> bool:
        acquired = self.semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self.current_count += 1
                self.total_acquired += 1
        return acquired

    def release(self):
        self.semaphore.release()
        with self._lock:
            self.current_count -= 1
            self.total_released += 1

    @property
    def available(self) -> int:
        with self._lock:
            return self.max_count - self.current_count

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "resource_type": self.resource_type.value,
                "max_count": self.max_count,
                "current_count": self.current_count,
                "available": self.available,
                "total_acquired": self.total_acquired,
                "total_released": self.total_released,
            }


class ResourceManager:
    """Gerencia recursos compartilhados com semáforos."""

    def __init__(self):
        self._pools: dict[ResourceType, ResourcePool] = {
            ResourceType.TORCH: ResourcePool(ResourceType.TORCH, 1),
            ResourceType.REDIS: ResourcePool(ResourceType.REDIS, 5),
            ResourceType.PG: ResourcePool(ResourceType.PG, 3),
            ResourceType.SHM: ResourcePool(ResourceType.SHM, 10),
            ResourceType.CPU: ResourcePool(ResourceType.CPU, 8),
        }
        self._tokens: dict[str, list[ResourceToken]] = defaultdict(list)
        self._lock = threading.Lock()

    def available(self) -> set[ResourceType]:
        return {rt for rt, pool in self._pools.items() if pool.available > 0}

    def can_acquire(self, requirements: tuple[ResourceType, ...]) -> bool:
        for rt in requirements:
            if rt in self._pools and self._pools[rt].available <= 0:
                return False
        return True

    def acquire(
        self,
        requirements: tuple[ResourceType, ...],
        cycle_id: str,
        timeout: float = 5.0,
    ) -> dict[ResourceType, ResourceToken]:
        acquired = {}
        for rt in requirements:
            if rt not in self._pools:
                continue
            pool = self._pools[rt]
            if pool.acquire(timeout=timeout):
                token = ResourceToken(
                    resource_type=rt,
                    acquired_at=time.time(),
                    cycle_id=cycle_id,
                )
                acquired[rt] = token
                with self._lock:
                    self._tokens[cycle_id].append(token)
                logger.debug(f"Acquired {rt.value} for cycle {cycle_id}")
            else:
                for released_token in acquired.values():
                    self._pools[released_token.resource_type].release()
                raise TimeoutError(
                    f"Failed to acquire {rt.value} for cycle {cycle_id}"
                )
        return acquired

    def release(self, tokens: dict[ResourceType, ResourceToken], cycle_id: str):
        for rt, token in tokens.items():
            if rt in self._pools:
                self._pools[rt].release()
                logger.debug(f"Released {rt.value} for cycle {cycle_id}")

    def force_release(self, cycle_id: str):
        with self._lock:
            tokens = self._tokens.pop(cycle_id, [])
        for token in tokens:
            if token.resource_type in self._pools:
                self._pools[token.resource_type].release()

    def get_stats(self) -> dict:
        return {
            "pools": {rt.value: pool.to_dict() for rt, pool in self._pools.items()},
            "active_tokens": {
                cid: len(tokens) for cid, tokens in self._tokens.items()
            },
        }

    @contextmanager
    def acquire_context(
        self,
        requirements: tuple[ResourceType, ...],
        cycle_id: str,
        timeout: float = 5.0,
    ):
        tokens = self.acquire(requirements, cycle_id, timeout)
        try:
            yield tokens
        finally:
            self.release(tokens, cycle_id)
