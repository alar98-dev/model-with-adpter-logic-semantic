"""
Shared Cycle State — estado compartilhado via shared memory.

Thread-safe com persistência em SHM para comunicação entre processos.
"""
from __future__ import annotations

import os
import json
import time
import threading
import logging
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger("swa.cycle.shared_state")


class SharedCycleState:
    """Estado compartilhado via shared memory - thread-safe."""

    def __init__(self, shm_path: str = "/dev/shm/swa_cycle_state.json"):
        self._path = Path(shm_path)
        self._lock = threading.RLock()
        self._state: dict[str, Any] = self._load()
        self._watchers: list[callable] = []

    def _load(self) -> dict:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load state from {self._path}: {e}")
        return {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save state to {self._path}: {e}")

    def update_cycle(self, cycle_name: str, metrics: dict[str, Any]):
        with self._lock:
            if cycle_name not in self._state:
                self._state[cycle_name] = {}
            self._state[cycle_name].update(metrics)
            self._state[cycle_name]["last_update"] = time.time()
            self._save()

    def get_cycle(self, cycle_name: str) -> dict:
        with self._lock:
            return self._state.get(cycle_name, {}).copy()

    def get_all(self) -> dict:
        with self._lock:
            return self._state.copy()

    def remove_cycle(self, cycle_name: str):
        with self._lock:
            self._state.pop(cycle_name, None)
            self._save()

    def clear(self):
        with self._lock:
            self._state.clear()
            self._save()

    def add_watcher(self, callback: callable):
        self._watchers.append(callback)

    def _notify_watchers(self, cycle_name: str, data: dict):
        for watcher in self._watchers:
            try:
                watcher(cycle_name, data)
            except Exception as e:
                logger.error(f"Watcher error: {e}")

    def update_and_notify(self, cycle_name: str, metrics: dict[str, Any]):
        with self._lock:
            if cycle_name not in self._state:
                self._state[cycle_name] = {}
            self._state[cycle_name].update(metrics)
            self._state[cycle_name]["last_update"] = time.time()
            self._save()
            self._notify_watchers(cycle_name, self._state[cycle_name])

    def get_stale_cycles(self, max_age_seconds: float = 300.0) -> list[str]:
        now = time.time()
        stale = []
        with self._lock:
            for cycle_name, data in self._state.items():
                if not isinstance(data, dict):
                    continue
                last_update = data.get("last_update", 0)
                if now - last_update > max_age_seconds:
                    stale.append(cycle_name)
        return stale
