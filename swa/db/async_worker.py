"""Asynchronous database worker for non-blocking writes."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional


class DBAsyncWorker:
    """Background worker that drains a queue of DB write operations."""

    def __init__(
        self,
        flush_interval: float = 5.0,
        max_batch: int = 100,
    ) -> None:
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._max_batch = max_batch
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[Callable[[List[Dict[str, Any]]], None]] = None

    def set_handler(self, handler: Callable[[List[Dict[str, Any]]], None]) -> None:
        self._handler = handler

    def enqueue(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._queue.append(item)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.flush()

    def flush(self) -> None:
        if self._handler is None:
            return
        with self._lock:
            batch = self._queue[:]
            self._queue.clear()
        if batch:
            try:
                self._handler(batch)
            except Exception:
                pass

    def _run_loop(self) -> None:
        while self._running:
            time.sleep(self._flush_interval)
            self.flush()
