"""
Event Bus assíncrono para ciclos SWA.

Pub/Sub com fila de prioridade e dispatch assíncrono.
Thread-safe e com suporte a handlers síncronos e assíncronos.
"""
from __future__ import annotations

import time
import threading
import queue
import logging
from typing import Any, Callable, Optional
from collections import defaultdict

logger = logging.getLogger("swa.cycle.event_bus")


class EventFilter:
    """Filtro para eventos."""

    def __init__(
        self,
        event_types: Optional[set[str]] = None,
        cycle_names: Optional[set[str]] = None,
        min_priority: int = 0,
    ):
        self.event_types = event_types
        self.cycle_names = cycle_names
        self.min_priority = min_priority

    def matches(self, event: Any, priority: int = 2) -> bool:
        if self.event_types and type(event).__name__ not in self.event_types:
            return False
        if self.cycle_names:
            cycle_name = getattr(event, "cycle_name", None)
            if cycle_name and cycle_name not in self.cycle_names:
                return False
        if priority > self.min_priority:
            return False
        return True


class Subscription:
    """Assinatura de evento."""

    def __init__(
        self,
        handler: Callable[[Any], Any],
        event_filter: Optional[EventFilter] = None,
        max_retries: int = 3,
    ):
        self.id = str(id(handler)) + str(time.time())
        self.handler = handler
        self.filter = event_filter
        self.max_retries = max_retries
        self.active = True
        self.errors = 0
        self.last_error: Optional[str] = None


class EventBus:
    """Event bus assíncrono com fila de prioridade."""

    def __init__(self, max_queue_size: int = 1000):
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._event_count = 0
        self._dispatch_count = 0
        self._error_count = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="EventBus"
        )
        self._thread.start()
        logger.info("EventBus started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("EventBus stopped")

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Any], Any],
        event_filter: Optional[EventFilter] = None,
    ) -> str:
        sub = Subscription(handler, event_filter)
        with self._lock:
            self._subscriptions[event_type].append(sub)
        logger.debug(f"Subscribed to {event_type}: {sub.id}")
        return sub.id

    def unsubscribe(self, event_type: str, subscription_id: str):
        with self._lock:
            if event_type in self._subscriptions:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type] if s.id != subscription_id
                ]

    def emit(self, event: Any, priority: int = 2):
        event_type = type(event).__name__
        try:
            self._queue.put((priority, time.time(), event_type, event), block=False)
            self._event_count += 1
        except queue.Full:
            logger.warning(f"EventBus queue full, dropping event: {event_type}")
            self._error_count += 1

    def _dispatch_loop(self):
        while self._running:
            try:
                priority, timestamp, event_type, event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                subs = list(self._subscriptions.get(event_type, []))

            for sub in subs:
                if not sub.active:
                    continue

                if sub.filter and not sub.filter.matches(event, priority):
                    continue

                retries = 0
                while retries <= sub.max_retries:
                    try:
                        sub.handler(event)
                        self._dispatch_count += 1
                        break
                    except Exception as e:
                        retries += 1
                        sub.errors += 1
                        sub.last_error = str(e)
                        if retries > sub.max_retries:
                            logger.error(
                                f"EventBus handler failed after {retries} retries: {e}"
                            )
                            self._error_count += 1

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "queue_size": self._queue.qsize(),
                "event_count": self._event_count,
                "dispatch_count": self._dispatch_count,
                "error_count": self._error_count,
                "subscriptions": {
                    k: len(v) for k, v in self._subscriptions.items()
                },
            }
