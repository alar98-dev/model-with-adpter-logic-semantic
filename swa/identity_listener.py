"""Redis-based identity presence listener for the OODA orchestrator.

Subscribes to ``swa:vision:identity_recognized`` and exposes the latest
identity presence data for the Observer to inject into the cognitive cycle.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

try:
    import redis as _redis
except ImportError:
    _redis = None


class IdentityPresenceListener:
    """Background listener that tracks who is currently present via face recognition.

    Thread-safe: the orchestrator can call ``snapshot()`` from any thread.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6379) -> None:
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._running = False

        self.current_identity: Optional[Dict[str, Any]] = None
        self.last_seen: float = 0.0
        self.presence_history: list = []
        self.max_history = 50

    def start(self) -> None:
        """Start the background Redis subscriber thread."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        print("[IDENTITY_LISTENER] Escutando swa:vision:identity_recognized...")

    def stop(self) -> None:
        self._running = False

    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the current identity presence (thread-safe)."""
        with self._lock:
            if self.current_identity is None:
                return None
            age = time.time() - self.last_seen
            return {**self.current_identity, "age_seconds": round(age, 1)}

    def _listen_loop(self) -> None:
        if _redis is None:
            print("[IDENTITY_LISTENER] redis-py não instalado — desabilitado")
            return

        try:
            r = _redis.Redis(host=self._host, port=self._port, db=0)
            r.ping()
        except Exception as e:
            print(f"[IDENTITY_LISTENER] Redis indisponível: {e}")
            return

        pubsub = r.pubsub()
        pubsub.subscribe("swa:vision:identity_recognized")

        for message in pubsub.listen():
            if not self._running:
                break
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"].decode("utf-8"))
                with self._lock:
                    self.current_identity = data
                    self.last_seen = time.time()
                    self.presence_history.append(data)
                    if len(self.presence_history) > self.max_history:
                        self.presence_history = self.presence_history[-self.max_history:]
            except Exception:
                pass

        pubsub.close()
