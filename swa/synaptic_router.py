"""
Synaptic Bus Router — Barramento de comunicacao sinaptica O(1).
(Pure Redis Version - Independente de modulos externos como RedisJSON)
"""

from __future__ import annotations
import json
import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from swa.synaptic_address import (
    SynapticAddress,
    SynapticChannel,
    SynapticNodeState,
    SynapticRegion,
)

logger = logging.getLogger("SynapticRouter")

DEFAULT_NODE_TTL = 3600
SENSORY_TTL = 300
MEMORY_TTL = 86400 * 30
CORTEX_TTL = 86400

def _ttl_for_region(region: SynapticRegion) -> int:
    return {
        SynapticRegion.SENSORY:  SENSORY_TTL,
        SynapticRegion.CORTEX:   CORTEX_TTL,
        SynapticRegion.ADAPTER:  CORTEX_TTL,
        SynapticRegion.MEMORY:   MEMORY_TTL,
        SynapticRegion.TRACE:    MEMORY_TTL,
        SynapticRegion.IDENTITY: MEMORY_TTL,
        SynapticRegion.GRID:     DEFAULT_NODE_TTL,
    }.get(region, DEFAULT_NODE_TTL)

class SynapticBusRouter:
    def __init__(self, redis_client=None, redis_host: str = "localhost", redis_port: int = 6379):
        if redis_client is not None:
            self.r = redis_client
        else:
            try:
                import redis
                self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, socket_connect_timeout=3)
                self.r.ping()
            except Exception:
                self.r = None

        self._subscribers: Dict[str, List[Callable]] = {}
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    def register_node(self, state: SynapticNodeState, ttl: Optional[int] = None) -> bool:
        if self.r is None: return False
        key = state.address.vector_path
        effective_ttl = ttl or _ttl_for_region(state.address.region)
        try:
            pipe = self.r.pipeline()
            # Salva como string JSON pura para compatibilidade universal
            pipe.set(key, json.dumps(state.model_dump(mode="json")))
            pipe.expire(key, effective_ttl)
            pipe.sadd(f"aeon:v6:_registry:{state.address.region.value}", key)
            pipe.execute()
            return True
        except Exception:
            return False

    def unregister_node(self, address: SynapticAddress) -> bool:
        if self.r is None: return False
        key = address.vector_path
        try:
            pipe = self.r.pipeline()
            pipe.delete(key)
            pipe.srem(f"aeon:v6:_registry:{address.region.value}", key)
            pipe.execute()
            return True
        except Exception:
            return False

    def pulse(self, target: SynapticAddress, payload: dict) -> bool:
        if self.r is None: return False
        key = target.vector_path
        payload["_ts"] = int(time.time() * 1000)
        try:
            # Fetch-update-set para evitar dependencia de JSON.SET
            current_data = self.r.get(key)
            if current_data:
                try:
                    state_dict = json.loads(current_data)
                    state_dict.setdefault("telemetry", {})["last_pulse"] = payload
                    state_dict["status"] = "firing"
                    self.r.set(key, json.dumps(state_dict))
                except json.JSONDecodeError:
                    pass
            
            # Fan-out no pub/sub
            self.r.publish(target.signal_channel, json.dumps({
                "vector": key,
                "payload": payload,
            }))
            return True
        except Exception:
            return False

    def get_node_state(self, address: SynapticAddress) -> Optional[dict]:
        if self.r is None: return None
        try:
            data = self.r.get(address.vector_path)
            return json.loads(data) if data else None
        except Exception:
            return None

    def get_shm_pointer(self, address: SynapticAddress) -> Optional[Tuple[int, int]]:
        state = self.get_node_state(address)
        if state and "shm_offset" in state and "shm_size" in state:
            return state["shm_offset"], state["shm_size"]
        return None

    def get_active_nodes(self, region: Optional[SynapticRegion] = None) -> List[str]:
        if self.r is None: return []
        try:
            if region:
                return list(self.r.smembers(f"aeon:v6:_registry:{region.value}"))
            nodes = []
            for r in SynapticRegion:
                nodes.extend(self.r.smembers(f"aeon:v6:_registry:{r.value}"))
            return nodes
        except Exception:
            return []

    def subscribe(self, region: SynapticRegion, callback: Callable[[str, dict], None]):
        channel = f"aeon:v6:{region.value}:signals"
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)
        if not self._running:
            self._start_listener()

    def _start_listener(self):
        if self.r is None or self._running: return
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

    def _listen_loop(self):
        try:
            pubsub = self.r.pubsub()
            for channel in self._subscribers:
                pubsub.subscribe(channel)
            for message in pubsub.listen():
                if not self._running: break
                if message["type"] != "message": continue
                channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
                try:
                    data = json.loads(message["data"])
                    for cb in self._subscribers.get(channel, []):
                        cb(data.get("vector", ""), data.get("payload", {}))
                except Exception:
                    continue
        finally:
            self._running = False

    def stop(self):
        self._running = False

    def is_alive(self) -> bool:
        if self.r is None: return False
        try:
            return self.r.ping()
        except Exception:
            return False

    def topology_snapshot(self) -> Dict[str, int]:
        return {r.value: len(self.get_active_nodes(r)) for r in SynapticRegion}
