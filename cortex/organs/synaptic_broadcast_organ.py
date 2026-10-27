"""
Synaptic Broadcast Organ — Ponte entre o IPC local e o barramento distribuido.

Este orgao acopla o Unix Domain Socket (/var/run/aeon/synapse.sock),
o mmap (/dev/shm/aeon_context_bus.dat) e o pipeline de orgaos existente
ao barramento dimensional distribuido em RedisJSON.

Fluxo:
  ContextBus (mmap local) --on_write--> SynapticBroadcastOrgan
  EventHub (pub/sub local) --subscribe--> SynapticBroadcastOrgan
  GridRegistry (shm local) --on_register--> SynapticBroadcastOrgan
  PostgreSQL (LISTEN/NOTIFY) --CDC--> SynapticBroadcastOrgan
                                           |
                                           v
                                    RedisJSON (O(1))
                                    WebSocket Pub/Sub

Hierarquia de temperatura:
  /dev/shm  (< 1us, local)  -->  Redis (< 100us, distribuido)  -->  PostgreSQL (< 10ms, duravel)

Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from swa.synaptic_address import (
    SynapticAddress,
    SynapticChannel,
    SynapticNodeState,
    SynapticRegion,
    cortex_addr,
    memory_addr,
    sensory_addr,
    trace_addr,
)
from swa.synaptic_router import SynapticBusRouter

logger = logging.getLogger("SynapticBroadcast")


# -- Constantes SHM herdadas do context_bus.py --------------------------------

CONTEXT_BUS_PATH = "/dev/shm/aeon_context_bus.dat"
SHM_TENSOR_DIR = "/dev/shm/aeon"
HEADER_SIZE = 4096
ENTRY_SIZE = 384
HEADER_FMT = "<IIIIIIIIQ"
HEADER_LEN = struct.calcsize(HEADER_FMT)


class SynapticBroadcastOrgan:
    """Orgao do Cortex que espelha ativacoes locais (IPC/SHM) para Redis distribuido.

    Opera como ponte bidirecional:
    - Local -> Redis: Espelha tokens do ContextBus, estados de orgaos, episodios
    - Redis -> Local: Recebe sinais de nos remotos e injeta no EventHub local

    Registra-se na GridRegistry como 'synaptic-broadcast' com capabilities:
    ['redis_vector', 'websocket', 'pg_listen', 'session_mgmt']
    """

    NODE_ID = "synaptic-broadcast"
    CAPABILITIES = ["redis_vector", "websocket", "pg_listen", "session_mgmt"]

    def __init__(
        self,
        router: Optional[SynapticBusRouter] = None,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
    ):
        self.router = router or SynapticBusRouter(redis_host=redis_host, redis_port=redis_port)
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_ctx_seq = 0  # Ultimo sequence number lido do ContextBus
        self._organ_states: Dict[str, dict] = {}  # Cache local de estados de orgaos

        # Registra o proprio orgao no barramento
        self._self_address = SynapticAddress(
            region=SynapticRegion.GRID,
            cluster="broadcast",
            synapse_id=self.NODE_ID,
            channel=SynapticChannel.CONTROL,
        )
        self._register_self()

    def _register_self(self):
        """Auto-registro no barramento sinaptico."""
        state = SynapticNodeState(
            address=self._self_address,
            status="active",
            telemetry={
                "started_at": int(time.time() * 1000),
                "capabilities": self.CAPABILITIES,
            },
        )
        self.router.register_node(state)
        logger.info("SynapticBroadcastOrgan registrado: %s", self._self_address.vector_path)

    # -- Registro de orgaos existentes no barramento --------------------------

    def register_existing_organ(self, organ_name: str, shm_path: Optional[str] = None):
        """Registra um orgao existente do Aeon no barramento sinaptico.

        Cria a coordenada aeon:v6:cortex:<organ_name>:state:meta e,
        se o orgao usa SHM, registra o ponteiro para zero-copy.
        """
        addr = cortex_addr(organ_name)

        shm_offset = 0
        shm_size = 0
        if shm_path and os.path.exists(shm_path):
            shm_size = os.path.getsize(shm_path)

        state = SynapticNodeState(
            address=addr,
            shm_offset=shm_offset,
            shm_size=shm_size,
            status="active",
            telemetry={"registered_at": int(time.time() * 1000)},
        )
        self.router.register_node(state)
        self._organ_states[organ_name] = state.model_dump(mode="json")
        logger.info("Orgao registrado: %s -> %s", organ_name, addr.vector_path)

    def register_all_known_organs(self):
        """Registra todos os orgaos conhecidos do Aeon no barramento."""
        known_organs = {
            "brain_mapper":    "/dev/shm/brain_map_vhd",
            "blake_gravity":   None,
            "fractal_memory":  None,
            "emergent_sense":  None,
            "eventual_state":  None,
            "context_bus":     CONTEXT_BUS_PATH,
            "ipc_bus":         None,
            "grid_registry":   "/dev/shm/grid_status.json",
            "decision_trace":  None,
            "episodic_memory": None,
            "broadcast_runtime": None,
        }
        for name, shm in known_organs.items():
            self.register_existing_organ(name, shm)

    # -- Ponte ContextBus (mmap) -> Redis ------------------------------------

    def sync_context_bus_delta(self) -> int:
        """Le novos tokens do ContextBus (mmap) e propaga para Redis.

        Retorna o numero de novas entradas propagadas.
        """
        if not os.path.exists(CONTEXT_BUS_PATH):
            return 0

        try:
            with open(CONTEXT_BUS_PATH, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    # Le o header
                    header_data = mm[:HEADER_LEN]
                    unpacked = struct.unpack(HEADER_FMT, header_data)
                    magic, version, num_agents, write_seq = unpacked[:4]

                    if magic != 0xAE0CB0C0:
                        return 0

                    # Calcula quantas entradas novas existem
                    new_count = write_seq - self._last_ctx_seq
                    if new_count <= 0:
                        return 0

                    # Propaga cada entrada nova para o Redis
                    propagated = 0
                    for seq in range(self._last_ctx_seq, write_seq):
                        idx = seq % 8192  # MAX_ENTRIES do ContextBus
                        offset = HEADER_SIZE + (idx * ENTRY_SIZE)

                        entry_data = mm[offset:offset + ENTRY_SIZE]
                        if len(entry_data) < ENTRY_SIZE:
                            break

                        # Parseia a entrada (formato do context_bus.py)
                        entry_fmt = "<IIIQHBB192s"
                        try:
                            (e_seq, agent_id, token_id, timestamp,
                             token_len, role, flags, text_bytes) = struct.unpack(entry_fmt, entry_data[:struct.calcsize(entry_fmt)])
                        except struct.error:
                            continue

                        token_text = text_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
                        if not token_text.strip():
                            continue

                        # Propaga como pulso sensorial
                        addr = sensory_addr("context_bus", f"token_{seq}")
                        self.router.pulse(addr, {
                            "seq": e_seq,
                            "agent_id": agent_id,
                            "role": role,
                            "token": token_text,
                            "timestamp": timestamp,
                        })
                        propagated += 1

                    self._last_ctx_seq = write_seq
                    return propagated

                finally:
                    mm.close()

        except Exception as e:
            logger.error("Erro ao sincronizar ContextBus: %s", e)
            return 0

    # -- Ponte SHM Tensor -> Redis (metadados apenas) ------------------------

    def bridge_shm_tensor(self, organ_name: str, tensor_label: str,
                          shm_offset: int, size: int, dtype: str = "float32"):
        """Registra um tensor SHM no barramento sem copiar dados.

        O Redis armazena apenas o ponteiro (offset/size) para zero-copy.
        O consumidor le diretamente do /dev/shm usando o ponteiro.
        """
        addr = SynapticAddress(
            region=SynapticRegion.CORTEX,
            cluster=organ_name,
            synapse_id=tensor_label,
            channel=SynapticChannel.ACTIVATION,
        )
        state = SynapticNodeState(
            address=addr,
            shm_offset=shm_offset,
            shm_size=size,
            status="active",
            telemetry={
                "dtype": dtype,
                "bridged_at": int(time.time() * 1000),
            },
        )
        self.router.register_node(state)

    # -- Ponte Episodic Memory -> Redis (cache de chunks) --------------------

    def cache_episodic_chunk(self, chunk_id: int, messages: List[dict],
                             seed: int = 0, metadata: Optional[dict] = None):
        """Persiste um chunk de memoria episodica no barramento.

        Gera vetor normalizado 2560D deterministico para busca semântica.
        """
        addr = memory_addr(chunk_id)

        # Vetor deterministico (mesma logica do dimensions.py)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(2560)
        vec_norm = (vec / np.linalg.norm(vec)).tolist()

        chunk_data = {
            "chunk_id": chunk_id,
            "vector_2560": vec_norm,
            "messages": messages,
            "meta": metadata or {},
            "created_at": int(time.time() * 1000),
        }

        state = SynapticNodeState(
            address=addr,
            status="active",
            telemetry={"message_count": len(messages)},
        )
        self.router.register_node(state)

        # Salva dados do chunk separadamente (mais eficiente para JSON.GET parcial)
        if self.router.r:
            try:
                self.router.r.json().set(addr.vector_path, "$.chunk_data", chunk_data)
            except Exception as e:
                logger.error("Erro ao cachear chunk %d: %s", chunk_id, e)

    # -- Ponte Decision Trace -> Redis ----------------------------------------

    def publish_decision_trace(self, decision_id: str, blake_sig: List[float],
                                semantic_emb: List[float], classification: str):
        """Publica um trace de decisao no barramento para replay instantaneo."""
        addr = trace_addr(decision_id)
        self.router.pulse(addr, {
            "blake_signature": blake_sig[:6],
            "semantic_embedding_384": semantic_emb[:384],
            "classification": classification,
        })

    # -- Polling loop (integra com OODA) --------------------------------------

    def pulse(self, telemetry: Optional[dict] = None):
        """Chamado pelo OODA loop do cortex/run.py a cada ciclo.

        Sincroniza delta do ContextBus e atualiza telemetria propria.
        """
        propagated = self.sync_context_bus_delta()

        # Atualiza telemetria do proprio orgao
        self.router.pulse(self._self_address, {
            "ctx_propagated": propagated,
            "last_ctx_seq": self._last_ctx_seq,
            "total_organs": len(self._organ_states),
            "redis_alive": self.router.is_alive(),
        })

    def start_background_sync(self, interval_seconds: float = 1.0):
        """Inicia sincronizacao periodica em background."""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._background_loop,
            args=(interval_seconds,),
            daemon=True,
            name="SynapticBroadcastSync",
        )
        self._poll_thread.start()
        logger.info("Background sync iniciado (intervalo=%.1fs)", interval_seconds)

    def _background_loop(self, interval: float):
        while self._running:
            try:
                self.pulse()
            except Exception as e:
                logger.error("Erro no background sync: %s", e)
            time.sleep(interval)

    def stop(self):
        """Para o orgao e desregistra do barramento."""
        self._running = False
        self.router.unregister_node(self._self_address)
        self.router.stop()
        logger.info("SynapticBroadcastOrgan parado")

    # -- Telemetria ------------------------------------------------------------

    def get_topology(self) -> dict:
        """Retorna snapshot da topologia do barramento."""
        return {
            "self": self._self_address.vector_path,
            "redis_alive": self.router.is_alive(),
            "registered_organs": list(self._organ_states.keys()),
            "last_ctx_seq": self._last_ctx_seq,
            "node_counts": self.router.topology_snapshot(),
        }

def init_organ():
    organ = SynapticBroadcastOrgan()
    organ.register_all_known_organs()
    # Inicia a thread de background para sincronizar /dev/shm -> Redis a cada 50ms
    organ.start_background_sync(interval_seconds=0.05)
    return organ
