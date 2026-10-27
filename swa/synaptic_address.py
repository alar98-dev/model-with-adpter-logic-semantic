"""
Synaptic Address — Enderecamento dimensional IPv6 para o barramento cognitivo.

Schema:  aeon:v6:<cortex_region>:<cluster_id>:<synapse_id>:<channel>

Cada vetor localiza um no cognitivo, seu estado de ativacao,
tensores de peso e buffers de memoria compartilhada (/dev/shm).

Dimensoes canonicas mapeadas:
  640D  (Sensorial)  -> aeon:v6:sens:<id>:stream
  1152D (Consciente) -> aeon:v6:cortex:<organ>:meta
  2560D (Memoria)    -> aeon:v6:mem:episodic:chunk:<seq>

Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class SynapticRegion(str, Enum):
    """Lobulos / regioes do cortex sinaptico."""
    CORTEX   = "cortex"    # Decisao, orquestracao, BlakeGravity
    MEMORY   = "mem"       # Episodica, fractal, chunks
    SENSORY  = "sens"      # Voice stream, input tokens, ContextBus
    ADAPTER  = "adapter"   # LoRA, SWA prefix, compiled adapters
    TRACE    = "trace"     # Decision trace, blake signatures
    IDENTITY = "identity"  # Face encodings, user profiles
    GRID     = "grid"      # Node registry, heartbeat


class SynapticChannel(str, Enum):
    """Canal de comunicacao dentro de um no sinaptico."""
    ACTIVATION = "act"      # Pulso de ativacao (forward pass)
    WEIGHTS    = "weights"  # Pesos / adapters carregados
    TRACE      = "trace"    # Log de decisao / telemetria
    CONTROL    = "ctrl"     # Sinais de controle (start/stop/reset)
    STREAM     = "stream"   # Dados continuos (audio, tokens)
    META       = "meta"     # Metadados estruturais


class SynapticAddress(BaseModel):
    """Endereco vetorial IPv6-like para roteamento sinaptico O(1).

    Formato: aeon:v6:<region>:<cluster>:<synapse_id>:<channel>

    Exemplos:
        aeon:v6:cortex:frontal:blake_gravity:act
        aeon:v6:mem:episodic:chunk_0:stream
        aeon:v6:sens:voice:stream_main:stream
        aeon:v6:adapter:gemma:lora_0:weights
        aeon:v6:trace:decision:trace_8f:trace
    """

    prefix: str = Field(default="aeon:v6", frozen=True)
    region: SynapticRegion
    cluster: str
    synapse_id: str
    channel: SynapticChannel = SynapticChannel.ACTIVATION

    @field_validator("cluster", "synapse_id")
    @classmethod
    def _no_colons(cls, v: str) -> str:
        if ":" in v:
            raise ValueError(f"Segmento nao pode conter ':': {v}")
        return v

    @property
    def vector_path(self) -> str:
        """Resolve o caminho completo do vetor sinaptico."""
        return f"{self.prefix}:{self.region.value}:{self.cluster}:{self.synapse_id}:{self.channel.value}"

    @property
    def signal_channel(self) -> str:
        """Canal Pub/Sub para sinais desta regiao."""
        return f"{self.prefix}:{self.region.value}:signals"

    @property
    def dimensional_space(self) -> int:
        """Retorna a dimensao canonica associada a regiao."""
        _REGION_DIM = {
            SynapticRegion.SENSORY:  640,
            SynapticRegion.CORTEX:   1152,
            SynapticRegion.ADAPTER:  1152,
            SynapticRegion.MEMORY:   2560,
            SynapticRegion.TRACE:    384,
            SynapticRegion.IDENTITY: 128,
            SynapticRegion.GRID:     0,
        }
        return _REGION_DIM.get(self.region, 0)

    @classmethod
    def from_vector(cls, vector: str) -> SynapticAddress:
        """Parseia um vetor sinaptico de volta para SynapticAddress."""
        parts = vector.split(":")
        if len(parts) < 5 or f"{parts[0]}:{parts[1]}" != "aeon:v6":
            raise ValueError(f"Vetor sinaptico invalido: {vector}")
        return cls(
            region=SynapticRegion(parts[2]),
            cluster=parts[3],
            synapse_id=parts[4],
            channel=SynapticChannel(parts[5]) if len(parts) > 5 else SynapticChannel.ACTIVATION,
        )

    def __str__(self) -> str:
        return self.vector_path

    def __hash__(self) -> int:
        return hash(self.vector_path)


class SynapticNodeState(BaseModel):
    """Estado completo de um no sinaptico no barramento."""

    address: SynapticAddress
    shm_offset: int = 0         # Offset em /dev/shm/aeon para zero-copy
    shm_size: int = 0           # Tamanho do buffer mapeado
    weight_bias: float = 1.0    # Modulacao gravitacional do no
    active_adapter: Optional[str] = None  # LoRA / SWA adapter ativo
    status: str = "idle"        # idle | active | firing | pruned
    telemetry: Dict[str, Any] = Field(default_factory=dict)

    @property
    def vector_path(self) -> str:
        return self.address.vector_path


# -- Construtores de conveniencia -----------------------------------------------


def sensory_addr(sensor_id: str, synapse: str = "input") -> SynapticAddress:
    """Atalho: aeon:v6:sens:<sensor_id>:<synapse>:stream"""
    return SynapticAddress(
        region=SynapticRegion.SENSORY,
        cluster=sensor_id,
        synapse_id=synapse,
        channel=SynapticChannel.STREAM,
    )


def cortex_addr(organ: str, synapse: str = "state") -> SynapticAddress:
    """Atalho: aeon:v6:cortex:<organ>:<synapse>:meta"""
    return SynapticAddress(
        region=SynapticRegion.CORTEX,
        cluster=organ,
        synapse_id=synapse,
        channel=SynapticChannel.META,
    )


def memory_addr(chunk_id: int) -> SynapticAddress:
    """Atalho: aeon:v6:mem:episodic:chunk_<N>:stream"""
    return SynapticAddress(
        region=SynapticRegion.MEMORY,
        cluster="episodic",
        synapse_id=f"chunk_{chunk_id}",
        channel=SynapticChannel.STREAM,
    )


def adapter_addr(model: str, adapter_name: str) -> SynapticAddress:
    """Atalho: aeon:v6:adapter:<model>:<adapter>:weights"""
    return SynapticAddress(
        region=SynapticRegion.ADAPTER,
        cluster=model,
        synapse_id=adapter_name,
        channel=SynapticChannel.WEIGHTS,
    )


def trace_addr(decision_id: str) -> SynapticAddress:
    """Atalho: aeon:v6:trace:decision:<id>:trace"""
    return SynapticAddress(
        region=SynapticRegion.TRACE,
        cluster="decision",
        synapse_id=decision_id,
        channel=SynapticChannel.TRACE,
    )
