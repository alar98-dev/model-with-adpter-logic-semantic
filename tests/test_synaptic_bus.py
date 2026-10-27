"""
Testes do barramento sinaptico — valida enderecamento, roteamento e bridging.

Execucao: pytest tests/test_synaptic_bus.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from swa.synaptic_address import (
    SynapticAddress,
    SynapticChannel,
    SynapticNodeState,
    SynapticRegion,
    cortex_addr,
    memory_addr,
    sensory_addr,
    adapter_addr,
    trace_addr,
)
from swa.synaptic_router import SynapticBusRouter, _ttl_for_region


# -- SynapticAddress ---------------------------------------------------------


class TestSynapticAddress:
    """Testes de construcao e parsing de enderecos sinapticos."""

    def test_vector_path_format(self):
        addr = SynapticAddress(
            region=SynapticRegion.CORTEX,
            cluster="frontal",
            synapse_id="blake_gravity",
            channel=SynapticChannel.ACTIVATION,
        )
        assert addr.vector_path == "aeon:v6:cortex:frontal:blake_gravity:act"

    def test_from_vector_roundtrip(self):
        original = "aeon:v6:mem:episodic:chunk_0:stream"
        addr = SynapticAddress.from_vector(original)
        assert addr.vector_path == original
        assert addr.region == SynapticRegion.MEMORY
        assert addr.cluster == "episodic"
        assert addr.synapse_id == "chunk_0"
        assert addr.channel == SynapticChannel.STREAM

    def test_from_vector_default_channel(self):
        addr = SynapticAddress.from_vector("aeon:v6:cortex:frontal:node_1")
        assert addr.channel == SynapticChannel.ACTIVATION

    def test_from_vector_invalid_prefix(self):
        with pytest.raises(ValueError, match="invalido"):
            SynapticAddress.from_vector("invalid:v1:cortex:x:y")

    def test_from_vector_too_short(self):
        with pytest.raises(ValueError, match="invalido"):
            SynapticAddress.from_vector("aeon:v6:cortex")

    def test_colon_in_segment_rejected(self):
        with pytest.raises(ValueError, match="conter"):
            SynapticAddress(
                region=SynapticRegion.CORTEX,
                cluster="fron:tal",
                synapse_id="x",
            )

    def test_dimensional_space(self):
        assert sensory_addr("mic").dimensional_space == 640
        assert cortex_addr("blake").dimensional_space == 1152
        assert memory_addr(0).dimensional_space == 2560
        assert trace_addr("abc").dimensional_space == 384

    def test_signal_channel(self):
        addr = cortex_addr("frontal")
        assert addr.signal_channel == "aeon:v6:cortex:signals"

    def test_hash_equality(self):
        a1 = cortex_addr("frontal")
        a2 = cortex_addr("frontal")
        assert hash(a1) == hash(a2)
        assert a1.vector_path == a2.vector_path


class TestConvenienceConstructors:
    """Testa os atalhos sensory_addr, cortex_addr, etc."""

    def test_sensory_addr(self):
        addr = sensory_addr("voice", "mic_0")
        assert addr.vector_path == "aeon:v6:sens:voice:mic_0:stream"

    def test_cortex_addr(self):
        addr = cortex_addr("blake_gravity")
        assert addr.vector_path == "aeon:v6:cortex:blake_gravity:state:meta"

    def test_memory_addr(self):
        addr = memory_addr(42)
        assert addr.vector_path == "aeon:v6:mem:episodic:chunk_42:stream"

    def test_adapter_addr(self):
        addr = adapter_addr("gemma", "lora_exp_7")
        assert addr.vector_path == "aeon:v6:adapter:gemma:lora_exp_7:weights"

    def test_trace_addr(self):
        addr = trace_addr("dec_8f")
        assert addr.vector_path == "aeon:v6:trace:decision:dec_8f:trace"


# -- SynapticNodeState -------------------------------------------------------


class TestSynapticNodeState:
    """Testes do modelo de estado de no."""

    def test_default_values(self):
        addr = cortex_addr("test")
        state = SynapticNodeState(address=addr)
        assert state.status == "idle"
        assert state.weight_bias == 1.0
        assert state.shm_offset == 0
        assert state.active_adapter is None

    def test_serialization_roundtrip(self):
        addr = sensory_addr("cam", "frame_0")
        state = SynapticNodeState(
            address=addr,
            shm_offset=4096,
            shm_size=1024,
            status="firing",
            telemetry={"fps": 30},
        )
        data = state.model_dump(mode="json")
        restored = SynapticNodeState.model_validate(data)
        assert restored.address.vector_path == addr.vector_path
        assert restored.shm_offset == 4096
        assert restored.telemetry["fps"] == 30


# -- SynapticBusRouter (com Redis mockado) -----------------------------------


class TestSynapticBusRouterMocked:
    """Testes do router com Redis mockado."""

    @pytest.fixture
    def mock_redis(self):
        r = MagicMock()
        r.ping.return_value = True
        r.pipeline.return_value = MagicMock()
        r.pipeline.return_value.execute.return_value = [True, True, True]
        r.json.return_value = MagicMock()
        r.smembers.return_value = set()
        return r

    @pytest.fixture
    def router(self, mock_redis):
        return SynapticBusRouter(redis_client=mock_redis)

    def test_register_node(self, router, mock_redis):
        addr = cortex_addr("test_organ")
        state = SynapticNodeState(address=addr, status="active")

        result = router.register_node(state)
        assert result is True

        pipe = mock_redis.pipeline.return_value
        assert pipe.json.return_value.set.called
        assert pipe.expire.called
        assert pipe.sadd.called
        assert pipe.execute.called

    def test_pulse(self, router, mock_redis):
        addr = sensory_addr("voice")
        result = router.pulse(addr, {"amplitude": 0.8})
        assert result is True

    def test_is_alive(self, router, mock_redis):
        assert router.is_alive() is True
        mock_redis.ping.side_effect = ConnectionError
        assert router.is_alive() is False

    def test_topology_snapshot(self, router, mock_redis):
        mock_redis.smembers.return_value = {"node_1", "node_2"}
        snap = router.topology_snapshot()
        assert isinstance(snap, dict)
        assert "cortex" in snap

    def test_ttl_for_region(self):
        assert _ttl_for_region(SynapticRegion.SENSORY) == 300
        assert _ttl_for_region(SynapticRegion.MEMORY) == 86400 * 30
        assert _ttl_for_region(SynapticRegion.CORTEX) == 86400

    def test_no_redis_graceful_degradation(self):
        router = SynapticBusRouter(redis_client=None)
        router.r = None  # Simula Redis indisponivel
        assert router.is_alive() is False
        assert router.get_active_nodes() == []
        assert router.register_node(SynapticNodeState(address=cortex_addr("x"))) is False
        assert router.pulse(cortex_addr("x"), {}) is False
