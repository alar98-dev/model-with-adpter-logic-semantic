import json

import numpy as np
import pytest

from swa.matriz_ooda_bridge import MatrizOodaEngine, MatrizOodaBridge, SwaMatriz
from swa.tensor_protocol import TensorIntegrityError, TensorPacketError, UdpTensorProtocol


def test_udp_tensor_roundtrip_preserves_shape_values_and_metadata():
    source = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    datagram = UdpTensorProtocol.encode(
        source,
        sequence=42,
        metadata={"event_type": "mouse_click", "phase": "OBSERVE"},
    )

    packet = UdpTensorProtocol.decode(datagram)

    assert packet.sequence == 42
    assert packet.tensor.shape == (8, 8)
    assert packet.metadata["event_type"] == "mouse_click"
    np.testing.assert_array_equal(packet.tensor, source)


def test_udp_tensor_rejects_tampering():
    datagram = bytearray(UdpTensorProtocol.encode(np.ones(16, dtype=np.float32), sequence=1))
    datagram[-1] ^= 0x01

    with pytest.raises(TensorIntegrityError):
        UdpTensorProtocol.decode(datagram)


def test_udp_tensor_rejects_nan_and_oversized_payload():
    with pytest.raises(TensorPacketError):
        UdpTensorProtocol.encode(np.array([np.nan], dtype=np.float32))
    with pytest.raises(TensorPacketError):
        UdpTensorProtocol.encode(np.zeros(20_000, dtype=np.float32))


def test_udp_tensor_fragments_stay_below_mtu_and_preserve_tensor():
    source = np.linspace(-1.0, 1.0, 4096, dtype=np.float32).reshape(64, 64)
    fragments = UdpTensorProtocol.encode_fragments(
        source,
        sequence=88,
        metadata={"event_type": "tensor_inference_heavy", "phase": "OBSERVE"},
    )

    assert len(fragments) > 1
    packets = [UdpTensorProtocol.decode(fragment) for fragment in fragments]
    assert all(len(fragment) <= UdpTensorProtocol.SAFE_DATAGRAM for fragment in fragments)
    assert [packet.metadata["fragment_index"] for packet in packets] == list(range(len(packets)))
    restored = np.concatenate([packet.tensor.reshape(-1) for packet in packets]).reshape(64, 64)
    np.testing.assert_array_equal(restored, source)


def test_ooda_closes_observe_orient_decide_act_and_publishes_mascot_state(tmp_path):
    engine = MatrizOodaEngine(
        matrix=SwaMatriz(path=str(tmp_path / "missing-matrix.bin")),
        state_path=tmp_path / "state.json",
    )
    published = []
    bridge = MatrizOodaBridge(
        publisher=lambda channel, payload: published.append((channel, payload)),
        engine=engine,
    )
    datagram = UdpTensorProtocol.encode(
        np.eye(8, dtype=np.float32), sequence=7, metadata={"event_type": "voice"}
    )

    ack = json.loads(bridge.handle_datagram(datagram, ("127.0.0.1", 12345)))

    assert ack["status"] == "accepted"
    assert ack["phase"] == "ACT"
    assert ack["decision"] in engine.ALLOWED_DECISIONS
    assert bridge.accepted == 1
    assert published[0][0] == "swa:mascot:state"
    assert published[1][0] == "aeon:matrix:ooda"
    assert json.loads((tmp_path / "state.json").read_text())["mascot"]["entity"] == "aeon_cell"
