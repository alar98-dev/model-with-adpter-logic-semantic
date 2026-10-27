import pytest
import asyncio
import json
import torch
from unittest.mock import MagicMock
from aeon.core.hybrid_stream import BicameralSSEEngine
from aeon.core.manifold import IsometricProjector
from aeon.ipc.shm_tensor import SharedMetamatterIPC

def test_bicameral_sse_handshake_and_completion():
    # 1. Mocks dos modelos neurais para validação de rota e streaming
    mock_fast_model = MagicMock()
    mock_reflex_model = MagicMock()
    mock_tok = MagicMock()

    mock_tok.encode.return_value = torch.tensor([[101, 102]])
    mock_tok.decode.return_value = "quantum"
    mock_tok.eos_token_id = 999

    # Simula logits retornando um tensor compatível
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 2, 1000)
    mock_fast_model.return_value = mock_output
    mock_reflex_model.return_value = mock_output

    mock_projector = MagicMock(spec=IsometricProjector)
    mock_projector.return_value = torch.randn(2560)

    mock_ipc = MagicMock(spec=SharedMetamatterIPC)
    mock_ipc.read_latest_vector.return_value = (torch.randn(640), 1, 123456)

    engine = BicameralSSEEngine(
        model_fast=mock_fast_model,
        tokenizer_fast=mock_tok,
        model_reflexive=mock_reflex_model,
        tokenizer_reflexive=mock_tok,
        projector_640_to_2560=mock_projector,
        shm_ipc=mock_ipc,
        device="cpu",
    )

    async def run_stream():
        events_received = []
        async for chunk in engine.stream_bicameral("Teste de pulso semântico", max_new_tokens=5):
            events_received.append(chunk)
        return events_received

    events_received = asyncio.run(run_stream())

    # 2. Verificações de protocolo SSE
    assert len(events_received) > 0
    assert any("data: [DONE]" in chunk for chunk in events_received)
    
    # Valida estrutura de payload dos chunks intermediários
    first_data_chunk = [c for c in events_received if "data: {" in c][0]
    raw_json = first_data_chunk.replace("data: ", "").strip()
    payload = json.loads(raw_json)

    assert "token" in payload
    assert "chamber" in payload
    assert "entropy" in payload
    assert "z_score" in payload
