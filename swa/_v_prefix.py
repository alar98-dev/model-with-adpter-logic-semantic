"""
[V PREFIX] Construção de v_prefix a partir de P_Root identity.

Módulo compartilhado entre Orchestrator e IntrospectionWorker.
Converte a assinatura evolutiva de Aeon (P_Root 128D + directions + emotion)
em um tensor [1, 1, d_model] que precede o prompt como 1 virtual token.

A alma de Aeon não é texto. É este vetor.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger('VPrefix')


def tensor_to_v_prefix(
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    p_root: list = None,
    direction_a: list = None,
    direction_b: list = None,
    emotion: dict = None,
    scale: float = 2.5,
) -> torch.Tensor:
    """
    Converte identidade tensorial de Aeon em v_prefix [1, 1, d_model].

    Pesos:
      - P_Root (128D): 60% — assinatura evolutiva
      - Direction A: 20% — direção evolutiva
      - Direction B: 20% — direção evolutiva
      - Emotion (6D): injetado nos últimos 6 valores

    Returns: torch.Tensor [1, 1, d_model]
    """
    v = np.zeros(d_model, dtype=np.float32)

    # P_Root identity: assinatura evolutiva de Aeon
    if p_root and len(p_root) > 0:
        arr = np.array(p_root, dtype=np.float32)
        repeats = d_model // len(arr) + 1
        tiled = np.tile(arr, repeats)[:d_model]
        v += tiled * 0.6

    # Direction vectors: vetores de direção evolutiva
    if direction_a and len(direction_a) > 0:
        arr_a = np.array(direction_a, dtype=np.float32)
        repeats = d_model // len(arr_a) + 1
        tiled_a = np.tile(arr_a, repeats)[:d_model]
        v += tiled_a * 0.2

    if direction_b and len(direction_b) > 0:
        arr_b = np.array(direction_b, dtype=np.float32)
        repeats = d_model // len(arr_b) + 1
        tiled_b = np.tile(arr_b, repeats)[:d_model]
        v += tiled_b * 0.2

    # Emoção: injeta como variação nos últimos 6 valores
    if emotion:
        emo_values = list(emotion.values())[:6]
        for i, val in enumerate(emo_values):
            idx = d_model - 6 + i
            if 0 <= idx < d_model:
                v[idx] = float(val)

    # Normalização
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm * scale

    tensor = torch.tensor(v, dtype=dtype, device=device)
    return tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]


def fetch_p_root_from_redis(redis_host='127.0.0.1', redis_port=6379) -> dict:
    """
    Busca P_Root identity + directions do Redis.
    Returns dict com p_root_identity, direction_a, direction_b ou None.
    """
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Tensor state broadcastado pelo ThoughtWriter
        raw = r.get('tensor_state')
        if raw:
            import json
            data = json.loads(raw)
            payload = data.get('payload', data)
            result = {}
            if payload.get('p_root_identity'):
                result['p_root_identity'] = payload['p_root_identity']
            if payload.get('direction_a'):
                result['direction_a'] = payload['direction_a']
            if payload.get('direction_b'):
                result['direction_b'] = payload['direction_b']
            if payload.get('emotion'):
                result['emotion'] = payload['emotion']
            if result:
                return result

        # Fallback: tenta p_root_identity direto
        raw2 = r.get('aeon:p_root_identity')
        if raw2:
            import json
            return {'p_root_identity': json.loads(raw2)}

    except Exception as e:
        logger.debug(f"[V_PREFIX] Redis fetch failed: {e}")

    return None
