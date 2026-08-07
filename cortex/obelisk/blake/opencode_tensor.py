"""Subespaço Blake — projeta padrões de conversa OpenCode no tensor de gravidade.

Cada sessão do OpenCode gera:
  - decision_chain: USER → [REASONING, TOOL, ...] → TEXT
  - semantic_geometry: centroid[6], span[6], type_distribution
  - part_signatures: blake[6] por parte

Esse subespaço projeta essas métricas nos 6 eixos do BlakeGravityTensor,
permitindo que o obelisco classifique o comportamento do agente como
mais uma dimensão do espaço de gravidade.
"""

import os
import json
import time
import hashlib
from typing import Optional
from collections import Counter

TOKEN_DATASET_DIR = os.environ.get(
    "TOKEN_DATASET_DIR",
    "/services/model-with-adpter-logic-semantic/models/token_dataset"
)


def _blake3(text: str) -> str:
    try:
        import blake3
        return blake3.blake3(text.encode()).hexdigest()
    except ImportError:
        return hashlib.sha256(text.encode()).hexdigest()


def load_latest_geometry(max_sessions: int = 5) -> list[dict]:
    """Carrega as últimas geometrias do dataset de tokens."""
    path = os.path.join(TOKEN_DATASET_DIR, "geometry.jsonl")
    if not os.path.exists(path):
        return []
    geos = []
    with open(path) as f:
        for line in f:
            try:
                geos.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return geos[-max_sessions:]


def load_latest_parts(max_parts: int = 500) -> list[dict]:
    """Carrega as últimas parts para análise de distribuição."""
    path = os.path.join(TOKEN_DATASET_DIR, "parts.jsonl")
    if not os.path.exists(path):
        return []
    parts = []
    with open(path) as f:
        for line in f:
            try:
                parts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return parts[-max_parts:]


def load_latest_sessions(max_sessions: int = 5) -> list[dict]:
    """Carrega as últimas sessões completas."""
    path = os.path.join(TOKEN_DATASET_DIR, "opencode_sessions.jsonl")
    if not os.path.exists(path):
        return []
    sessions = []
    with open(path) as f:
        for line in f:
            try:
                sessions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return sessions[-max_sessions:]


def compute_opencode_tensor(
    sessions: Optional[list] = None,
    parts: Optional[list] = None,
    geometry: Optional[list] = None,
) -> list[float]:
    """Projeta padrões OpenCode nos 6 eixos do Blake Gravity Tensor.

    Eixos:
      ex0: entropia de tipos de parte (quão diverso é o comportamento)
      ex1: profundidade média das cadeias de decisão
      ex2: taxa de tool_call vs reasoning (viés de ação vs pensamento)
      ex3: coerência temporal (consistência de customer_text → assistant_text)
      ex4: intensidade de tráfego (partes por sessão normalizado)
      ex5: assinatura gravitacional (hash BLAKE3 da geometria)
    """
    if sessions is None:
        sessions = load_latest_sessions(3)
    if parts is None:
        parts = load_latest_parts(500)
    if geometry is None:
        geometry = load_latest_geometry(3)

    if not parts and not sessions:
        return [0.0] * 6

    # ex0: entropia de tipos de parte
    type_counts = Counter(p.get("part_type", "?") for p in parts)
    type_total = sum(type_counts.values()) or 1
    import math
    entropy = 0.0
    for t, c in type_counts.items():
        p = c / type_total
        if p > 0:
            entropy += -p * math.log2(p)
    ex0 = max(0.0, min(entropy / 3.0, 1.0))

    # ex1: profundidade média das cadeias de decisão
    chain_counts = [s.get("decision_chain", {}).get("chain_count", 0) for s in sessions]
    avg_chain_depth = sum(chain_counts) / max(len(chain_counts), 1)
    chain_steps = []
    for s in sessions:
        for c in s.get("decision_chain", {}).get("chains", []):
            chain_steps.append(len(c.get("steps", 0)))
    avg_steps = sum(chain_steps) / max(len(chain_steps), 1)
    ex1 = min((avg_chain_depth * avg_steps) / 100.0, 1.0)

    # ex2: tool_call vs reasoning ratio
    tool_count = type_counts.get("tool_call", 0)
    reasoning_count = type_counts.get("reasoning", 0)
    total_actions = tool_count + reasoning_count
    if total_actions > 0:
        ex2 = tool_count / max(total_actions, 1)
    else:
        ex2 = 0.5

    # ex3: coerência — razão de respostas por prompt
    user_texts = type_counts.get("user_text", 0)
    assistant_texts = type_counts.get("assistant_text", 0)
    if user_texts > 0:
        ex3 = min(assistant_texts / max(user_texts, 1) / 2.0, 1.0)
    else:
        ex3 = 0.5

    # ex4: intensidade de tráfego
    parts_per_session = type_total / max(len(sessions), 1)
    ex4 = min(parts_per_session / 200.0, 1.0)

    # ex5: assinatura gravitacional
    geo_hash_input = json.dumps([g.get("centroid", []) for g in geometry], sort_keys=True)
    geo_hash = _blake3(geo_hash_input)
    ex5 = int(geo_hash[:2], 16) / 255.0

    vec = [ex0, ex1, ex2, ex3, ex4, ex5]
    magnitude = sum(abs(v) for v in vec)
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def opencode_space_signature(sessions: Optional[list] = None) -> float:
    """Score de anomalia do espaço OpenCode — quão distorcida está a conversa."""
    if sessions is None:
        sessions = load_latest_sessions(3)
    if not sessions:
        return 0.0
    score = 0.0
    for s in sessions:
        geom = s.get("semantic_geometry", {})
        span = geom.get("span", [0.0] * 6)
        span_mag = sum(abs(v) for v in span)
        score += span_mag
        if geom.get("classification") in ("tool_execution", "decision_point"):
            score += 2.0
        if s.get("tokens_input", 0) > 100000:
            score += 1.0
        if s.get("tokens_output", 0) > 50000:
            score += 0.5
    return score / max(len(sessions), 1)


def _auto_persist(sessions: list, parts: list, geometry: list):
    """Persiste sessions + parts no PostgreSQL decision_traces."""
    try:
        from swa.decision_trace_db import persist_part_trace, persist_geometry, flush
    except ImportError:
        return
    for s in sessions:
        sid = s.get("session_id", "")
        geom = s.get("semantic_geometry", {})
        persist_geometry(
            session_id=sid,
            classification=geom.get("classification", "unknown"),
            centroid=geom.get("centroid", [0.0]*6),
            span=geom.get("span", [0.0]*6),
            type_distribution=geom.get("type_distribution", {}),
            part_count=geom.get("part_count", 0),
            total_tokens=geom.get("total_tokens", 0),
        )
    for p in parts:
        persist_part_trace(
            session_id=p.get("session_id", ""),
            part_id=p.get("part_id", ""),
            classification=p.get("trace_classification", "unknown"),
            part_type=p.get("part_type", ""),
            blake_signature=p.get("blake_signature", [0.0]*6),
            text_snippet=p.get("text", ""),
            token_count=p.get("tokens_total", 0),
            tool_name=p.get("tool_name", ""),
            finish_reason=p.get("finish_reason", ""),
            agent="opencode",
        )
    flush()


def collect_opencode_space() -> dict:
    """Coleta completa do subespaço OpenCode para o Blake Gravity Tensor.
    
    Tenta buscar do PG primeiro; se não houver dados, carrega JSONL.
    """
    tensor = None
    sessions = []
    parts = []
    geometry = []
    try:
        from swa.decision_trace_db import search_similar, get_latest_traces, count_by_classification
        counts = count_by_classification()
        if sum(counts.values()) > 10:
            latest = get_latest_traces(limit=20)
            if latest:
                def _parse_vec(v):
                    if isinstance(v, str):
                        return [float(x) for x in v.strip("[]").split(",")]
                    return v or [0.0]*6
                sigs = [_parse_vec(t.get("blake_signature")) for t in latest if t.get("blake_signature")]
                if sigs:
                    centroid = [sum(e)/len(sigs) for e in zip(*sigs)]
                sessions = []
                parts_real = load_latest_parts(500)
                parts = parts_real if parts_real else []
                geometry = load_latest_geometry(3)
                tensor = compute_opencode_tensor(
                    sessions=load_latest_sessions(3),
                    parts=parts,
                    geometry=geometry,
                )
    except ImportError:
        pass
    if tensor is None:
        sessions = load_latest_sessions(3)
        parts = load_latest_parts(500)
        geometry = load_latest_geometry(3)
        tensor = compute_opencode_tensor(sessions, parts, geometry)
    _auto_persist(sessions, parts, geometry)
    signature = opencode_space_signature(sessions or load_latest_sessions(3))
    return {
        "gravity_tensor": tensor,
        "space_signature": signature,
        "session_count": len(sessions or load_latest_sessions(3)),
        "part_count": len(parts or load_latest_parts(500)),
        "has_data": True,
    }


def search_traces(tensor: list[float], k: int = 5, classification: str = None) -> list[dict]:
    """Busca traces similares por tensor Blake no PostgreSQL."""
    try:
        from swa.decision_trace_db import search_similar
        return search_similar(tensor, k=k, classification=classification)
    except ImportError:
        return []


def get_trace_counts() -> dict:
    """Distribuição de classificações no banco."""
    try:
        from swa.decision_trace_db import count_by_classification
        return count_by_classification()
    except ImportError:
        return {}
