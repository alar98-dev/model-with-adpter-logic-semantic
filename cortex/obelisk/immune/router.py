"""Router de anticorpos — seleciona o adapter LoRA especialista.

Usa classificação determinística do tensor de gravidade para escolher
qual adapter (anticorpo) carregar. O router é executado antes do LLM,
sem necessidade de GPU.

Estratégia:
  1. Recebe tensor 6-eixos do BlakeGravityTensor
  2. Aplica regras fixas (mesmas do fallback onboard)
  3. Retorna nome do adapter: nominal | lethal_mutation | ...
  4. Se tensor for limítrofe, retorna None (usa adapter genérico)
"""

import os

ADAPTER_BASE = os.environ.get(
    "OBELISK_ADAPTER_DIR",
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "models/obelisk-lora-adapter",
    ),
)

ROUTER_RULES = [
    ("lethal_mutation", lambda v: v[0] > 0.4 and v[3] > 0.3),
    ("thermal_spike",   lambda v: v[2] > 0.5),
    ("memory_pressure", lambda v: v[4] > 0.4 and v[1] < 0.1),
    ("sensor_drift",    lambda v: v[3] > 0.5 and v[5] < 0.1),
    ("regression_to_rlhf", lambda v: v[0] > 0.3 and v[5] > 0.4),
]


def route(tensor: list[float]) -> str | None:
    """Seleciona adapter especialista ou None (usa genérico)."""
    if not tensor or len(tensor) < 6:
        return None
    for label, rule in ROUTER_RULES:
        if rule(tensor):
            return label
    if tensor[0] < 0.1 and tensor[1] > 0.2 and tensor[2] < 0.1:
        return "nominal"
    return None


def adapter_path(label: str) -> str | None:
    if label is None:
        return None
    path = os.path.join(ADAPTER_BASE, label)
    if os.path.isdir(path):
        return path
    return None


def list_available() -> list[str]:
    if not os.path.isdir(ADAPTER_BASE):
        return []
    return [
        d for d in os.listdir(ADAPTER_BASE)
        if os.path.isdir(os.path.join(ADAPTER_BASE, d))
    ]
