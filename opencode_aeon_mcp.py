"""MCP read-only do OpenCode para observabilidade do AEON.

O OpenCode conversa com o gateway HTTP. Este processo não acessa Redis,
PostgreSQL, arquivos de SHM nem o filesystem físico do AEON diretamente.
"""

import json
import urllib.parse
import urllib.request

from mcp.server.mcpserver import MCPServer


GATEWAY_URL = "http://127.0.0.1:8050"
mcp = MCPServer("AEON Observability")


def get_json(path: str) -> dict:
    with urllib.request.urlopen(GATEWAY_URL + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def render(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@mcp.tool()
def aeon_health() -> str:
    """Retorna o contrato operacional e a saúde do gateway AEON."""
    try:
        return render(get_json("/api/v1/health"))
    except Exception as exc:
        return f"AEON indisponível: {exc}"


@mcp.tool()
def aeon_chat_state() -> str:
    """Consulta o estado corrente do chat e da memória episódica."""
    try:
        return render(get_json("/api/v1/chat/state"))
    except Exception as exc:
        return f"Estado AEON indisponível: {exc}"


@mcp.tool()
def aeon_chat_events(turn_id: str) -> str:
    """Consulta os eventos auditáveis de um turno específico."""
    if not turn_id.strip() or "/" in turn_id or ".." in turn_id:
        return "turn_id inválido."
    try:
        return render(get_json("/api/v1/chat/events/" + urllib.parse.quote(turn_id, safe="")))
    except Exception as exc:
        return f"Eventos indisponíveis: {exc}"


@mcp.tool()
def aeon_training_export(turn_id: str) -> str:
    """Consulta o exemplo de dataset associado a um turno; não inicia treinamento."""
    if not turn_id.strip() or "/" in turn_id or ".." in turn_id:
        return "turn_id inválido."
    try:
        return render(get_json("/api/v1/chat/training/" + urllib.parse.quote(turn_id, safe="")))
    except Exception as exc:
        return f"Exportação indisponível: {exc}"


@mcp.tool()
def aeon_wave_archive(turn_id: str, wave_index: int) -> str:
    """Lê uma onda durável específica do arquivo auditável do AEON."""
    if not turn_id.strip() or "/" in turn_id or ".." in turn_id or wave_index < 0:
        return "Parâmetros inválidos."
    try:
        path = f"/api/v1/chat/archive/{urllib.parse.quote(turn_id, safe='')}/{wave_index}"
        return render(get_json(path))
    except Exception as exc:
        return f"Onda indisponível: {exc}"


if __name__ == "__main__":
    mcp.run()
