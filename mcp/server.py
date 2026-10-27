import sys
import os
import anyio
import uvicorn
from contextlib import asynccontextmanager
import starlette.responses
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport

# Garante que o diretório pai (raiz do projeto) esteja no path para importar o SWA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swa.quantum_bridge import QuantumPhaseBridge
from main import EventualState
import torch

mcp = Server("swa-quantum-mcp")
eventual_state = EventualState()
quantum_bridge = None
sse_transport = None

@asynccontextmanager
async def lifespan(app: Starlette):
    global quantum_bridge
    print("🔌 Inicializando MCP SWA Quantum Server...")
    try:
        quantum_bridge = QuantumPhaseBridge()
        # Pre-carrega a massa semântica na CPU
        quantum_bridge.load_semantic_mass_cache(device="cpu")
        print("✅ QuantumPhaseBridge conectado e cache carregado.")
    except Exception as e:
        print(f"⚠️ Atenção: Falha na inicialização do Quantum Bridge: {e}")
    yield
    print("🛑 Encerrando MCP SWA Quantum Server...")

@mcp.tool()
async def get_spatial_vectors() -> str:
    """Returns the current spatial vectors from the SWA eventual state."""
    return f"Vetores atuais: {eventual_state.vectors}"

@mcp.tool()
async def get_obstacles() -> str:
    """Returns the current active obstacles."""
    return f"Obstáculos atuais: {eventual_state.obstacles}"

@mcp.tool()
async def add_obstacle(direction: str) -> str:
    """Adds an obstacle in the specified direction ('front', 'left', 'right')."""
    if direction not in eventual_state.obstacles:
        eventual_state.obstacles.append(direction)
        if direction == 'front':
            eventual_state.vectors['forward'] = max(0.0, eventual_state.vectors['forward'] - 0.5)
            eventual_state.vectors['stop'] += 0.3
        return f"Obstáculo adicionado na direção: {direction}"
    return f"Obstáculo já existe na direção: {direction}"

@mcp.tool()
async def update_spatial_vector(vector_name: str, strength: float) -> str:
    """Updates the strength of a spatial vector ('forward', 'left', 'right', 'reverse', 'stop')."""
    if vector_name in eventual_state.vectors:
        eventual_state.vectors[vector_name] = min(1.0, eventual_state.vectors.get(vector_name, 0.0) + strength)
        return f"Vetor espacial {vector_name} atualizado para {eventual_state.vectors[vector_name]}"
    return f"Vetor desconhecido: {vector_name}"

@mcp.tool()
async def resolve_phase_jump(vector_values: list[float]) -> str:
    """Executes the ultra-fast search in pgvector using the deformed cosine distance operator (Blake similarity)."""
    if quantum_bridge is None:
        return "Erro: QuantumPhaseBridge não está ativo. Verifique a conexão com o Postgres."
    
    try:
        # Cria um tensor 1D a partir da lista
        steer_tensor = torch.tensor(vector_values, dtype=torch.float32)
        unique_id, file_path, distance = quantum_bridge.resolve_phase_jump(steer_tensor)
        if unique_id:
            return f"Salto de Fase Resolvido! ID: {unique_id}, Arquivo: {file_path}, Distância de Blake: {distance}"
        return "Nenhuma âncora de salto de fase foi encontrada no banco."
    except Exception as e:
        return f"Erro ao resolver phase jump: {e}"

async def sse(request: Request):
    global sse_transport
    # Instancia o transporte SSE para esta conexão
    sse_transport = SseServerTransport("/messages")
    # Conecta o servidor MCP ao transporte em background
    anyio.create_task_group().start_soon(mcp.connect, sse_transport)
    # Retorna o streaming de eventos (mantém a conexão HTTP aberta)
    return await sse_transport.handle_sse(request)

async def messages(request: Request):
    global sse_transport
    if sse_transport is None:
        return starlette.responses.Response(status_code=400, content="SSE Connection not initialized")
    # Processa as requisições POST do cliente
    return await sse_transport.handle_post_message(request)

app = Starlette(
    routes=[
        Route("/sse", endpoint=sse),
        Route("/messages", endpoint=messages, methods=["POST"])
    ],
    lifespan=lifespan
)

if __name__ == "__main__":
    print("🚀 Iniciando Servidor MCP (SSE) interno em http://127.0.0.1:8000")
    print("Endpoint SSE: http://localhost:8000/sse")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
