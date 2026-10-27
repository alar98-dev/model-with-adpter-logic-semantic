"""
Synaptic API Gateway — Ponte Async WebSocket para o Barramento Dimensional.

Permite que clientes externos (UI, CRM, Dashboard) assinem canais 
dimensionais O(1) diretamente pelo Redis, sem tocar no SHM.

Exemplos de conexão:
  ws://localhost:8100/ws/synapse/cortex
  ws://localhost:8100/ws/synapse/sens

Authorship: ALifer Ramos & Antigravity
"""
import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from swa.synaptic_address import SynapticRegion

logger = logging.getLogger("SynapticGateway")

app = FastAPI(title="Aeon Synaptic Gateway", version="v6.0")

# Pool de conexões ativas por região
class SynapticConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            region.value: set() for region in SynapticRegion
        }
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.listening_task = None

    async def connect(self, websocket: WebSocket, region: str):
        await websocket.accept()
        if region in self.active_connections:
            self.active_connections[region].add(websocket)
            logger.info(f"Cliente conectado à região sináptica: {region}")

            # Inicia o background listener do Redis se for a primeira conexão
            if self.listening_task is None:
                self.listening_task = asyncio.create_task(self._listen_to_redis())
                
            # Garante que o gateway está escutando este canal específico
            await self.pubsub.subscribe(f"aeon:v6:{region}:signals")

    def disconnect(self, websocket: WebSocket, region: str):
        if region in self.active_connections and websocket in self.active_connections[region]:
            self.active_connections[region].remove(websocket)

    async def _listen_to_redis(self):
        """Loop assíncrono consumindo o Pub/Sub do Redis O(1)"""
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    # Extrai a região do canal (ex: aeon:v6:cortex:signals -> cortex)
                    region = channel.split(':')[2] 
                    
                    if region in self.active_connections:
                        # Faz broadcast (Fan-Out) para todos os WebSockets daquela região
                        dead_sockets = set()
                        for ws in self.active_connections[region]:
                            try:
                                await ws.send_text(message['data'])
                            except Exception:
                                dead_sockets.add(ws)
                                
                        # Limpa conexões caídas
                        for ws in dead_sockets:
                            self.active_connections[region].remove(ws)
        except Exception as e:
            logger.error(f"Erro fatal no listener do Redis: {e}")

manager = SynapticConnectionManager()

@app.websocket("/ws/synapse/{region}")
async def websocket_synaptic_endpoint(websocket: WebSocket, region: str):
    """
    Endpoint WebSocket para conectar diretamente à malha vetorial do Aeon.
    Regiões válidas: cortex, sens, mem, adapter, trace.
    """
    # Valida se a região existe no Schema
    valid_regions = [r.value for r in SynapticRegion]
    if region not in valid_regions:
        await websocket.close(code=1008, reason=f"Região sináptica inválida. Use: {valid_regions}")
        return

    await manager.connect(websocket, region)
    try:
        while True:
            # Mantém a conexão viva e permite que o cliente envie sinais de controle (se necessário)
            data = await websocket.receive_text()
            # Por enquanto, comandos do cliente são apenas para log
            logger.debug(f"Sinal recebido da UI na região {region}: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, region)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
