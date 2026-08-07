import asyncio
from fastapi import WebSocket

class SystemicEventBroadcaster:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📡 [WS] Novo observador conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"📡 [WS] Observador desconectado. Total: {len(self.active_connections)}")

    async def broadcast(self, event_type: str, data: dict):
        message = {"type": event_type, "data": data}
        if not self.active_connections:
            return
        await asyncio.gather(
            *[conn.send_json(message) for conn in self.active_connections],
            return_exceptions=True
        )

broadcaster = SystemicEventBroadcaster()
