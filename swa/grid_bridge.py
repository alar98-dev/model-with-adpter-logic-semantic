import redis
import json
import socket
import numpy as np
import sys
import os
from swa.paths import PROJECT_ROOT

# Adiciona o path do projeto para os imports funcionarem
sys.path.insert(0, str(PROJECT_ROOT))
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame

class InterdimensionalBridge:
    def __init__(self, node_id="swa-grid-bridge"):
        self.node_id = node_id
        self.r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
        self.pubsub = self.r.pubsub()
        self.pubsub.subscribe("grid_events")
        print("🌌 [Bridge] Canal interdimensional estabelecido.")

    def run(self):
        """Ouve o Redis Global e transmite para o IPC Local."""
        for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    self.forward_to_ipc(data)
                except Exception as e:
                    print(f"⚠️ [Bridge] Erro na traducao: {e}")

    def forward_to_ipc(self, data):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            # Transforma em Frame Binario para a Grid
            topology = np.zeros((64, 64), dtype=np.float32).flatten()
            payload = json.dumps(data).encode("utf-8")
            sock.sendall(pack_aeon_frame(topology, payload))
            sock.close()
        except Exception as e:
            print(f"⚠️ [Bridge] Falha no forward IPC: {e}")

if __name__ == "__main__":
    bridge = InterdimensionalBridge()
    bridge.run()
