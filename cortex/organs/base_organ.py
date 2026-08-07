import socket
import json
import numpy as np
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame

class AeonOrgan:
    def __init__(self, node_id: str, capabilities: list[str] | None = None):
        self.node_id = node_id
        self.capabilities = capabilities or []
        self.register_node()

    def register_node(self):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            registration_data = {
                "type": "REGISTER",
                "node_id": self.node_id,
                "capabilities": self.capabilities,
            }
            topology = np.zeros((64, 64), dtype=np.float32).flatten()
            sock.sendall(pack_aeon_frame(topology, json.dumps(registration_data).encode("utf-8")))
            sock.close()
        except Exception as e:
            print(f"\u26a0\ufe0f [{self.node_id}] Falha no registro: {e}")

    def emit_event(self, event_type, payload_data):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            payload_data["type"] = event_type
            topology = payload_data.get("topology", np.zeros((64, 64), dtype=np.float32).flatten())
            payload = json.dumps(payload_data).encode("utf-8")
            sock.sendall(pack_aeon_frame(topology, payload))
            sock.close()
        except Exception as e:
            print(f"\u26a0\ufe0f [{self.node_id}] Falha no envio IPC: {e}")

    def pulse(self, telemetry: dict | None = None):
        pass
