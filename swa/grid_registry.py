import json
import threading
import struct
from typing import Dict, Any

class GridRegistry:
    STATUS_FILE = "/dev/shm/grid_status.json"
    
    def __init__(self):
        self._nodes = {}
        self._lock = threading.Lock()

    def _update_shm_status(self):
        try:
            with open(self.STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._nodes, f)
        except Exception as e:
            print(f"⚠️ [GRID REGISTRY] Falha ao atualizar SHM: {e}")

    def process_binary_registration(self, topology, payload):
        try:
            data = json.loads(payload.decode("utf-8"))
            if data.get("type") == "REGISTER":
                self.register_node(data["node_id"], data)
        except json.JSONDecodeError:
            # Não é um payload de registro (pode ser texto humano)
            pass
        except Exception as e:
            print(f"⚠️ [GRID REGISTRY] Erro no processamento: {e}")

    def register_node(self, node_id, metadata):
        with self._lock:
            self._nodes[node_id] = metadata
            self._update_shm_status()
            print(f"📡 [GRID REGISTRY] Origem neural registrada: {node_id}")

    def unregister_node(self, node_id):
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                self._update_shm_status()
                print(f"🔌 [GRID REGISTRY] Origem neural removida: {node_id}")

    def get_nodes(self):
        with self._lock:
            return self._nodes.copy()
