import socket
import json
import numpy as np
import torch
import torch.nn as nn
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame, RedisSynapse

class SensoryTranslator(nn.Module):
    """
    AEON Sensory Translator:
    Ponte interna que converte telemetria bruta e pulsos de entropia em embeddings latentes.
    Usa Redis como ponte interna tensorial para injeção de estímulos.
    """
    def __init__(self, node_id="swa-sensory-translator", embedding_dim=2560):
        super().__init__()
        self.node_id = node_id
        self.embedding_dim = embedding_dim
        self.register_node()
        
        # [SWA REDIS SYNAPSE] Conexão com a malha tensorial
        try:
            self.synapse = RedisSynapse()
        except Exception:
            self.synapse = None

    def register_node(self):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            registration_data = {"type": "REGISTER", "node_id": self.node_id, "capabilities": ["data_translation", "sensory_injection"]}
            topology = np.zeros((64, 64), dtype=np.float32).flatten()
            sock.sendall(pack_aeon_frame(topology, json.dumps(registration_data).encode("utf-8")))
            sock.close()
        except Exception:
            pass

    def get_sensory_embedding(self, device):
        """
        Consulta a malha Redis para verificar estímulos de 'física pura'.
        """
        if self.synapse is not None:
            try:
                # Exemplo: Verifica uma chave específica de estímulo direto (raw_stimulus)
                stim = self.synapse.r.get("swa_sensory_stimulus")
                if stim:
                    data = json.loads(stim)
                    # Se houver um vetor, converte para o device do modelo
                    # [SWA DTYPE ALIGNMENT] Force to float32 or bfloat16
                    emb = torch.tensor(data["vector"], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    return emb, True
            except Exception:
                pass
        
        return None, False

    def emit_event(self, event_type, payload_data):
        # Envia via UDS (Bus Clássico)
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            payload_data["type"] = event_type
            topology = payload_data.get("topology", np.zeros((64, 64), dtype=np.float32).flatten())
            payload = json.dumps(payload_data).encode("utf-8")
            sock.sendall(pack_aeon_frame(topology, payload))
            sock.close()
        except Exception:
            pass
        
        # E também via Redis (Ponte Tensorial)
        if self.synapse:
            try:
                self.synapse.broadcast_event(event_type, payload_data)
            except Exception:
                pass

def init_organ():
    return SensoryTranslator()
