import socket
import json
import numpy as np
import time
import os
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame

class ObeliskDevourerOrgan:
    def __init__(self, node_id="swa-obelisk-devourer"):
        self.node_id = node_id
        self.register_node()

    def register_node(self):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            registration_data = {"type": "REGISTER", "node_id": self.node_id, "capabilities": ["system_watchdog"]}
            topology = np.zeros((64, 64), dtype=np.float32).flatten()
            sock.sendall(pack_aeon_frame(topology, json.dumps(registration_data).encode("utf-8")))
            sock.close()
        except Exception as e:
            print(f"⚠️ [" + self.node_id + "] Falha no registro: " + str(e))

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
            print(f"⚠️ [" + self.node_id + "] Falha no envio IPC: " + str(e))

    def on_event(self, event_type, payload):
        if event_type == "opencode_context_update":
            print(f"[{self.node_id}] Digerindo contexto OpenCode: Sessão {payload.get('session_id')}")
            
            # --- INTEGRAÇÃO SEMÂNTICA (A PONTE DE CONHECIMENTO) ---
            session_title = payload.get('title', 'Sem Título')
            messages = payload.get('messages', [])
            
            # Formata o texto real da sessão
            knowledge_text = f"Sessão OpenCode: {session_title}\n"
            for msg in messages:
                # Extraindo o texto de dentro do JSON interno do OpenCode
                try:
                    data = json.loads(msg.get("data", "{}"))
                    text = data.get("content") or data.get("text") or str(data)
                except:
                    text = str(msg.get("data", ""))
                
                # Ignora metadados puros
                if len(text) > 5:
                    knowledge_text += f"- {text}\n"
            
            # Injeta diretamente no banco da Memória Fractal do Córtex
            db_path = "/home/alifer/swa-aeon/swa_kernel_knowledge_fractal.json"
            try:
                kb = {}
                if os.path.exists(db_path):
                    with open(db_path, "r", encoding="utf-8") as f:
                        kb = json.load(f)
                
                session_id = payload.get("session_id", "default")
                kb[session_id] = {
                    "tipo": "opencode_session",
                    "text": knowledge_text,
                    "timestamp": time.time()
                }
                
                with open(db_path, "w", encoding="utf-8") as f:
                    json.dump(kb, f, indent=2, ensure_ascii=False)
                    
                print(f"[{self.node_id}] Sucesso: Texto da sessão injetado na Memória Fractal Semântica.")
            except Exception as e:
                print(f"[{self.node_id}] Erro ao salvar na memória fractal: {e}")
            
            # --- INTEGRAÇÃO OODA (TOKEN DE SÍNTESE ESTRUTURAL MÍNIMA) ---
            try:
                import sys
                if '/home/alifer/swa-aeon' not in sys.path:
                    sys.path.insert(0, '/home/alifer/swa-aeon')
                from swa.context_bus import ContextBus, Role
                
                # Gera hash_val caso não exista no escopo
                hash_val = hash(knowledge_text)
                
                synthesis_token = f"<|opencode_state|> session={session_id} state_hash={hash_val}"
                
                bus = ContextBus()
                bus.register_agent("opencode_devourer")
                bus.write_token(synthesis_token, role=Role.SYSTEM)
                print(f"[{self.node_id}] Token de síntese injetado no Context Bus (OODA Primordial).")
            except Exception as e:
                print(f"[{self.node_id}] Erro ao injetar no ContextBus: {e}")
                
            # --- INTEGRAÇÃO ESTRUTURAL/TENSORIAL (OODA) ---

            # Cria a matriz topológica de alerta para o núcleo (OODA Loop acorda)
            hash_val = hash(knowledge_text)
            np.random.seed(abs(hash_val) % (2**32))
            topology = np.random.randn(64, 64).astype(np.float32)
            
            self.emit_event("fractal_memory_injection", {
                "source": "opencode",
                "topology": topology.flatten().tolist()
            })

def init_organ():
    return ObeliskDevourerOrgan()
