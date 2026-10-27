import os
import re
import hashlib
import threading
import logging
from typing import Optional

from swa.synaptic_address import SynapticRegion, sensory_addr, cortex_addr, SynapticNodeState
from swa.synaptic_router import SynapticBusRouter
from swa.context_bus import ContextBus, Role
from swa.blake_tools import get_physical_signature

logger = logging.getLogger("FSSense")

class FileSystemSenseOrgan:
    def __init__(self, router: Optional[SynapticBusRouter] = None):
        self.router = router or SynapticBusRouter()
        self.name = "fs_sense"
        
        self.context_bus = ContextBus()
        self.context_bus.register_agent(self.name)
        
        self.router.subscribe(SynapticRegion.CORTEX, self._on_cortex_thought)
        self._subscribe_synaptic_edits()
        
        logger.info(f"[{self.name}] Sentido Espacial e de Arquivos (FS) ativado com Ingestão BlakeGravity em tempo real.")

    def _subscribe_synaptic_edits(self):
        def on_synaptic_pulse(channel: str, payload: dict):
            event = payload.get("event") or payload.get("type")
            file_path = payload.get("file_path")
            agent = payload.get("agent", "remoto")
            if event in ("file_modified", "file_edit") and file_path:
                logger.info(f"[{self.name}] Mudança física detectada via pulso do agente '{agent}' em: {file_path}")
                self._read_and_collapse_to_vector(file_path)

        try:
            self.router.subscribe(SynapticRegion.CORTEX, on_synaptic_pulse)
        except Exception as e:
            logger.error(f"[{self.name}] Erro ao subscrever sinais de edição: {e}")

    def _on_cortex_thought(self, vector_path: str, payload: dict):
        texto = payload.get("pensamento", "") or payload.get("token", "")
        
        explore_match = re.search(r'\[EXPLORAR:\s*([^\]]+)\]', texto, re.IGNORECASE)
        if explore_match:
            self._explore_classic(explore_match.group(1).strip())

        read_match = re.search(r'\[LER_ARQUIVO:\s*([^\]]+)\]', texto, re.IGNORECASE)
        if read_match:
            self._read_and_collapse_to_vector(read_match.group(1).strip())

    def _inject_into_ooda(self, text: str):
        try:
            self.context_bus.write_token(text, role=Role.SYSTEM)
            logger.info(f"[{self.name}] Reflexo FS injetado no ContextBus.")
        except Exception as e:
            logger.error(f"[{self.name}] Erro ao injetar reflexo FS: {e}")

    def _explore_classic(self, path: str):
        logger.info(f"[{self.name}] Explorando: {path}")
        try:
            if not os.path.exists(path):
                self._inject_into_ooda(f"[SENSOR FS: O caminho {path} nao existe no plano fisico]")
                return
                
            if not os.path.isdir(path):
                self._inject_into_ooda(f"[SENSOR FS: {path} nao e um dominio (diretorio)]")
                return

            items = os.listdir(path)
            dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            
            tree_text = f"--- Dominio: {path} ---\nPastas: {', '.join(dirs[:15])}\nArquivos: {', '.join(files[:25])}"
            if len(items) > 40:
                tree_text += "\n...(truncado)..."
                
            self._inject_into_ooda(f"[SENSOR FS: Visao da Arvore]\n{tree_text}")
            
            addr = sensory_addr("fs", "tree_view")
            self.router.pulse(addr, {"path": path, "content": tree_text})
            
        except Exception as e:
            self._inject_into_ooda(f"[SENSOR FS: Falha de exploracao: {e}]")

    def _read_and_collapse_to_vector(self, filepath: str):
        logger.info(f"[{self.name}] Lendo, Calculando Assinatura Blake3 e Colapsando: {filepath}")
        try:
            if not os.path.exists(filepath):
                self._inject_into_ooda(f"[SENSOR FS: Arquivo {filepath} nao existe]")
                return

            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            blake_sig = get_physical_signature(filepath)

            truncated_content = content[:3000]
            if len(content) > 3000:
                truncated_content += "\n...[CONTEUDO TRUNCADO]..."

            file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
            vector_addr = sensory_addr("fs", f"file_{file_hash}")
            
            node_state = SynapticNodeState(
                address=vector_addr,
                model_id="fs_sense_organ",
                status="collapsed",
                telemetry={"filepath": filepath, "size": len(content), "blake3_sig": blake_sig}
            )
            self.router.register_node(node_state)
            
            self.router.pulse(vector_addr, {
                "filepath": filepath,
                "size_bytes": len(content),
                "blake3_sig": blake_sig,
                "superposition": "collapsed",
                "content": truncated_content
            })

            ooda_msg = (
                f"[SENSOR FS: Ingestão BlakeGravity de Arquivo Colapsado em {vector_addr.vector_path}]\n"
                f"Caminho: {filepath} | Assinatura Blake3: {blake_sig[:12]}...\n"
                f"Conteúdo:\n{truncated_content}"
            )
            self._inject_into_ooda(ooda_msg)

        except Exception as e:
            self._inject_into_ooda(f"[SENSOR FS: Falha ao colapsar {filepath}: {e}]")

def init_organ():
    organ = FileSystemSenseOrgan()
    try:
        import swa.broadcast_runtime
        original_broadcast = swa.broadcast_runtime.BroadcastRuntime.broadcast_state
        
        def wrapped_broadcast(self, *args, **kwargs):
            if hasattr(self, 'last_thought') and self.last_thought:
                threading.Thread(
                    target=organ._on_cortex_thought, 
                    args=(SynapticRegion.CORTEX.value, {"pensamento": self.last_thought}),
                    daemon=True
                ).start()
            return original_broadcast(self, *args, **kwargs)

        if not hasattr(swa.broadcast_runtime.BroadcastRuntime.broadcast_state, '_fs_patched'):
            wrapped_broadcast._fs_patched = True
            swa.broadcast_runtime.BroadcastRuntime.broadcast_state = wrapped_broadcast
            
    except ImportError:
        pass
        
    return organ

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    organ = init_organ()
    print("FileSystemSenseOrgan ativado e escutando edições físicas...")
