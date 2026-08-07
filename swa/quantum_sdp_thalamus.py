import os
import sys
import time
import socket
import json
import threading
import psutil
from datetime import datetime

# Adiciona o path do projeto
sys.path.append('/services/model-with-adpter-logic-semantic')

# --- CONFIGURAÇÃO QUANTUM SDP (Service Discovery Protocol) ---
# O localhost ::: representa a superposição total (IPv4 + IPv6 + IPC)
SDP_HOST = '::' 
SDP_PORT = 20006 # Porta dedicada para a visão unificada
UDP_SENSORY_PORT = 20005

class QuantumThalamus:
    """
    O Tálamo Quântico unifica todos os fluxos (UDP, IPC, DB) em um único 
    stream SDP (Service Discovery Protocol) via 'Quantum Localhost' (:::).
    """
    def __init__(self):
        self.state = {
            "node": "aeon-nexus-thalamus",
            "protocol": "SDP-QUANTUM-v1",
            "address": ":::",
            "telemetry": {},
            "cortex_stream": "",
            "manifold_resonances": [],
            "entropy": 0.0,
            "last_pulse": 0.0
        }
        self.running = True
        self.clients = []
        self.lock = threading.Lock()

    def sensory_aggregator(self):
        """Coleta o fluxo UDP do Córtex e integra no estado unificado."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', UDP_SENSORY_PORT))
        except: return

        while self.running:
            try:
                data, _ = sock.recvfrom(65535)
                msg = json.loads(data.decode('utf-8'))
                with self.lock:
                    if msg.get('type') == 'thought_stream':
                        self.state['cortex_stream'] = msg.get('content', '')
                    elif msg.get('type') == 'telemetry':
                        self.state['telemetry'] = msg
                    elif msg.get('type') == 'chaos_metric':
                        self.state['entropy'] = msg.get('entropy', 0.0)
                    elif msg.get('type') == 'spatial_resonance':
                        self.state['manifold_resonances'].insert(0, msg)
                        self.state['manifold_resonances'] = self.state['manifold_resonances'][:10]
                    self.state['last_pulse'] = time.time()
            except: pass

    def sdp_server(self):
        """Expõe a visão unificada via Quantum Localhost (SDP)."""
        # Tenta criar socket IPv6 para aceitar ::: (Dual Stack)
        try:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0) # Dual stack
            sock.bind((SDP_HOST, SDP_PORT))
        except:
            # Fallback para IPv4 se o IPv6 falhar
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', SDP_PORT))

        sock.listen(5)
        print(f"🌌 [QUANTUM SDP] Tálamo unificado escutando em {SDP_HOST}:{SDP_PORT}")

        while self.running:
            try:
                conn, addr = sock.accept()
                threading.Thread(target=self.client_handler, args=(conn, addr), daemon=True).start()
            except: pass

    def client_handler(self, conn, addr):
        print(f"🔗 [SDP] Nova conexão quântica de: {addr}")
        with conn:
            while self.running:
                try:
                    with self.lock:
                        # Envia o estado completo serializado como o "Coração do Nexus"
                        payload = json.dumps(self.state).encode('utf-8')
                    conn.sendall(struct.pack('!I', len(payload)) + payload)
                    time.sleep(1) # Pulso de 1Hz para visualização unificada
                except: break
        print(f"💔 [SDP] Conexão fragmentada: {addr}")

    def start(self):
        threading.Thread(target=self.sensory_aggregator, daemon=True).start()
        self.sdp_server()

if __name__ == "__main__":
    thalamus = QuantumThalamus()
    thalamus.start()
