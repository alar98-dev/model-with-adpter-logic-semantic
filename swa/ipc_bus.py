# Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
import os
import socket
import struct
import threading
from typing import Callable, Optional, Tuple

AEON_SOCKET_PATH = "/var/run/aeon/synapse.sock"
HEADER_FORMAT = "!4096f I"  # Matriz Topológica 64x64 (4096 floats) + Length (I)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

class AeonIPCFrame:
    def __init__(self, topology_matrix, payload: bytes):
        self.topology = topology_matrix
        self.payload = payload

def pack_aeon_frame(topology_list, payload: bytes) -> bytes:
    """Packs an AEON tensor frame: Topology (4096f) | Length (I) | Payload"""
    header = struct.pack(HEADER_FORMAT, *topology_list, len(payload))
    return header + payload

def unpack_aeon_frame(sock: socket.socket):
    """Unpacks an AEON tensor frame from a socket stream."""
    try:
        import numpy as np
        header_bytes = b""
        while len(header_bytes) < HEADER_SIZE:
            chunk = sock.recv(HEADER_SIZE - len(header_bytes))
            if not chunk: return None
            header_bytes += chunk
            
        unpacked = struct.unpack(HEADER_FORMAT, header_bytes)
        topology_flat = unpacked[:4096]
        length = unpacked[4096]
        
        topology_matrix = np.array(topology_flat, dtype=np.float32).reshape(64, 64)
        
        payload = b""
        while len(payload) < length:
            chunk = sock.recv(length - len(payload))
            if not chunk:
                break
            payload += chunk
            
        if len(payload) != length:
            return None
            
        return topology_matrix, payload
    except Exception:
        return None

class AeonIPCServer:
    """Unix Domain Socket server executing the binary AEON IPC Bus."""
    def __init__(self, handler: Callable[[int, bytes], None], socket_path: str = AEON_SOCKET_PATH):
        self.socket_path = socket_path
        self.handler = handler
        self.running = False
        self.server_thread = None
        self.sock = None

    def start(self):
        self.running = True
        # Ensure path directory exists
        socket_dir = os.path.dirname(self.socket_path)
        os.makedirs(socket_dir, exist_ok=True)
        
        # Clean up existing socket if any
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
                
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        # Give read/write permissions to the socket
        try:
            os.chmod(self.socket_path, 0o666)
        except OSError:
            pass
            
        self.sock.listen(5)
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True, name="AEON-IPC-Server")
        self.server_thread.start()
        print(f"🔌 [IPC BUS] Servidor binário UDS iniciado em {self.socket_path}")

    def _server_loop(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                conn.settimeout(2.0)
                client_thread = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
                client_thread.start()
            except Exception:
                if not self.running:
                    break

    def _client_handler(self, conn: socket.socket):
        with conn:
            while self.running:
                frame = unpack_aeon_frame(conn)
                if frame is None:
                    break
                topology_matrix, payload = frame
                try:
                    self.handler(topology_matrix, payload)
                except Exception as e:
                    print(f"⚠️ [IPC BUS] Erro ao tratar matriz topológica: {e}")

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        if self.server_thread:
            self.server_thread.join(timeout=1.0)

import redis
import threading
import json

class RedisSynapse:
    def __init__(self, host="localhost", port=6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        
    def broadcast_event(self, event_type, payload):
        data = {"event_type": event_type, "payload": payload}
        self.r.publish("grid_events", json.dumps(data))

def pack_aeon_frame_with_clock(topology, payload_dict, clock):
    payload_dict["_clock"] = clock
    return pack_aeon_frame(topology, json.dumps(payload_dict).encode("utf-8"))
