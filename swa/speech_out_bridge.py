import socket
import json

class SpeechOutputBridge:
    def __init__(self, port=9100):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Permite broadcast local
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def broadcast(self, token: str, voice_type: str = "Voz Interna", origin: str = "Desconhecido", entropy: float = 0.0):
        if not token:
            return
            
        payload = {
            "token": token,
            "voice_type": voice_type,
            "origin": origin,
            "entropy": entropy
        }
        try:
            data = json.dumps(payload).encode('utf-8')
            self.sock.sendto(data, ('127.0.0.1', self.port))
        except Exception as e:
            pass
