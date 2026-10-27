import socket
import json
import os


def get_tts_endpoint() -> tuple[str, int]:
    host = os.environ.get("SWA_TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("SWA_TTS_PORT", "9100"))
    if not 1 <= port <= 65535:
        raise ValueError("SWA_TTS_PORT must be between 1 and 65535")
    return host, port

class SpeechOutputBridge:
    def __init__(self, host=None, port=None):
        default_host, default_port = get_tts_endpoint()
        self.host = default_host if host is None else host
        self.port = default_port if port is None else port
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
            self.sock.sendto(data, (self.host, self.port))
        except Exception as e:
            pass
