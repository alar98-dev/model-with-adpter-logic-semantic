import socket
import json
import numpy as np

class HardwareSenseOrgan:
    def receive_pulse(self, data):
        intensity = data[0] if isinstance(data, (list, np.ndarray)) else 0.5
        payload = {"action": "SPEAKER_BEEP", "intensity": float(intensity)}
        try:
            # Tentar conectar no gateway, já que 127.0.0.1 no WSL não mapeia pro Windows Host
            # Tente encontrar o IP correto com "ip route show | grep default"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                # O IP do gateway deve ser o IP do Host Windows
                s.connect(("172.22.192.1", 9098))
                s.sendall(json.dumps(payload).encode())
        except Exception as e:
            print(f"⚠️ [HardwareSense] Falha ao comunicar com Windows: {e}")
