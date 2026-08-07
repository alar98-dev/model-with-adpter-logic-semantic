import socket
import numpy as np
import threading

class TensorBridge:
    """[SWA BRIDGE] Ponte de tensores via TCP Socket."""
    def __init__(self, port=20006):
        self.port = port
        self.addr = ('127.0.0.1', port)

    def send_tensor(self, tensor: np.ndarray):
        """Córtex (WSL) envia a topologia."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(self.addr)
                s.sendall(tensor.tobytes())
        except Exception:
            pass # Falha silenciosa se o corpo não estiver ativo

    def start_server(self, handler):
        """Corpo (Windows) recebe a topologia."""
        def server_loop():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(self.addr)
                s.listen()
                while True:
                    conn, _ = s.accept()
                    with conn:
                        data = conn.recv(64 * 64 * 4) # 4096 floats
                        if data:
                            grid = np.frombuffer(data, dtype=np.float32).reshape(64, 64)
                            handler(grid)
        
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
