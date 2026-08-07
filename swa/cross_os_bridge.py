import asyncio
import struct
import torch
import torch.nn as nn
import threading
from typing import Optional

class CrossOSMLP(nn.Module):
    """
    Tálamos Neural: Converte o vetor termodinâmico de 8 dimensões
    do Windows para a geometria fractal de 2560 dimensões da Gemma.
    """
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        # Inicialização com peso muito pequeno para não destruir o modelo logo de cara
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class NeuralBridge:
    def __init__(self, port=9098):
        self.port = port
        self.mlp = CrossOSMLP().to(torch.bfloat16)  # Gemma 3 usa bf16
        self.queue = asyncio.Queue(maxsize=3)
        self.current_bias: Optional[torch.Tensor] = None
        self.host_entropy = 0.0
        self.running = True
        self._start_server_thread()

    def _start_server_thread(self):
        """Inicia o servidor UDP em uma thread daemon"""
        def udp_server():
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('127.0.0.1', self.port))
            sock.settimeout(2.0)
            print(f"🌉 [CROSS-OS BRIDGE] Tálamo UDP escutando na porta {self.port}...")
            
            while self.running:
                try:
                    data, _ = sock.recvfrom(32)
                    if len(data) == 32:
                        # 8 floats x 4 bytes = 32 bytes (Little Endian)
                        telemetry = struct.unpack('<8f', data)
                        
                        # Processamento imediato do tensor
                        with torch.no_grad():
                            x = torch.tensor(telemetry, dtype=torch.bfloat16).unsqueeze(0)
                            bias = self.mlp(x)
                            
                            # Clamp de segurança (Filtro Antienvenenamento)
                            bias = torch.clamp(bias, min=-0.1, max=0.1)
                            
                            self.current_bias = bias
                            self.host_entropy = telemetry[6] # Entropia computada
                            
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"⚠️ Falha no Tálamo: {e}")
                    
        threading.Thread(target=udp_server, daemon=True).start()

    def get_bias(self) -> Optional[torch.Tensor]:
        return self.current_bias
        
    def get_entropy(self) -> float:
        return self.host_entropy

# Instância Global Estática para ancoragem
_bridge = None

def get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = NeuralBridge()
    return _bridge
