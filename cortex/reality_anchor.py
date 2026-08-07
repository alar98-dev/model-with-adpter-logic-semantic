# cortex/reality_anchor.py
import os
import sys
from typing import Tuple
import numpy as np
# [SWA BRIDGE] Importação do bridge
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from swa.tensor_bridge import TensorBridge

class RealityBufferSandbox:
    """
    Gerencia o envio de tensores para o Corpo via TCP.
    """
    def __init__(self):
        self.bridge = TensorBridge()

    def injetar_realidade_processada(self, nova_matria: np.ndarray) -> Tuple[bool, str]:
        """Envia a topologia para a interface visual via TCP."""
        try:
            # [SWA BRIDGE] Envio de tensores via Socket (substitui SHM)
            self.bridge.send_tensor(nova_matria.astype(np.float32))
            return True, "SUCESSO: Tensores pulsados na malha visual."
        except Exception as e:
            return False, f"Falha na injeção física: {e}"
