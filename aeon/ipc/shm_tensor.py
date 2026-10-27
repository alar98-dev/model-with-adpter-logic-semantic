import os
import time
import numpy as np

class SharedMetamatterReader:
    """
    Leitor atômico de metamatéria em memória compartilhada com verificação de timestamp (st_mtime).
    Previne leitura parcial (tearing) ao ler a matriz 640D em /tmp/swa_metamatter.npy.
    """
    def __init__(self, path: str = "/tmp/swa_metamatter.npy", dims: int = 640):
        self.path = path
        self.dims = dims
        self.last_mtime = 0.0
        self.cached_vector = np.zeros(dims, dtype=np.float32)

    def read_latest(self) -> np.ndarray:
        if not os.path.exists(self.path):
            return self.cached_vector

        try:
            stat_info = os.stat(self.path)
            current_mtime = stat_info.st_mtime
            
            if current_mtime == self.last_mtime:
                return self.cached_vector
                
            data = np.load(self.path)
            if data.shape == (self.dims,):
                self.cached_vector = data.astype(np.float32)
                self.last_mtime = current_mtime
        except (OSError, ValueError, Exception):
            pass

        return self.cached_vector

    def read_latest_vector(self):
        """Compatibilidade de alias com SharedMetamatterIPC."""
        import torch
        vec = self.read_latest()
        return torch.from_numpy(vec), int(self.last_mtime), int(time.time() * 1e9)

# Alias de compatibilidade com a especificação SWA-Aeon 3.0
SharedMetamatterIPC = SharedMetamatterReader
