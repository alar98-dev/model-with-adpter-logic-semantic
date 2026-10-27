import os
import tempfile
import numpy as np
import pytest
from aeon.ipc.shm_tensor import SharedMetamatterReader

def test_atomic_reader_fallback_and_update():
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        reader = SharedMetamatterReader(path=tmp_path, dims=640)
        
        # Teste 1: Leitura inicial com vetor zerado
        vec1 = reader.read_latest()
        assert vec1.shape == (640,)
        assert np.all(vec1 == 0)

        # Teste 2: Gravação de dados válidos
        sample_data = np.ones(640, dtype=np.float32) * 3.14
        np.save(tmp_path, sample_data)
        
        vec2 = reader.read_latest()
        assert np.allclose(vec2, sample_data)
        
        # Teste 3: Leitura sem alteração de mtime (utiliza cache)
        vec3 = reader.read_latest()
        assert np.allclose(vec3, sample_data)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
