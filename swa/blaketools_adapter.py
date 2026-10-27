import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/alifer/swa-aeon')

from swa.dimensions import NATIVE_DIM, MODEL_DIM, MEMORY_DIM

ADAPTER_PT_PATH = '/home/alifer/swa-aeon/compiled_adapters/blaketools_adapter.pt'
SHM_ADAPTER_PATH = '/dev/shm/blaketools_adapter.pt'

class BlakeToolsTensorAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_steering = nn.Linear(NATIVE_DIM, MODEL_DIM, bias=False)
        self.proj_memory = nn.Linear(NATIVE_DIM, MEMORY_DIM, bias=False)
        
        nn.init.orthogonal_(self.proj_steering.weight)
        nn.init.orthogonal_(self.proj_memory.weight)

    def forward(self, v_native_640: torch.Tensor):
        v_float = v_native_640.float()
        steering = F.normalize(self.proj_steering(v_float), p=2, dim=-1)
        memory = F.normalize(self.proj_memory(v_float), p=2, dim=-1)
        return steering, memory

_GLOBAL_ADAPTER = None

def get_blaketools_adapter() -> BlakeToolsTensorAdapter:
    global _GLOBAL_ADAPTER
    if _GLOBAL_ADAPTER is not None:
        return _GLOBAL_ADAPTER

    adapter = BlakeToolsTensorAdapter()
    
    target_path = SHM_ADAPTER_PATH if os.path.exists(SHM_ADAPTER_PATH) else ADAPTER_PT_PATH
    
    if os.path.exists(target_path):
        try:
            state_dict = torch.load(target_path, map_location='cpu', weights_only=True)
            adapter.load_state_dict(state_dict)
        except Exception as e:
            print(f"⚠️ Aviso ao carregar adaptador .pt ({e}). Reinicializando matriz ortogonal.")
            _save_adapter(adapter)
    else:
        _save_adapter(adapter)
        
    _GLOBAL_ADAPTER = adapter
    return _GLOBAL_ADAPTER

def _save_adapter(adapter: BlakeToolsTensorAdapter):
    os.makedirs('/home/alifer/swa-aeon/compiled_adapters', exist_ok=True)
    state = adapter.state_dict()
    torch.save(state, ADAPTER_PT_PATH)
    try:
        torch.save(state, SHM_ADAPTER_PATH)
    except Exception:
        pass

def project_blake_hash_to_tensors(blake3_hex: str):
    raw_bytes = bytes.fromhex(blake3_hex[:640]) if len(blake3_hex) >= 640 else blake3_hex.encode('utf-8')
    expanded = [float(b) / 255.0 for b in raw_bytes]
    if len(expanded) < NATIVE_DIM:
        expanded = (expanded * (NATIVE_DIM // len(expanded) + 1))[:NATIVE_DIM]
    else:
        expanded = expanded[:NATIVE_DIM]
        
    t_640 = torch.tensor(expanded, dtype=torch.float32)
    adapter = get_blaketools_adapter()
    
    with torch.no_grad():
        v_1152, v_2560 = adapter(t_640)
        
    return t_640, v_1152, v_2560

if __name__ == "__main__":
    adapter = get_blaketools_adapter()
    print("✅ BlakeToolsTensorAdapter inicializado e dimensionado com sucesso!")
    t640, v1152, v2560 = project_blake_hash_to_tensors("a1b2c3d4e5f67890a1b2c3d4e5f67890")
    print(f"  -> Tensor 640D shape: {t640.shape}")
    print(f"  -> Tensor Steering 1152D shape: {v1152.shape} (norm: {v1152.norm():.4f})")
    print(f"  -> Tensor Memory 2560D shape: {v2560.shape} (norm: {v2560.norm():.4f})")
