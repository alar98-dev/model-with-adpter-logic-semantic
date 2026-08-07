import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.adapter import SemanticWarpAdapter

m = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
m.to(torch.float32)

v_global = torch.randn(1, 2560, dtype=torch.bfloat16)

try:
    v_f = v_global.to(dtype=torch.float32)
    z = m.latent_gelu(v_f)
    print(f'Z dtype: {z.dtype}')
    print(f'Proj weight dtype: {m.prefix_projection.weight.dtype}')
    print(f'Proj weight data: {m.prefix_projection.weight.data.dtype}')
    
    # Test if projection works with float32
    v_prefix = m.prefix_projection(z)
    print(f'V_prefix dtype: {v_prefix.dtype}')
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
