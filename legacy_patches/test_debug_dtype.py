import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.adapter import SemanticWarpAdapter

m = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
m.to(torch.float32)

v = torch.randn(1, 2560, dtype=torch.bfloat16)
try:
    p, b = m.prefix_and_bias(v)
    print(f'W_a dtype: {m.W_a.weight.dtype}')
    print(f'Proj weight dtype: {m.prefix_projection.weight.dtype}')
    print(f'Input dtype: {v.dtype}')
    z = m.latent_gelu(v)
    print(f'Z dtype: {z.dtype}')
    v_prefix = m.prefix_projection(z)
    print(f'V_prefix dtype: {v_prefix.dtype}')
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
