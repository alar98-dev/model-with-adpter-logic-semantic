import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.adapter import SemanticWarpAdapter

print("Testing adapter with BFloat16...")
# Ensure it initializes as float32
m = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
m.to(torch.float32)

v = torch.randn(1, 2560, dtype=torch.bfloat16)
print(f'Adapter W_a dtype: {m.W_a.weight.dtype}')
print(f'Input dtype: {v.dtype}')
try:
    z = m.latent_gelu(v)
    print(f'Gelu Output dtype: {z.dtype}')
    p, b = m.prefix_and_bias(v)
    print(f'Prefix dtype: {p.dtype}')
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
