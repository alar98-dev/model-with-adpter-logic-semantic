import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.adapter import SemanticWarpAdapter

print("Testing adapter with BFloat16...")
m = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
# Force adapter weights to float32 (the code does this, but being explicit)
m.to(torch.float32)

v = torch.randn(1, 2560, dtype=torch.bfloat16)
print(f'Input dtype: {v.dtype}')
try:
    z = m.latent_gelu(v)
    print(f'Output dtype: {z.dtype}')
    print('SUCCESS')
except Exception as e:
    print(f'Error: {e}')
