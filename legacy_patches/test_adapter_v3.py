import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.adapter import SemanticWarpAdapter

m = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
m.to(torch.float32)

v = torch.randn(1, 2560, dtype=torch.bfloat16)
print(f'W_a weight dtype: {m.W_a.weight.dtype}')
print(f'Input dtype: {v.dtype}')
try:
    v_f = v.to(dtype=torch.float32)
    v_flat = v_f.view(-1, m.d)
    print(f'V_flat dtype: {v_flat.dtype}')
    z = m.W_a(v_flat)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
