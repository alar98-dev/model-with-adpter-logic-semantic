import sys
import os
import torch
sys.path.insert(0, os.getcwd())
from swa.logical_engine import LogicalEngine
from swa.tokenizer_manager import TokenizerManager
from swa.adapter import SemanticWarpAdapter

print("Loading...")
tm = TokenizerManager()
tok = tm.load('models/gemma-3-4-b')
a_local = SemanticWarpAdapter(d=2560, k=32)
a_macro = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)
eng = LogicalEngine(model_dir='models/gemma-3-4-b', adapter_local=a_local, adapter_macro=a_macro, device='cpu', use_gemma=True, tokenizer=tok)

# Force dtype of adapter_macro to float32
a_macro.to(torch.float32)

v_global = torch.randn(1, 2560, dtype=torch.bfloat16)
print(f'Adapter macro dtype: {a_macro.W_a.weight.dtype}')
print(f'V_global dtype: {v_global.dtype}')
try:
    p, b = a_macro.prefix_and_bias(v_global)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
