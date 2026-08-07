import torch
import sys
import os
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

v_global = torch.randn(1, 2560, dtype=torch.bfloat16)

# Simulate generate_guided_stream logic
target_dtype = eng.gemma_gen.embeddings.weight.dtype
v_global_cast = v_global.to(device=eng.device, dtype=target_dtype)
print(f'Target dtype: {target_dtype}')
print(f'V_global_cast dtype: {v_global_cast.dtype}')

try:
    p, b = eng.adapter_macro.prefix_and_bias(v_global_cast)
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
