import sys
import os
import torch

sys.path.insert(0, os.getcwd())

from swa.logical_engine import LogicalEngine
from swa.tokenizer_manager import TokenizerManager
from swa.adapter import SemanticWarpAdapter

print("Loading tokenizer...")
tm = TokenizerManager()
tok = tm.load('models/gemma-3-4-b')

print("Initializing engine...")
a_local = SemanticWarpAdapter(d=2560, k=32)
a_macro = SemanticWarpAdapter(d=2560, k=32, prefix_mode=True, d_model=2560)

eng = LogicalEngine(
    model_dir='models/gemma-3-4-b',
    adapter_local=a_local,
    adapter_macro=a_macro,
    device='cpu',
    use_gemma=True,
    tokenizer=tok,
)

prompt = "Test"
v_global = torch.zeros(1, 2560)

print("Testing stream...")
for i, t in enumerate(eng.generate_guided_stream(prompt, v_global, max_new_tokens=2)):
    print(f"Token {i} type: {type(t)}")
    print(f"Token {i} value: {t}")
print("Done")
