import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.gemma_integration import GemmaGenerator
from swa.tokenizer_manager import TokenizerManager
tm = TokenizerManager()
tok = tm.load('models/gemma-3-4-b')
gen = GemmaGenerator('models/gemma-3-4-b', device='cpu', tokenizer=tok)
print(f'LM Head weight shape: {gen.lm.lm_head.weight.shape}')

full_embeds = torch.randn(1, 10, gen.d_model, dtype=torch.bfloat16)
with torch.no_grad():
    outputs = gen.lm(inputs_embeds=full_embeds, return_dict=True)
    print(f'Logits shape: {outputs.logits.shape}')
    logits = outputs.logits[:, -1, :]
    print(f'Logits[:, -1, :] shape: {logits.shape}')
