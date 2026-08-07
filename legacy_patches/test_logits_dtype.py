import torch
import sys
import os
sys.path.insert(0, os.getcwd())
from swa.gemma_integration import GemmaGenerator
from swa.tokenizer_manager import TokenizerManager

tm = TokenizerManager()
tok = tm.load('models/gemma-3-4-b')
gen = GemmaGenerator('models/gemma-3-4-b', device='cpu', tokenizer=tok)

full_embeds = torch.randn(1, 10, gen.d_model, dtype=torch.bfloat16)
attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long)
with torch.no_grad():
    outputs = gen.lm(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )
print(f'Outputs.logits dtype: {outputs.logits.dtype}')
