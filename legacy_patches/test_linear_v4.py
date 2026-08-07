import torch
import torch.nn.functional as F
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
    outputs = gen.lm(inputs_embeds=full_embeds, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, -1, :] 

    steer_vector = torch.randn(1, gen.d_model, dtype=torch.float32)
    
    print(f'Weight dtype: {gen.lm.lm_head.weight.dtype}')
    print(f'Input dtype: {steer_vector.dtype}')
    
    try:
        F.linear(steer_vector, gen.lm.lm_head.weight)
        print('SUCCESS')
    except Exception as e:
        print(f'FAILED: {e}')
        # Check if casting the weight works
        try:
            F.linear(steer_vector, gen.lm.lm_head.weight.to(dtype=steer_vector.dtype))
            print('SUCCESS (cast weight)')
        except Exception as e2:
            print(f'FAILED AGAIN: {e2}')
