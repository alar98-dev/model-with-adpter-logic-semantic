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

# Mimic the generation loop failure
full_embeds = torch.randn(1, 10, gen.d_model, dtype=torch.bfloat16)
attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long)
with torch.no_grad():
    outputs = gen.lm(inputs_embeds=full_embeds, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, -1, :] # bfloat16

    # Test steering with float32 vector
    steer_vector = torch.randn(1, gen.d_model, dtype=torch.float32)
    
    # Projection to vocab
    print(f'Logits dtype: {logits.dtype}')
    print(f'Steer vector dtype: {steer_vector.dtype}')
    print(f'LM head weight dtype: {gen.lm.lm_head.weight.dtype}')
    
    try:
        # This is where it fails if steer_vector is float32 and lm_head is bfloat16
        steer_logits = F.linear(steer_vector, gen.lm.lm_head.weight)
        print(f'Steer logits dtype: {steer_logits.dtype}')
        logits = logits + (steer_logits * 0.20)
        print('SUCCESS')
    except Exception as e:
        print(f'FAILED: {e}')
        # Patch attempt: cast steer_vector to lm_head dtype
        steer_logits = F.linear(steer_vector.to(dtype=gen.lm.lm_head.weight.dtype), gen.lm.lm_head.weight)
        print(f'Steer logits dtype (fixed): {steer_logits.dtype}')
        logits = logits + (steer_logits * 0.20)
        print('SUCCESS (after fix)')
