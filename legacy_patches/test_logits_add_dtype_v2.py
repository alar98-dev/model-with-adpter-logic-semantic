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
    outputs = gen.lm(inputs_embeds=full_embeds, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, -1, :] 
    print(f'Logits dtype: {logits.dtype}')
    
    # Test steer_logits
    steer_logits = torch.randn(1, 262144, dtype=torch.float32)
    
    print(f'Steer logits dtype: {steer_logits.dtype}')
    
    try:
        # Check if logits addition fails here if they have different dtypes
        combined_logits = logits + (steer_logits * 0.20)
        print('SUCCESS')
    except Exception as e:
        print(f'FAILED: {e}')
        # Verify if casting steer_logits helps
        try:
            combined_logits = logits + (steer_logits.to(dtype=logits.dtype) * 0.20)
            print('SUCCESS (after cast)')
        except Exception as e2:
            print(f'FAILED AGAIN: {e2}')
