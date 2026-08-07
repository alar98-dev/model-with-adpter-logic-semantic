import sys
import os
sys.path.insert(0, os.getcwd())
from swa.gemma_integration import GemmaGenerator
from swa.tokenizer_manager import TokenizerManager
tm = TokenizerManager()
tok = tm.load('models/gemma-3-4-b')
gen = GemmaGenerator('models/gemma-3-4-b', device='cpu', tokenizer=tok)
print(f'LM Head weight shape: {gen.lm.lm_head.weight.shape}')
print(f'Vocab size: {tok.vocab_size}')
