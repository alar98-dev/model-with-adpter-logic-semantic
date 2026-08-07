import torch
import sys
sys.path.append('.')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/models/gemma-2-2b-it')
from swa.gemma_integration import GemmaGenerator
engine = GemmaGenerator(model_dir='/models/gemma-2-2b-it', tokenizer=tokenizer)
t1 = engine.tokenizer('oi bom dia', return_tensors='pt').input_ids.to(engine.lm.device)
t2 = engine.tokenizer('me liste os arquivos', return_tensors='pt').input_ids.to(engine.lm.device)
t3 = engine.tokenizer('boa tarde', return_tensors='pt').input_ids.to(engine.lm.device)

v1 = engine.lm.model.embed_tokens(t1).mean(dim=1)
v2 = engine.lm.model.embed_tokens(t2).mean(dim=1)
v3 = engine.lm.model.embed_tokens(t3).mean(dim=1)

import torch.nn.functional as F
print('Dist(Banality, Command):', 1.0 - F.cosine_similarity(v1, v2).item())
print('Dist(Banality, Casual):', 1.0 - F.cosine_similarity(v1, v3).item())
