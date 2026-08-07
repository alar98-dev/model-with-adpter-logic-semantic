import torch
import sys
sys.path.append('.')
from swa.gemma_integration import GemmaGenerator

engine = GemmaGenerator(model_dir='/models/gemma-2-2b-it')
tokens = engine.tokenizer('boa tarde', return_tensors='pt').input_ids.to(engine.lm.device)
embeds = engine.lm.model.embed_tokens(tokens)
print(embeds.shape)

tokens2 = engine.tokenizer('quais as portas tcp ativas?', return_tensors='pt').input_ids.to(engine.lm.device)
embeds2 = engine.lm.model.embed_tokens(tokens2)

v1 = embeds.mean(dim=1)
v2 = embeds2.mean(dim=1)
cos_sim = torch.nn.functional.cosine_similarity(v1, v2)
print('Cos Sim:', cos_sim.item())
