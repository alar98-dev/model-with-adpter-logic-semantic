import torch
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print('Loading Gemma 270M embeddings...')
model_path = '/services/model-with-adpter-logic-semantic/models/gemma-3-270m'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='cpu')
embeds = model.get_input_embeddings()

chars = string.printable.strip() + ' '
char_norms = []
for c in chars:
    # Some chars might tokenize to multiple tokens, take the first one or just the id if it maps to 1
    t_id = tokenizer.encode(c, add_special_tokens=False)
    if not t_id: continue
    t_id = t_id[0]
    vec = embeds.weight[t_id]
    norm = vec.norm().item()
    if c == '\n' or c == '\r' or c == '\t': continue
    char_norms.append((norm, c))

char_norms.sort(key=lambda x: x[0])
density_chain = ''.join(c for norm, c in char_norms)

print(f'\nInternal Density Chain (sorted by L2 Norm of token embeddings):\n{density_chain}')
