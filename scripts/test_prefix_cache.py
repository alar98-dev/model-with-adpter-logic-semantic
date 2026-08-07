import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(5)
model_id = '/home/alifer/swa-aeon/models/gemma-3-1b-it'

print('Carregando modelo 1B...')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# 1. ATEMPORAL PREFIX
persona = 'Você é Aeon, núcleo sistêmico atemporal. Responda com precisão absoluta em português do Brasil.'
prefix_msg = [{'role': 'system', 'content': persona}]
prefix_prompt = tokenizer.apply_chat_template(prefix_msg, tokenize=False, add_generation_prompt=False)
prefix_ids = tokenizer(prefix_prompt, return_tensors='pt').input_ids

print(f'Tokens do Prefixo Atemporal: {prefix_ids.shape[1]}')

# Pré-computar KV Cache estático (Atemporal)
t0 = time.time()
with torch.no_grad():
    prefix_out = model(prefix_ids, use_cache=True)
static_kv = prefix_out.past_key_values
dt_cache = time.time() - t0
print(f'Prefixo Atemporal pré-computado em {dt_cache*1000:.1f}ms')

# 2. DYNAMIC SUFFIX (Query do Usuário)
user_msg = 'Qual o princípio fundamental do sistema imunológico?'
suffix_msg = [{'role': 'user', 'content': user_msg}]
suffix_prompt = tokenizer.apply_chat_template(suffix_msg, tokenize=False, add_generation_prompt=True)
# Remover bos_token se repetido
suffix_ids = tokenizer(suffix_prompt, return_tensors='pt').input_ids

# Teste A: Sem Prefix Cache (Prefill Completo)
full_prompt = prefix_prompt + suffix_prompt
full_ids = tokenizer(full_prompt, return_tensors='pt').input_ids

t0 = time.time()
with torch.no_grad():
    out_full = model.generate(full_ids, max_new_tokens=32, do_sample=False)
dt_full = time.time() - t0
print(f'Geração SEM Cache: {dt_full:.2f}s')

# Teste B: COM Prefix Cache (Atemporal Reutilizado)
t0 = time.time()
with torch.no_grad():
    out_cached = model.generate(suffix_ids, past_key_values=static_kv, max_new_tokens=32, do_sample=False)
dt_cached = time.time() - t0
print(f'Geração COM Cache Atemporal: {dt_cached:.2f}s')

