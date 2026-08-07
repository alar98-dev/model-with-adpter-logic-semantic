import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(5)
model_id = '/home/alifer/swa-aeon/models/gemma-3-1b-it'
draft_id = '/home/alifer/swa-aeon/models/gemma-3-270m-it'

print('Carregando 1B...')
m1b = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
print('Carregando 270M...')
m270 = AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

prompt = 'Explique sucintamente como funciona um sistema imunologico em computadores.'
ids = tok(prompt, return_tensors='pt')

print('--- Teste Sem Draft (1B Puro) ---')
t0 = time.time()
with torch.inference_mode():
    out1 = m1b.generate(**ids, max_new_tokens=128, do_sample=True, temperature=0.7)
dt1 = time.time() - t0
n1 = out1.shape[1] - ids['input_ids'].shape[1]
print(f'1B Puro: {n1} tokens em {dt1:.2f}s ({n1/dt1:.2f} tps)')

print('--- Teste Com Draft (270M Speculative) ---')
t0 = time.time()
with torch.inference_mode():
    out2 = m1b.generate(**ids, assistant_model=m270, max_new_tokens=128, do_sample=True, temperature=0.7)
dt2 = time.time() - t0
n2 = out2.shape[1] - ids['input_ids'].shape[1]
print(f'270M Draft Speculative: {n2} tokens em {dt2:.2f}s ({n2/dt2:.2f} tps)')
