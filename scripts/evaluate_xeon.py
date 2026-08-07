import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os
import psutil
import ctypes
import time

def setup_xeon():
    # Afinidade para Xeon Nó 0
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([0, 2, 4, 6, 8, 10, 12, 14])
        os.nice(-10)
        print("Hardware: Xeon Nó 0 (CPUs Pares) Vinculados.")
    except:
        print("Hardware: Falha na afinidade (não rodando como root?)")

def lock_memory():
    # Mlock
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mlockall(3) # MCL_CURRENT | MCL_FUTURE
        print("Memory: RAM Travada (Mlock).")
    except:
        print("Memory: Falha no mlock.")

def run_eval():
    setup_xeon()
    model_id = "models/gemma3-4b"
    
    print(f"Carregando {model_id}...")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Usamos float32 para evitar emulação lenta no Xeon antigo
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float32, 
        device_map="cpu"
    )
    
    lock_memory()
    print(f"Modelo carregado em {time.time() - start_load:.2f}s")
    
    prompt = "Explique a diferença entre uma matriz de covariância e uma matriz de correlação de forma concisa e técnica."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\n" + "="*50)
    print("RESPOSTA DO GEMMA-3 (STREAMING):")
    print("-" * 50)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    start_gen = time.time()
    # Adicionamos do_sample=True com temperatura baixa para ver a 'dança' dos tokens
    outputs = model.generate(
        **inputs, 
        max_new_tokens=250, 
        streamer=streamer,
        do_sample=True,
        temperature=0.1
    )
    
    print("-" * 50)
    print("="*50)
    print(f"Tempo de geração: {time.time() - start_gen:.2f}s")

if __name__ == "__main__":
    run_eval()


if __name__ == "__main__":
    run_eval()
