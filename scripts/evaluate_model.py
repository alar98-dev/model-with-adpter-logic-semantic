import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import psutil
import time

# Otimização específica para Dual Xeon E5620 (Nó 1 - Interface)
def set_interface_affinity():
    try:
        p = psutil.Process(os.getpid())
        # Núcleos ímpares (Nó 1)
        p.cpu_affinity([1, 3, 5, 7, 9, 11, 13, 15])
        os.nice(-10)
        print("Afinidade de Interface (Nó 1) configurada.")
    except:
        print("Aviso: Falha ao configurar afinidade (requer sudo).")

def evaluate_gemma3(prompt):
    model_id = "models/gemma3-4b"  # Ajustado para o modelo baixado
    
    print(f"--- Avaliando Gemma-3-4B-IT no Xeon ---")
    print(f"Carregando modelo (isso pode levar alguns minutos em CPU)...")
    
    set_interface_affinity()
    
    # Carregando com float32 para estabilidade no Xeon antigo
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Prompt de teste
    chat = [
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
    
    print(f"\nPrompt: {prompt}")
    print("Gerando resposta...")
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- RESPOSTA ---")
    print(response.split("model\n")[-1] if "model\n" in response else response)
    print(f"\nTempo de geração: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    test_text = "Explique a relação entre álgebra linear e redes neurais de forma concisa em português."
    evaluate_gemma3(test_text)
