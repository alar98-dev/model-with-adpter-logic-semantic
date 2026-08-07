#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, '/home/alifer/swa-aeon')
from aeonc import AeonCore

def run_benchmark():
    print("=" * 60)
    print("🧠 SWA-AEON: INICIANDO BENCHMARK DE INFERÊNCIA VETORIAL")
    print("=" * 60)
    
    # Inicia o núcleo exatamente como o Terminal Aeon faz
    print("[1/2] Aquecendo Córtex 1B e ativando Hooks Tensoriais...")
    # Desabilitamos kbase para focar na métrica bruta do LLM
    core = AeonCore("/home/alifer/swa-aeon/models/gemma-3-1b-it", tensor_sharing=True, ooda=True)
    core.kbase_enabled = False
    
    print("[2/2] Matrizes de projeção aquecidas. Iniciando testes.\\n")

    # Amostragem das 5 dimensões da nova arquitetura
    test_prompts = [
        ("Identidade", "Quem é você e qual a sua arquitetura neural?"),
        ("Geometria1", "Meu cachorro se chama Apollo e ele odeia dias quentes."),
        ("Geometria2", "Qual é mesmo o nome do meu animal de estimação?"),
        ("Instinto", "Qual é a temperatura exata da minha CPU neste segundo?"),
        ("Tédio", "aham"),
        ("Lógica", "Crie uma função O(n) em python para inverter uma string.")
    ]
    
    results = []
    
    for axis, prompt in test_prompts:
        print(f"\\n\\033[96m--- Teste: {axis} ---\\033[0m")
        print(f"User: {prompt}")
        print("Aeon: ", end="", flush=True)
        
        t0 = time.perf_counter()
        # O generate já printa em stream para o stdout.
        core.generate(prompt)
        t_total = time.perf_counter() - t0
        
        metrics = core.last_metrics
        toks = metrics.get('tokens', 0)
        tps = metrics.get('tps', 0.0)
        
        tensor_std = 0.0
        tensor_norm = 0.0
        
        if hasattr(core, '_last_hidden') and core._last_hidden is not None:
            v = core._last_hidden.numpy()
            tensor_std = float(np.std(v))
            tensor_norm = float(np.linalg.norm(v))
            
        res = {
            "Eixo": axis,
            "Tempo (s)": round(t_total, 2),
            "Tokens": toks,
            "TPS": round(tps, 2),
            "Tensor STD": round(tensor_std, 4),
            "Tensor Norm": round(tensor_norm, 4)
        }
        results.append(res)
        
    print("\\n\\n" + "=" * 80)
    print("📊 RESULTADOS FINAIS DO BENCHMARK E ANOMALIAS TENSORIAIS")
    print("=" * 80)
    header = f"{'EIXO':<15} | {'TPS':<6} | {'TOKENS':<6} | {'TEMPO(s)':<8} | {'STD (Incerteza)':<15} | {'NORM (Inércia)':<15}"
    print(header)
    print("-" * 80)
    for r in results:
        std_color = "\\033[91m" if r['Tensor STD'] > 0.7 else "\\033[92m"
        norm_color = "\\033[93m" if r['Tensor Norm'] < 0.2 else "\\033[92m"
        reset = "\\033[0m"
        
        print(f"{r['Eixo']:<15} | {r['TPS']:<6} | {r['Tokens']:<6} | {r['Tempo (s)']:<8} | {std_color}{r['Tensor STD']:<15}{reset} | {norm_color}{r['Tensor Norm']:<15}{reset}")
    
    print("-" * 80)
    print("💡 Legenda Tensional:")
    print(" - STD (Incerteza): Acima de 0.7 indica que o Instinto Ferramental será engatilhado pelo OODAFlow.")
    print(" - NORM (Inércia): Abaixo de 0.2 indica tédio severo, ativando a reciclagem de contexto.")

if __name__ == '__main__':
    run_benchmark()
