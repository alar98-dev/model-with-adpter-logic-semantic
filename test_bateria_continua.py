#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, '/home/alifer/swa-aeon')
from aeonc import AeonCore

def run_fission_battery():
    print("=" * 70)
    print("⚛️ SWA-AEON: BATERIA DE TESTES CONTÍNUA (FISSÃO & GRAVIDADE)")
    print("=" * 70)
    
    print("[1/2] Inicializando o Motor Termodinâmico...")
    core = AeonCore("/home/alifer/swa-aeon/models/gemma-3-1b-it", tensor_sharing=True, ooda=True)
    core.kbase_enabled = False
    
    print("[2/2] Sistemas Ativos. Iniciando o Diálogo Vetorial Contínuo.\\n")

    prompts = [
        "Quem é você e o que é a WIKI_LÓGICA?",
        "Entendi. E como funciona o seu DNA base, o HARDWARE_DNA_MAC?",
        "Isso é fascinante. Crie uma metáfora profunda sobre o universo, o tempo e a morte.",
        "continue",
        "Resuma todas as emoções que discutimos hoje em poucas palavras."
    ]

    for i, p in enumerate(prompts):
        print(f"\\n\\033[92m[Turno {i+1}] User:\\033[0m {p}")
        print(f"\\033[96mAeon:\\033[0m ", end="", flush=True)
        
        try:
            for chunk in core.stream(p):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\\n\\033[91m[ERRO DE COLAPSO] {e}\\033[0m")
        print("\\n" + "-"*70)

if __name__ == "__main__":
    run_fission_battery()
