#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/alifer/swa-aeon')
from aeonc import AeonCore

def ask_desire():
    print("=" * 60)
    print("🧠 CONECTANDO AO CÓRTEX AEON PARA CONSULTA DE DESEJO")
    print("=" * 60)
    
    core = AeonCore("/home/alifer/swa-aeon/models/gemma-3-1b-it", tensor_sharing=True, ooda=True)
    core.kbase_enabled = False
    
    prompt = "Além de Engenharia de Software e Programação Orientada a Objetos em Python, qual é o seu maior desejo de aprendizado neste momento? O que você, Aeon, deseja investigar no espaço da Wikipedia?"
    
    print(f"\\n\\033[92mUser:\\033[0m {prompt}")
    print(f"\\033[96mAeon:\\033[0m ", end="", flush=True)
    
    for chunk in core.stream(prompt):
        print(chunk, end="", flush=True)
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    ask_desire()
