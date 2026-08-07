import torch
import os
from swa.logical_engine import LogicalEngine
from swa.adapter import SemanticWarpAdapter

MODEL_PATH = "./models/gemma1b"  # Use 1B for faster CPU validation (3-4B shards missing)

def testar_camada_logica():
    print("--- 1. Inicializando Componentes SWA ---")
    
    d_model = 2560
    
    adapter_local = SemanticWarpAdapter(d_model)
    adapter_macro = SemanticWarpAdapter(d_model)

    print(f"--- 2. Carregando Gemma-3-4B na RAM (CPU) ---")
    try:
        engine = LogicalEngine(
            model_dir=MODEL_PATH,
            adapter_local=adapter_local,
            adapter_macro=adapter_macro,
            device='cpu',
            use_gemma=True
        )
        
        print("✅ Motor Lógico e Gemma carregados com sucesso.")

        print("\n--- 3. Testando Processamento de Evento (Vetorização) ---")
        teste_texto = "Definir protocolo de segurança para o módulo de semântica."
        v_global = engine.process_event(teste_texto)
        print(f"Entrada: {teste_texto}")
        print(f"Vetor Global Gerado (Shape): {v_global.shape}")

        print("\n--- 4. Testando Geração de Resposta Normalizada ---")
        prompt_usuario = "Como posso validar esse protocolo?"
        resposta = engine.generate_response(prompt_usuario)
        
        print(f"Pergunta: {prompt_usuario}")
        print(f"Resposta do Gemma:\n{'-'*30}\n{resposta}\n{'-'*30}")

    except Exception as e:
        print(f"\n❌ Erro durante a validação: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testar_camada_logica()
