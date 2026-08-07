import json
import torch
import numpy as np

def calculate_synapse_coordinates():
    print("🌌 INICIANDO MAPEAMENTO TOPOLÓGICO DA SINAPSE DE AÇÃO")
    
    dataset_path = "/home/alifer/swa-aeon/dataset_sinapses.json"
    
    try:
        with open(dataset_path, "r") as f:
            dados = json.load(f)
    except FileNotFoundError:
        print("Dataset de treinamento não encontrado. Execute o train_synapse.py primeiro.")
        return

    # Organizar os tensores por categoria
    categorias_tensores = {}
    for item in dados:
        cat = item["categoria"]
        if cat not in categorias_tensores:
            categorias_tensores[cat] = []
        categorias_tensores[cat].append(item["tensor_2560"])

    centroides = {}
    
    for cat, tensores in categorias_tensores.items():
        # Converte a lista de tensores [10, 2560] para um tensor do PyTorch
        t_matrix = torch.tensor(tensores, dtype=torch.float32)
        
        # Calcula a Coordenada Central (Centróide / Média)
        centroide = t_matrix.mean(dim=0)
        
        # Opcional: Calcular a variância para saber quão "espalhado" é esse desejo de ação
        variancia = t_matrix.var(dim=0).mean().item()
        
        # Normaliza o centróide para facilitar o Cosine Similarity no futuro
        centroide_norm = torch.nn.functional.normalize(centroide, p=2, dim=0)
        
        centroides[cat] = {
            "centroide": centroide_norm.tolist(),
            "variancia": variancia,
            "exemplos_treino": len(tensores)
        }
        
        print(f"🎯 Sinapse de Ação '{cat}' mapeada! (Variância: {variancia:.6f})")

    out_file = "/home/alifer/swa-aeon/action_synapses.json"
    with open(out_file, "w") as f:
        json.dump(centroides, f)
        
    print(f"\n✅ Coordenadas de Ação gravadas no núcleo em: {out_file}")
    print("O motor do Aeon agora pode carregar este arquivo e disparar ferramentas quando a Similaridade Cosseno bater as coordenadas!")

if __name__ == "__main__":
    calculate_synapse_coordinates()
