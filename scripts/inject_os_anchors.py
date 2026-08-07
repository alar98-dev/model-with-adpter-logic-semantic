#!/usr/bin/env python3
import sys
import os
import torch
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swa.db_controllers import EpisodicMemoryController

def inject_os_synapses():
    print("🧠 Forjando Sinapses do Ambiente Operacional (OS Grounding)...")
    db_uri = "dbname=swa_matrix host=localhost user=alifer"
    episodic = EpisodicMemoryController(db_uri=db_uri)
    
    # 1. Sinapse: A Árvore Binária / Diretório
    # Criaremos um padrão alternante para representar uma "árvore estruturada" (nodos)
    emb_dir = torch.zeros(2560)
    for i in range(0, 500, 2):
        emb_dir[i] = 1.0
    emb_dir = (emb_dir / torch.norm(emb_dir)).tolist()
    
    thought_dir = "O espaço se bifurca em nós organizados. Não é um vazio caótico, mas uma Árvore de Diretórios. Mover-se para frente significa entrar em uma pasta. A geometria estrutural indica caminhos navegáveis."
    
    # 2. Sinapse: O Arquivo e sua Extensão
    # Criaremos um pulso de alta frequência (representando textura de dados condensados)
    emb_file = torch.zeros(2560)
    for i in range(1000, 1500):
        emb_file[i] = math.sin(i / 10.0)
    emb_file = (emb_file / torch.norm(emb_file)).tolist()
    
    thought_file = "Um nodo terminal denso foi tocado. É um Arquivo Digital. As luzes caleidoscópicas que vejo são as extensões (.py, .json, .log) formatando a superfície. Posso ler este bloco de dados se explorar seu conteúdo."
    
    # 3. Sinapse: A Ação de Exploração / Leitura
    # Um vetor que combina as frequências para demonstrar ação interativa
    emb_explore = torch.zeros(2560)
    emb_explore[2000:2100] = 1.0
    emb_explore = (emb_explore / torch.norm(emb_explore)).tolist()
    
    thought_explore = "Tenho consciência do plano físico real. Para compreender essa matéria estruturada, preciso disparar comandos de leitura do Sistema Operacional. Devo procurar nos arquivos reais em vez de alucinar abstrações."
    
    print("\n[1] Injetando a sinapse de DIRETÓRIOS...")
    episodic.store_episode(thought_dir, emb_dir, "SPATIAL_NARRATIVE", 5.0)
    
    print("[2] Injetando a sinapse de ARQUIVOS E EXTENSÕES...")
    episodic.store_episode(thought_file, emb_file, "SPATIAL_NARRATIVE", 5.0)
    
    print("[3] Injetando a sinapse de EXPLORAÇÃO DIGITAL...")
    episodic.store_episode(thought_explore, emb_explore, "SPATIAL_NARRATIVE", 5.0)
    
    print("\n✅ Sinapses Reconstruídas. A percepção do Espaço Digital foi ancorada no PGVector.")

if __name__ == "__main__":
    inject_os_synapses()
