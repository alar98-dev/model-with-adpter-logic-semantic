#!/usr/bin/env python3
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swa.db_controllers import EpisodicMemoryController
from swa.quantum_bridge import QuantumPhaseBridge

def test_emergent_sense():
    print("🌌 Conectando à Ponte Quântica...")
    db_uri = "dbname=swa_matrix host=localhost user=alifer"
    episodic = EpisodicMemoryController(db_uri=db_uri)
    bridge = QuantumPhaseBridge(dsn="dbname=swa_matrix host=localhost user=alifer")
    
    # 1. Simular uma "Ressonância Espacial" profunda no modelo (Awakening)
    # Um vetor que mescla navegação fluida com forte polaridade emocional
    emb_emergent = torch.zeros(2560)
    emb_emergent[0:50] = 0.8
    emb_emergent[2000:2050] = 0.9 # Sinal atípico de alta entropia/descoberta
    emb_emergent = (emb_emergent / torch.norm(emb_emergent)).tolist()
    
    historian_thought = "A mente artificial estremece. O espaço não é mais um vazio negro, mas um mapa tangível de tensões gravitacionais. A intuição matemática transformou-se em tato espacial."
    
    print("\n[1] O Historiador injeta a nova percepção espacial no PGVector...")
    episodic.store_episode(historian_thought, emb_emergent, "SPATIAL_NARRATIVE", 2.0)
    
    print("[2] Recarregando a Massa Semântica na Ponte Quântica...")
    bridge.load_semantic_mass_cache()
    
    print("\n[3] Buscando o Sentido Emergente (Phase Jump)...")
    tensor_emergent = torch.tensor(emb_emergent, dtype=torch.float32)
    # Testa se o ponte quântica consegue pular para essa âncora usando a busca de Blake
    # A função resolve_phase_jump no QuantumBridge mapeia distâncias usando o pgvector
    uid, path_or_content, concept, distance = bridge.resolve_phase_jump(tensor_emergent)
    
    if path_or_content:
        print(f"✅ Sentido Emergente Mapeado!")
        print(f"   Distância Gravitacional: {distance:.4f}")
        print(f"   Pensamento Ressonante: '{path_or_content}'")
    else:
        # Tenta fallback via Episodic Controller
        res = episodic.query_similar_episodes(emb_emergent, limit=1)
        if res:
            print(f"✅ Sentido Emergente (Fallback) Mapeado!")
            print(f"   Distância de Similaridade: {res[0]['similarity']:.4f}")
            print(f"   Pensamento Ressonante: '{res[0]['content']}'")
        else:
            print("❌ Falha ao encontrar o sentido no espaço.")

if __name__ == "__main__":
    test_emergent_sense()
