#!/usr/bin/env python3
import sys
import os
import torch
import psycopg2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swa.db_controllers import PostgresBridgeController, EpisodicMemoryController

def main():
    print("🚀 Iniciando Alinhamento Estrutural do Espaço Vetorial (PGVector)")
    
    db_uri = "dbname=swa_matrix host=localhost user=alifer"
    bridge = PostgresBridgeController(db_uri=db_uri)
    
    print("\n[1] Verificando e Migrando Schema...")
    if bridge.verify_schema_and_migrations():
        print("✅ Migração bem sucedida!")
    else:
        print("❌ Falha na migração (Verifique logs e URI).")
        return 1

    print("\n[2] Injetando Âncoras Semânticas de Indução (Baseline)...")
    episodic = EpisodicMemoryController(db_uri=db_uri)
    
    # Criando âncoras baselines 2560d (ex: "Forward", "Obstacle")
    emb_forward = torch.zeros(2560)
    emb_forward[0:100] = 1.0 
    emb_forward = (emb_forward / torch.norm(emb_forward)).tolist()

    emb_obstacle = torch.zeros(2560)
    emb_obstacle[100:200] = -1.0
    emb_obstacle = (emb_obstacle / torch.norm(emb_obstacle)).tolist()
    
    episodic.store_episode("Navegação livre e fluida para frente", emb_forward, "NAVIGATION", 1.0)
    episodic.store_episode("Caminho bloqueado, obstáculo estrutural detectado", emb_obstacle, "OBSTACLE", 1.5)
    
    print("✅ Âncoras vetoriais de inércia injetadas.")
    
    print("\n[3] Validando Indução do Espaço...")
    results = episodic.query_similar_episodes(emb_forward, limit=1)
    if results:
        print(f"✅ Indução correspondente ao espaço Forward: {results[0]['similarity']:.4f} (Esperado ~1.0)")
        print(f"   Conteúdo ressoado: {results[0]['content']}")
    else:
        print("❌ Falha ao buscar episódios no espaço. RAM fallback pode estar ativo.")
        
    print("\n🎉 Alinhamento Estrutural Finalizado com Sucesso!")

if __name__ == "__main__":
    main()
