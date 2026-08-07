#!/usr/bin/env python3
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import psycopg2
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para importar swa
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from swa.adapter import SemanticWarpAdapter, triplet_loss
try:
    from cortex.native_engine import AeonTopologicalKernel
except ImportError:
    # Caso esteja rodando direto do script sem path root ajustado
    from native_engine import AeonTopologicalKernel

print("\033[95m[Fase REM] Iniciando ciclo de sono profundo e reestruturação topológica...\033[0m")

def dream_cycle():
    device = torch.device("cpu")
    
    # 1. Instanciar Adaptadores
    print("🧠 [REM] Invocando SWA 270M (Local) e SWA 1B (Macro)...")
    adapter_local = SemanticWarpAdapter(d=640).to(device)
    adapter_macro = SemanticWarpAdapter(d=640, prefix_mode=True, d_model=1152).to(device)
    kernel = AeonTopologicalKernel(hidden_dim=64, num_blocks=4).to(device)
    
    opt_local = optim.Adam(adapter_local.parameters(), lr=1e-4)
    opt_macro = optim.Adam(adapter_macro.parameters(), lr=1e-4)
    opt_kernel = optim.Adam(kernel.parameters(), lr=5e-4)
    
    # 2. Conectar ao Banco de Memórias
    try:
        conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
        cur = conn.cursor()
    except Exception as e:
        print(f"⚠️ [REM] Falha ao conectar ao hipocampo (PostgreSQL): {e}")
        return

    # Loop de Sono Profundo
    epoch = 0
    try:
        print("🌌 [REM] Entrando no vazio. Treinamento auto-supervisionado ativado.")
        while True:
            epoch += 1
            
            # Durante o sono, tentamos recuperar as memórias episódicas reais
            cur.execute("SELECT embedding_2560 FROM swa_episodic_episodes ORDER BY RANDOM() LIMIT 16;")
            memories = cur.fetchall()
            
            if memories:
                v_list = []
                import ast
                for row in memories:
                    raw_emb = row[0]
                    if isinstance(raw_emb, str):
                        emb = ast.literal_eval(raw_emb)
                    else:
                        emb = list(raw_emb)
                    # Slice biológico: Aproveita as primeiras 640 dimensões da memória para o adaptador rápido
                    v_list.append(emb[:640])
                
                v_anchor = torch.tensor(v_list, dtype=torch.float32)
                # Se não conseguiu pegar 16, preenche o resto com randn
                if v_anchor.size(0) < 16:
                    pad_size = 16 - v_anchor.size(0)
                    v_anchor = torch.cat([v_anchor, torch.randn(pad_size, 640)])
            else:
                v_anchor = torch.randn(16, 640)

            # O Alvo Positivo (v_pos) é a memória real com leve deriva térmica (generalização abstrata)
            v_pos = v_anchor + torch.randn(16, 640) * 0.05
            
            # O Alvo Negativo (v_neg) é o vácuo randômico puro (Polo Banal/Entropia estática)
            v_neg = torch.randn(16, 640) * 1.5
                
            opt_local.zero_grad()
            opt_macro.zero_grad()
            
            # Treinamento Auto-Supervisionado (Reconstrução e Ortogonalidade)
            z_local = adapter_local(v_anchor)
            loss_recon_local = F.mse_loss(z_local, v_anchor)
            loss_ortho_local = adapter_local.ortho_loss()
            loss_triplet_local = triplet_loss(adapter_local(v_anchor), adapter_local(v_pos), adapter_local(v_neg))
            
            loss_local = loss_recon_local + loss_ortho_local + loss_triplet_local
            loss_local.backward()
            opt_local.step()
            
            z_macro = adapter_macro(v_anchor)
            loss_recon_macro = F.mse_loss(z_macro, v_anchor)
            loss_ortho_macro = adapter_macro.ortho_loss()
            loss_triplet_macro = triplet_loss(adapter_macro(v_anchor), adapter_macro(v_pos), adapter_macro(v_neg))
            
            loss_macro = loss_recon_macro + loss_ortho_macro + loss_triplet_macro
            loss_macro.backward()
            opt_macro.step()
            
            # --- Treinamento do Motor Topológico (AeonKernel Denoising) ---
            # Pad da memória 640D para 4096D e conversão para matriz 2D (Física)
            v_anchor_4096 = F.pad(v_anchor, (0, 4096 - 640), "constant", 0.0)
            v_spatial = v_anchor_4096.view(-1, 1, 64, 64)
            # Injeta Ruído Térmico Destrutivo
            v_noisy = v_spatial + torch.randn_like(v_spatial) * 0.5
            
            opt_kernel.zero_grad()
            v_pred = kernel(v_noisy)
            loss_kernel = kernel.energy_loss(v_pred, v_spatial)
            loss_kernel.backward()
            opt_kernel.step()
            
            # Printa o eletroencefalograma do sonho
            if epoch % 20 == 0:
                print(f"💤 [REM Ciclo {epoch}] Loss Local: {loss_local.item():.4f} | Loss Macro: {loss_macro.item():.4f} | Loss Kernel: {loss_kernel.item():.4f}")
            
            time.sleep(5.0) # CICLO CONTROLADO: consolidação de sonho a ~0,2Hz libera CPU p/ inferência central
            
    except KeyboardInterrupt:
        print("\n🌅 [REM] Acordando. Ciclo de sono interrompido.")
    finally:
        cur.close()
        conn.close()
        # Salva os pesos consolidados do sonho
        os.makedirs("/services/model-with-adpter-logic-semantic/models", exist_ok=True)
        torch.save(adapter_local.state_dict(), "/services/model-with-adpter-logic-semantic/models/adapter_local_rem.pt")
        torch.save(adapter_macro.state_dict(), "/services/model-with-adpter-logic-semantic/models/adapter_macro_rem.pt")
        torch.save(kernel.state_dict(), "/services/model-with-adpter-logic-semantic/models/aeon_kernel_rem.pt")
        print("💾 [REM] Sinapses consolidadas (Adaptadores + Kernel) e salvas em disco. Bom dia.")

if __name__ == "__main__":
    dream_cycle()
