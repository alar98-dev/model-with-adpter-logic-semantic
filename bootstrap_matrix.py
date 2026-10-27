#!/usr/bin/env python3
import os
import torch
import sys

# Adiciona o diretório atual ao path para importações relativas
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swa.dimensions import PROJECTION_VERSION, projection_matrix

def bootstrap_projection():
    print('🌌 [SWA-AEON BOOTSTRAP] Verificando Ponte Tensorial de Gravidade...')
    source_dim = 1152
    target_dim = 2560
    path = os.path.join(os.path.dirname(__file__), 'models', 'history_proj.pt')
    
    if os.path.exists(path):
        print(f'✅ Matriz de projeção já alinhada em: {path}')
        return

    print('⚠️ Matriz não encontrada! Iniciando Forja Ortogonal isolada (CPU Livre)...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    layer = torch.nn.Linear(source_dim, target_dim, bias=False, device='cpu', dtype=torch.float32)
    layer.weight.data.copy_(projection_matrix(source_dim, target_dim, dtype=torch.float32))

    torch.save(
        {
            'meta': {
                'version': PROJECTION_VERSION,
                'source_dim': source_dim,
                'target_dim': target_dim,
            },
            'state_dict': layer.state_dict(),
        },
        path,
    )
    print(f'✅ Matriz Gravitacional forjada e estabilizada com sucesso: {path}')

if __name__ == '__main__':
    bootstrap_projection()
