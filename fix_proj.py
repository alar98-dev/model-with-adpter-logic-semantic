import torch
import sys
sys.path.insert(0, '/home/alifer/swa-aeon')
from swa.dimensions import PROJECTION_VERSION, projection_matrix

source_dim = 1152
target_dim = 2560
path = '/home/alifer/swa-aeon/logs/semantic_proj.pt'
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
print('Matriz gravada com sucesso em', path)
