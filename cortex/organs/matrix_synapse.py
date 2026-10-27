import torch
import torch.nn as nn
import os
import json
import hashlib
from datetime import datetime

class MatrixSynapse(nn.Module):
    def __init__(self, dim=2560):
        super().__init__()
        self.gravity_well = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.gravity_well.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        return self.gravity_well(x)

def text_to_tensor(text, dim=2560):
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vals = [float(b)/255.0 for b in h]
    expanded = (vals * (dim // len(vals) + 1))[:dim]
    return torch.tensor(expanded, dtype=torch.float32)

def init_or_load_synapse(direction="in", dim=2560):
    pt_path = f'/home/alifer/swa-aeon/cortex/organs/swa_matrix_synapse_{direction}.pt'
    model = MatrixSynapse(dim=dim)
    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location='cpu', weights_only=True))
    else:
        torch.save(model.state_dict(), pt_path)
    return model

def filter_flow(payload_text, direction="in"):
    # direction can be "in" (intent to execute) or "out" (result from debian)
    model = init_or_load_synapse(direction=direction)
    with torch.no_grad():
        x = text_to_tensor(payload_text)
        out = model(x)
        dist = torch.norm(out).item()
        
    is_anomaly = dist > 5.0 
    
    log_path = '/services/model-with-adpter-logic-semantic/swa_matrix_metrics.log'
    timestamp = datetime.now().isoformat()
    record = {
        'timestamp': timestamp,
        'direction': direction.upper(),
        'payload_preview': payload_text[:200],
        'topological_temperature': float(dist),
        'status': 'ANOMALY' if is_anomaly else 'STABLE',
        'action': 'PERMITTED' 
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    return record
