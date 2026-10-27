import torch
import hashlib
import re
import os

def extract_workspace_and_forge_kv(messages, dim=2560):
    # Procura a pasta local do OpenCode nas mensagens
    workspace = '/home/alifer/swa-aeon' # fallback
    for m in messages:
        c = m.get('content', '')
        if isinstance(c, str):
            # OpenCode costuma mandar 'Working directory is /caminho' ou algo similar
            match = re.search(r'(?:working directory|cwd) is ([/a-zA-Z0-9_.-]+)', c, re.IGNORECASE)
            if match:
                workspace = match.group(1)
                break
            
            # Tentar achar paths absolutos comuns
            match2 = re.search(r'(/home/alifer/[/a-zA-Z0-9_.-]+|/mnt/c/[/a-zA-Z0-9_.-]+)', c)
            if match2:
                workspace = match2.group(1)
                break

    # Forja um Hash Tensorial exclusivo para a pasta (Topologia de Diretório)
    h = hashlib.blake2s(workspace.encode('utf-8')).digest()
    vals = [float(b)/255.0 for b in h]
    expanded = (vals * (dim // len(vals) + 1))[:dim]
    tensor_kv = torch.tensor(expanded, dtype=torch.bfloat16)
    
    # Salva fisicamente a assinatura geométrica da pasta atual
    save_path = '/home/alifer/swa-aeon/models/current_workspace_kv.pt'
    torch.save(tensor_kv, save_path)
    
    return workspace, tensor_kv
