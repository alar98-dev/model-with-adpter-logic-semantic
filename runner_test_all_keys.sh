#!/bin/bash
echo "=== TESTANDO TODAS AS CHAVES SSH EM 192.168.10.107:8022 ==="
keys=(
    "/home/alifer/.ssh/id_ed25519"
    "/mnt/c/Users/alifer.ramos/.ssh/id_ed25519"
    "/mnt/c/Users/alifer.ramos/.ssh/wsl_id_ed25519"
    "/mnt/c/Users/alifer.ramos/.ssh/qwen_matrix_mmap_key"
    "/mnt/c/Users/alifer.ramos/.ssh/id_ed25519_backup"
    "/mnt/c/Users/alifer.ramos/.ssh/alifer.ramos@automotion.com.br-id_ed25519"
)

for k in "${keys[@]}"; do
    echo "Testando chave: $k"
    ssh -p 8022 -i "$k" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=2 192.168.10.107 "echo '🟢 SUCESSO COM CHAVE $k'" 2>/dev/null && break
done
