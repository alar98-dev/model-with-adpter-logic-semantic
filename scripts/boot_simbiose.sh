#!/bin/bash

# boot_simbiose.sh - Inicialização otimizada para Dual Xeon E5620
# Configura afinidade NUMA e mlock para o projeto model-with-adapter-logic-semantic

MODEL_DIR="models/gemma3-4b"
DATA_FILE="data/train_triplets.jsonl"
CHECKPOINT_DIR="swa/checkpoints/gemma3-4b"

echo "=== Iniciando Simbiose Dual Xeon ==="

# 1. Configurar limites de recursos
ulimit -l unlimited  # Permite mlock (travar RAM)
echo "Ulimit configurado para mlock."

# 2. Criar diretório de checkpoints
mkdir -p $CHECKPOINT_DIR

# 3. Preparar Memória Compartilhada (Shared Memory) para transferência vetorial
if [ ! -d "/dev/shm/swa_bridge" ]; then
    echo "Criando bridge em /dev/shm/swa_bridge..."
    sudo mkdir -p /dev/shm/swa_bridge
    sudo chmod 777 /dev/shm/swa_bridge
fi

# 4. Rodar o treinamento no Nó 0 (Primeiro Soquete Xeon)
# --cpunodebind=0: Trava CPUs no Nó 0
# --membind=0: Trava alocação de RAM no Nó 0 (evita latência QPI)
echo "Lançando Especialista (Expert) no Nó 0..."
numactl --cpunodebind=0 --membind=0 python3 -m swa.train_adapter \
    --model-dir $MODEL_DIR \
    --triplets $DATA_FILE \
    --save-dir $CHECKPOINT_DIR \
    --epochs 5 \
    --batch 4

echo "Treinamento finalizado com afinidade de hardware."
