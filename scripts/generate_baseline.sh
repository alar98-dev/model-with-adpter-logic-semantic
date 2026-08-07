#!/bin/bash
set -euo pipefail

# Gera baseline BLAKE3 para todos os arquivos monitorados.
# Uso: sudo bash scripts/generate_baseline.sh
# Delega para Python para usar o mesmo algoritmo do collector.py

SWA_ROOT="/services/model-with-adpter-logic-semantic"
SCRIPT="$SWA_ROOT/scripts/generate_baseline.py"
VENV_PYTHON="$SWA_ROOT/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

exec "$VENV_PYTHON" "$SCRIPT"
