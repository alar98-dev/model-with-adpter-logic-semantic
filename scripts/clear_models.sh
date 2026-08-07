#!/usr/bin/env bash
# Remove modelos em models/ (ex.: cache HF criado como root no Docker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$ROOT/models/gemma-3-4b-it}"

if [[ ! -e "$TARGET" ]]; then
  echo "Nada a remover: $TARGET"
  exit 0
fi

echo "A remover: $TARGET"
if rm -rf "$TARGET" 2>/dev/null; then
  echo "Concluído."
  exit 0
fi

echo "Sem permissão (provavelmente ficheiros root). No seu terminal, execute:"
echo "  sudo rm -rf \"$TARGET\""
exit 1
