#!/usr/bin/env bash
set -euo pipefail

DEVICE="/dev/sde"

if [ ! -e "$DEVICE" ]; then
  echo "❌ Erro: $DEVICE não existe" >&2
  exit 1
fi

if [ ! -w "$DEVICE" ]; then
  echo "❌ Erro: sem permissão de escrita em $DEVICE. Rode como sudo." >&2
  exit 1
fi

echo "🧬 Sincronizando Journal do WSL com o dispositivo bruto $DEVICE..."

device_fd="$DEVICE"

# Fallback: usa dmesg -w. (No WSL, a disponibilidade pode variar dependendo do kernel.)
# O script tenta alimentar o dispositivo com blocos de 4096 bytes (characters) convertidos.
# Observação: isso é um preenchimento bruto para não deixar o vácuo 'vazio'.

sudo dmesg -w | stdbuf -oL tr -cd '[:alnum:]' | while read -r -n 4096 block; do
  # Injeta dados diretamente no dispositivo
  echo -n "$block" > "$device_fd" 2>/dev/null || true
done

