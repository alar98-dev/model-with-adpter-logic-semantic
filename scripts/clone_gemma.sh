#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo ".env not found in project root. Create it with HF_TOKEN=..." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is empty. Set HF_TOKEN in $ENV_FILE" >&2
  exit 1
fi

DEST_DIR="$ROOT_DIR/models/gemma3-4b"
mkdir -p "$(dirname "$DEST_DIR")"

if [ -d "$DEST_DIR" ] && { [ -f "$DEST_DIR/snapshots/"*/model-00001-of-*.safetensors ] || [ -f "$DEST_DIR/snapshots/"*/model.safetensors ] || [ -f "$DEST_DIR/pytorch_model.bin" ]; } 2>/dev/null; then
echo "Gemma3-4B already complete at $DEST_DIR"
  exit 0
fi

echo "Downloading/updating Gemma3-4B..."
rm -rf "$DEST_DIR"

python3 -c "
import os
os.environ['HF_HUB_DISABLE_XSS'] = '1'
from huggingface_hub import snapshot_download
print('Downloading model files...')
snapshot_download(
repo_id='google/gemma-3-4b-it',
    local_dir='$DEST_DIR',
    token='$HF_TOKEN'
)
print('Download complete')
"

echo "Clone finished."
ls -la "$DEST_DIR" | sed -n '1,20p'
