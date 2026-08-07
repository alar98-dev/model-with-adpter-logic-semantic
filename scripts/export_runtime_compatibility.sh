#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/gemma-3-4-b}"
OUT="${OUT:-runtime_compatibility_matrix.json}"

python scripts/export_runtime_compatibility.py --model-dir "$MODEL_DIR" --out "$OUT"

