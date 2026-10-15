#!/usr/bin/env bash
# fetch_weights.sh -- Download Cosmos-H-Surgical-Simulator weights from HuggingFace.
#
# Runs on a cluster LOGIN node (assumes outbound HF access).
#
# Usage:
#   export HF_TOKEN="hf_..."
#   ./cluster/fetch_weights.sh [WEIGHTS_DIR]
#
# Downloads model_ema_bf16.pt into WEIGHTS_DIR (default: ./weights).
# The .pt file is ~5 GB (bf16 EMA weights, recommended for inference).
#
# IMPORTANT: The checkpoint MUST be .pt format.
# The model loader (model_loader.py line 205) checks `ckpt_path.endswith(".pt")`
# to select the single-file loading path.  A .safetensors file would be
# misinterpreted as a DCP (distributed checkpoint) directory and FAIL.
#
# Prerequisites:
#   1. pip install huggingface_hub   (or: pip install "huggingface_hub[cli]")
#   2. Accept the NVIDIA Open Model License Agreement at:
#      https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator
#   3. export HF_TOKEN="hf_..."  OR  huggingface-cli login
set -euo pipefail

HF_REPO="nvidia/Cosmos-H-Surgical-Simulator"
WEIGHTS_DIR="${1:-${WEIGHTS_DIR:-./weights}}"

echo "=== Fetching Cosmos-H-Surgical-Simulator Weights ==="
echo "HF repo:     $HF_REPO"
echo "Destination: $WEIGHTS_DIR"

# ------------------------------------------------------------------
# Sanity-check authentication
# ------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null 2>&1; then
        echo ""
        echo "ERROR: No HF_TOKEN set and no cached login session found."
        echo "  Option 1:  export HF_TOKEN=\"hf_...\" && $0"
        echo "  Option 2:  huggingface-cli login   (interactive, then re-run)"
        exit 1
    fi
fi

mkdir -p "$WEIGHTS_DIR"

# ------------------------------------------------------------------
# Download model_ema_bf16.pt
# ------------------------------------------------------------------
echo ""
echo "--- Downloading model_ema_bf16.pt ---"
python3 -c "
from huggingface_hub import hf_hub_download
import os, sys

token = os.environ.get('HF_TOKEN', None)
local_dir = sys.argv[1]

path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='model_ema_bf16.pt',
    local_dir=local_dir,
    token=token,
)
print(f'Downloaded to: {path}')
" "$WEIGHTS_DIR"

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
CKPT="$WEIGHTS_DIR/model_ema_bf16.pt"
if [ -f "$CKPT" ]; then
    SIZE=$(du -sh "$CKPT" | cut -f1)
    echo ""
    echo "=== Success: $CKPT ($SIZE) ==="
else
    echo ""
    echo "ERROR: $CKPT not found after download."
    echo "       Check that the file exists in the HF repo and that your token"
    echo "       has read access.  You may also need to accept the license at:"
    echo "       https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator"
    exit 1
fi
