#!/usr/bin/env bash
#SBATCH --job-name=cosmos-h-infer
#SBATCH --partition=large             # H200 partition (also: express for 4h max)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1                # 1x H200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cosmos-infer-%j.out
#SBATCH --error=logs/cosmos-infer-%j.err
# ==========================================================================
# run_cosmos.sbatch -- Inference via enroot create/start (no pyxis, no Docker).
#
# The cluster has enroot 3.5.0 only.  This sbatch:
#   1. Creates the container from the .sqsh (if not already created)
#   2. Starts it with bind mounts and runs the inference wrapper
#
# Usage:
#   mkdir -p logs
#   export SQSH_DIR=/scratch/containers
#   export OPENH_ROOT=/data/openh
#   export WEIGHTS_DIR=/data/weights
#   export COSMOS_OUT=/scratch/output/cosmos
#   export SCRIPTS_DIR=$(pwd)/cluster/scripts
#   export EMBODIMENTS="jhu_dvrk_mono:/data/jhu/suturebot_2"
#   export HF_TOKEN="hf_..."
#   sbatch cluster/run_cosmos.sbatch
# ==========================================================================

set -euo pipefail

# ==========================================================================
# Required paths — set as env vars before sbatch, or edit defaults here.
# ==========================================================================
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to the directory containing cosmos-h.sqsh}"
OPENH_ROOT="${OPENH_ROOT:?Set OPENH_ROOT to the Open-H datasets root on cluster storage}"
WEIGHTS_DIR="${WEIGHTS_DIR:?Set WEIGHTS_DIR to the directory containing model_ema_bf16.pt}"
COSMOS_OUT="${COSMOS_OUT:?Set COSMOS_OUT to the output directory for generated mp4s}"
SCRIPTS_DIR="${SCRIPTS_DIR:?Set SCRIPTS_DIR to the directory containing run_inference.sh}"

SQSH_PATH="${SQSH_DIR}/cosmos-h.sqsh"
CONTAINER_NAME="cosmos-h"

# HuggingFace
HF_TOKEN="${HF_TOKEN:-}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

# ==========================================================================
# Inference parameters — override via env before sbatch.
# ==========================================================================
# EMBODIMENTS: space-separated "embodiment:dataset_path" pairs.
# dataset_path is the CONTAINER-SIDE path (under /data, the mount of OPENH_ROOT).
EMBODIMENTS="${EMBODIMENTS:?Set EMBODIMENTS as 'embodiment:/data/sub/path' pairs}"
EPISODE_IDS="${EPISODE_IDS:-0,1,2}"
DATA_SPLIT="${DATA_SPLIT:-test}"
GUIDANCE="${GUIDANCE:-0}"
SEED="${SEED:-42}"
SAVE_FPS="${SAVE_FPS:-10}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
SAVE_COMPARISON="${SAVE_COMPARISON:-0}"

# ==========================================================================
# Prepare directories
# ==========================================================================
mkdir -p "$COSMOS_OUT"
mkdir -p "$HF_CACHE"

# ==========================================================================
# Diagnostics
# ==========================================================================
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Node:         $(hostname)"
echo "Container:    $SQSH_PATH"
echo "Weights:      $WEIGHTS_DIR"
echo "Open-H root:  $OPENH_ROOT"
echo "Output:       $COSMOS_OUT"
echo "Scripts:      $SCRIPTS_DIR"
echo "Embodiments:  $EMBODIMENTS"
echo "Episodes:     $EPISODE_IDS"
echo "Split:        $DATA_SPLIT"
echo "============================================================"
nvidia-smi 2>/dev/null || true
echo ""

# ==========================================================================
# Verify prerequisites
# ==========================================================================
if [ ! -f "$SQSH_PATH" ]; then
    echo "ERROR: $SQSH_PATH not found.  Run import_sqsh.sh first."
    exit 1
fi

if [ ! -f "${WEIGHTS_DIR}/model_ema_bf16.pt" ]; then
    echo "ERROR: ${WEIGHTS_DIR}/model_ema_bf16.pt not found.  Run fetch_weights.sh first."
    exit 1
fi

if [ ! -f "${SCRIPTS_DIR}/run_inference.sh" ]; then
    echo "ERROR: ${SCRIPTS_DIR}/run_inference.sh not found."
    exit 1
fi

# ==========================================================================
# STEP 3: Create enroot container (if not already created on this node)
# ==========================================================================
# enroot containers are per-node (stored in $ENROOT_DATA_PATH or ~/.local/share/enroot).
# If the job lands on a different node, the container won't exist yet.
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
    echo "--- Creating enroot container '$CONTAINER_NAME' from $SQSH_PATH ---"
    enroot create --name "$CONTAINER_NAME" "$SQSH_PATH"
else
    echo "--- enroot container '$CONTAINER_NAME' already exists on $(hostname) ---"
fi
echo ""

# ==========================================================================
# STEP 4: Run inference via enroot start
# ==========================================================================
# --rw:           writable overlay (needed for /tmp, pip cache, etc.)
# --mount:        bind mounts (host:container)
# NVIDIA hook:    enroot on DGX auto-exposes GPUs via libnvidia-container
#                 (no --nv flag; the hook is configured system-wide)
#
# Environment:
#   HF_HOME=/hf          redirect HF cache to the writable mount
#   HF_TOKEN              for gated model access
#   PYTHONPATH=/workspace  repo root for Python imports
#   All inference params are forwarded via the env block below.
#
# NOTE: Do NOT use --root.  enroot runs as the calling user by default.
echo "--- Starting enroot container ---"
enroot start --rw \
    --mount "${OPENH_ROOT}:/data" \
    --mount "${WEIGHTS_DIR}:/weights" \
    --mount "${COSMOS_OUT}:/out" \
    --mount "${HF_CACHE}:/hf" \
    --mount "${SCRIPTS_DIR}:/scripts" \
    --env HF_HOME=/hf \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env PYTHONPATH=/workspace \
    --env EMBODIMENTS="${EMBODIMENTS}" \
    --env EPISODE_IDS="${EPISODE_IDS}" \
    --env DATA_SPLIT="${DATA_SPLIT}" \
    --env GUIDANCE="${GUIDANCE}" \
    --env SEED="${SEED}" \
    --env SAVE_FPS="${SAVE_FPS}" \
    --env MAX_CHUNKS="${MAX_CHUNKS}" \
    --env SAVE_COMPARISON="${SAVE_COMPARISON}" \
    "$CONTAINER_NAME" \
    bash /scripts/run_inference.sh

echo ""
echo "=== Slurm job ${SLURM_JOB_ID:-N/A} finished ==="
echo "Output written to: $COSMOS_OUT"
echo "Predicted videos:  $COSMOS_OUT/<embodiment>/predicted/episode_XXXX.mp4"
