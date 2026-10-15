#!/usr/bin/env bash
#SBATCH --job-name=cosmos-smoke
#SBATCH --partition=gpu              # <-- adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                     # 1x H200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00              # smoke test should be quick
#SBATCH --output=logs/cosmos-smoke-%j.out
#SBATCH --error=logs/cosmos-smoke-%j.err
# ==========================================================================
# smoke_test.sbatch -- Minimal inference run to verify the full pipeline.
#
# One embodiment, one episode, one chunk.  Must produce a real .mp4 or fail.
# Run this BEFORE any full inference job.
#
# Usage:
#   mkdir -p logs
#   export SQSH_DIR=/scratch/containers
#   export OPENH_ROOT=/data/openh
#   export WEIGHTS_DIR=/data/weights
#   export COSMOS_OUT=/scratch/output/smoke_test
#   export SCRIPTS_DIR=$(pwd)/cluster/scripts
#   export HF_TOKEN="hf_..."
#   sbatch cluster/smoke_test.sbatch
# ==========================================================================

set -euo pipefail

# ==========================================================================
# Paths (same as run_cosmos.sbatch)
# ==========================================================================
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to the directory containing cosmos-h.sqsh}"
OPENH_ROOT="${OPENH_ROOT:?Set OPENH_ROOT to the Open-H datasets root}"
WEIGHTS_DIR="${WEIGHTS_DIR:?Set WEIGHTS_DIR to the directory containing model_ema_bf16.pt}"
COSMOS_OUT="${COSMOS_OUT:?Set COSMOS_OUT to the smoke test output directory}"
SCRIPTS_DIR="${SCRIPTS_DIR:?Set SCRIPTS_DIR to the directory containing run_inference.sh}"

SQSH_PATH="${SQSH_DIR}/cosmos-h.sqsh"
CONTAINER_NAME="cosmos-h"
HF_TOKEN="${HF_TOKEN:-}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

# ==========================================================================
# Smoke test parameters — single embodiment, single episode, single chunk.
# ==========================================================================
# Using jhu_dvrk_mono as the default because it has simpler meta requirements
# (no clutch filter cache needed, unlike cmr_versius).
# Adjust the dataset path to match your cluster layout.
EMBODIMENTS="${EMBODIMENTS:-jhu_dvrk_mono:/data/jhu/suturebot_2}"
EPISODE_IDS="0"
DATA_SPLIT="${DATA_SPLIT:-test}"
MAX_CHUNKS="1"
GUIDANCE="0"
SEED="42"
SAVE_FPS="10"
SAVE_COMPARISON="0"

mkdir -p "$COSMOS_OUT"
mkdir -p "$HF_CACHE"

echo "============================================================"
echo "SMOKE TEST — Cosmos-H-Surgical-Simulator"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Node:        $(hostname)"
echo "Embodiments: $EMBODIMENTS"
echo "Episode:     $EPISODE_IDS  (single)"
echo "Max chunks:  $MAX_CHUNKS   (single)"
echo "Output:      $COSMOS_OUT"
echo "============================================================"
nvidia-smi 2>/dev/null | head -5 || true
echo ""

# Verify prerequisites
for f in "$SQSH_PATH" "${WEIGHTS_DIR}/model_ema_bf16.pt" "${SCRIPTS_DIR}/run_inference.sh"; do
    if [ ! -e "$f" ]; then echo "ERROR: $f not found"; exit 1; fi
done

# Create container if needed
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
    echo "--- Creating enroot container ---"
    enroot create --name "$CONTAINER_NAME" "$SQSH_PATH"
fi

# Run
echo "--- Running smoke test ---"
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

# ==========================================================================
# Post-run verification
# ==========================================================================
echo ""
echo "============================================================"
echo "SMOKE TEST POST-RUN CHECK"
echo "============================================================"

# Extract the embodiment name from the first pair
EMB="${EMBODIMENTS%% *}"
EMB="${EMB%%:*}"
PRED_DIR="${COSMOS_OUT}/${EMB}/predicted"

echo "Checking: $PRED_DIR"

if [ -d "$PRED_DIR" ]; then
    MP4_COUNT=$(find "$PRED_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
    if [ "$MP4_COUNT" -gt 0 ]; then
        echo ""
        echo "SMOKE TEST PASSED: $MP4_COUNT mp4 file(s) found"
        ls -lh "$PRED_DIR"/*.mp4
        echo ""
        echo "Resolved output path: $PRED_DIR/episode_0000.mp4"

        # Quick sanity: file must be > 1KB (not just a header)
        SIZE=$(stat --format="%s" "$PRED_DIR/episode_0000.mp4" 2>/dev/null || stat -f "%z" "$PRED_DIR/episode_0000.mp4" 2>/dev/null || echo "0")
        if [ "$SIZE" -lt 1024 ]; then
            echo "WARNING: episode_0000.mp4 is only $SIZE bytes — may be corrupt."
        fi
    else
        echo ""
        echo "SMOKE TEST FAILED: No .mp4 files in $PRED_DIR"
        echo "Check logs above.  Common causes:"
        echo "  - Episode 0 doesn't exist in the '$DATA_SPLIT' split"
        echo "    → try DATA_SPLIT=full"
        echo "  - Missing meta files (stats_cosmos.json, modality.json)"
        exit 1
    fi
else
    echo ""
    echo "SMOKE TEST FAILED: $PRED_DIR does not exist"
    echo "The inference script may have crashed before writing output."
    exit 1
fi
