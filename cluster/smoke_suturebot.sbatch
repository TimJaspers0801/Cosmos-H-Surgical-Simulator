#!/bin/bash
#SBATCH --job-name=cosmos-smoke
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --time=0:30:00
#SBATCH --output=/home/tjaspers/codex/logs/%x_%j.out
#SBATCH --error=/home/tjaspers/codex/logs/%x_%j.err
#
# Cosmos-H SutureBot smoke test — 1 test episode, 1 chunk (13 frames).
#
# Prerequisites:
#   1. cosmos-h.sqsh at /storage/tjaspers/containers/cosmos-h.sqsh
#      (built via: TAR_PATH=... SQSH_DIR=/storage/tjaspers/containers bash cluster/import_sqsh.sh)
#   2. Model weights at /home/tjaspers/codex/weights/model_ema_bf16.pt
#   3. Suturebot dataset at /nfs/data/public/Open-H/Surgical/jhu/imerse/suturebot/
#      with meta/stats_cosmos.json present
#
# Usage:
#   mkdir -p ~/codex/logs
#   sbatch cluster/smoke_suturebot.sbatch
#
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────
SQSH_PATH="/storage/tjaspers/containers/cosmos-h.sqsh"
CONTAINER_NAME="cosmos-h"
WEIGHTS_DIR="/home/tjaspers/codex/weights"
OPENH_ROOT="/nfs/data/public/Open-H"
COSMOS_OUT="/storage/tjaspers/cosmos_output/smoke"
SCRIPTS_DIR="${HOME}/Cosmos-H-Surgical-Simulator/cluster/scripts"
HF_CACHE="${HOME}/.cache/huggingface"

# Dataset inside container (OPENH_ROOT is mounted at /data)
SUTUREBOT="/data/Surgical/jhu/imerse/suturebot"

# ─── Setup ───────────────────────────────────────────────────────────────
mkdir -p "${COSMOS_OUT}" "${HF_CACHE}"

echo "============================================================"
echo "COSMOS-H SMOKE TEST — SutureBot ep 1306, 1 chunk"
echo "============================================================"
echo "Node:       $(hostname)"
echo "Container:  ${SQSH_PATH}"
echo "Weights:    ${WEIGHTS_DIR}/model_ema_bf16.pt"
echo "Dataset:    ${OPENH_ROOT}/Surgical/jhu/imerse/suturebot"
echo "Output:     ${COSMOS_OUT}"
echo "============================================================"
nvidia-smi 2>/dev/null | head -5 || true
echo ""

# ─── Validate ────────────────────────────────────────────────────────────
FATAL=0
for f in "${SQSH_PATH}" "${WEIGHTS_DIR}/model_ema_bf16.pt" "${SCRIPTS_DIR}/run_inference.sh"; do
    [[ ! -e "$f" ]] && echo "FATAL: not found: $f" >&2 && FATAL=1
done
[[ ! -d "${OPENH_ROOT}/Surgical/jhu/imerse/suturebot/meta" ]] && echo "FATAL: dataset missing" >&2 && FATAL=1
[[ "${FATAL}" -eq 1 ]] && exit 1

# ─── Create enroot container if needed ───────────────────────────────────
if ! enroot list 2>/dev/null | grep -qx "${CONTAINER_NAME}"; then
    echo "Creating enroot container from ${SQSH_PATH} ..."
    enroot create --name "${CONTAINER_NAME}" "${SQSH_PATH}"
fi

# ─── Run inference ───────────────────────────────────────────────────────
enroot start --rw \
    --mount "${OPENH_ROOT}:/data" \
    --mount "${WEIGHTS_DIR}:/weights" \
    --mount "${COSMOS_OUT}:/out" \
    --mount "${HF_CACHE}:/hf" \
    --mount "${SCRIPTS_DIR}:/scripts" \
    --env HF_HOME=/hf \
    --env PYTHONPATH=/workspace \
    --env EMBODIMENTS="jhu_dvrk_mono:${SUTUREBOT}" \
    --env EPISODE_IDS="1306" \
    --env DATA_SPLIT="full" \
    --env MAX_CHUNKS="1" \
    --env GUIDANCE="0" \
    --env SEED="42" \
    --env SAVE_FPS="10" \
    --env SAVE_COMPARISON="0" \
    "${CONTAINER_NAME}" \
    bash /scripts/run_inference.sh

# ─── Check output ────────────────────────────────────────────────────────
PRED="${COSMOS_OUT}/jhu_dvrk_mono/predicted"
if [[ -f "${PRED}/episode_1306.mp4" ]]; then
    echo ""
    echo "SMOKE TEST PASSED"
    ls -lh "${PRED}"/episode_1306.mp4
else
    echo "SMOKE TEST FAILED: ${PRED}/episode_1306.mp4 not found" >&2
    exit 1
fi
