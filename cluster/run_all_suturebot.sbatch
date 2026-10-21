#!/bin/bash
#SBATCH --job-name=cosmos-suturebot
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/tjaspers/codex/logs/%x_%A_%a.out
#SBATCH --error=/home/tjaspers/codex/logs/%x_%A_%a.err
#
# Cosmos-H SutureBot full inference — all 146 test episodes (1306-1451).
#
# Uses a Slurm array (15 tasks × 10 episodes each) to parallelise across GPUs.
# Each array task loads the model once and processes its slice of episodes
# sequentially.  All tasks write to a shared output directory; episode files
# are named episode_XXXX.mp4 so there are no collisions.
#
# Timing estimate (H200):
#   ~2-5 min/chunk, ~5-10 chunks/episode → ~15-50 min/episode
#   10 episodes/task → ~2.5-8 h/task (8 h wall limit)
#
# Prerequisites (same as smoke test):
#   1. cosmos-h.sqsh at /storage/tjaspers/containers/cosmos-h.sqsh
#   2. Model weights at /home/tjaspers/codex/weights/model_ema_bf16.pt
#   3. Suturebot dataset at /nfs/data/public/Open-H/Surgical/jhu/imerse/suturebot/
#
# Usage:
#   mkdir -p ~/codex/logs
#   sbatch cluster/run_all_suturebot.sbatch
#
# To run a subset, override the array range:
#   sbatch --array=0-4 cluster/run_all_suturebot.sbatch   # first 50 episodes
#
set -euo pipefail

# ─── Episode slicing ─────────────────────────────────────────────────────
FIRST_EP=1306          # first test episode (codex catalogue)
LAST_EP=1451           # last test episode  (inclusive)
EPS_PER_TASK=10        # episodes per array task

TASK_ID="${SLURM_ARRAY_TASK_ID}"
START_EP=$(( FIRST_EP + TASK_ID * EPS_PER_TASK ))
END_EP=$(( START_EP + EPS_PER_TASK - 1 ))
# Clamp to last episode
if (( END_EP > LAST_EP )); then
    END_EP=${LAST_EP}
fi

# Build comma-separated list: 1306,1307,...,1315
EPISODE_IDS=""
for (( ep=START_EP; ep<=END_EP; ep++ )); do
    EPISODE_IDS="${EPISODE_IDS:+${EPISODE_IDS},}${ep}"
done

echo "Array task ${TASK_ID}: episodes ${START_EP}-${END_EP} → ${EPISODE_IDS}"

# ─── Paths ───────────────────────────────────────────────────────────────
SQSH_PATH="/storage/tjaspers/containers/cosmos-h.sqsh"
CONTAINER_NAME="cosmos-h-${TASK_ID}"     # per-task container name avoids race
WEIGHTS_DIR="/home/tjaspers/codex/weights"
OPENH_ROOT="/nfs/data/public/Open-H"
COSMOS_OUT="/storage/tjaspers/cosmos_output/suturebot_test"
SCRIPTS_DIR="${HOME}/Cosmos-H-Surgical-Simulator/cluster/scripts"
HF_CACHE="${HOME}/.cache/huggingface"

# Dataset inside container (OPENH_ROOT mounted at /data)
SUTUREBOT="/data/Surgical/jhu/imerse/suturebot"

# ─── Setup ───────────────────────────────────────────────────────────────
mkdir -p "${COSMOS_OUT}" "${HF_CACHE}"

echo "============================================================"
echo "COSMOS-H SUTUREBOT FULL INFERENCE"
echo "============================================================"
echo "Job:        ${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
echo "Node:       $(hostname)"
echo "Episodes:   ${START_EP}-${END_EP}  (${EPISODE_IDS})"
echo "Container:  ${SQSH_PATH}"
echo "Weights:    ${WEIGHTS_DIR}/model_ema_bf16.pt"
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
# Use per-task container names so parallel tasks don't conflict during create.
if ! enroot list 2>/dev/null | grep -qx "${CONTAINER_NAME}"; then
    echo "Creating enroot container '${CONTAINER_NAME}' ..."
    enroot create --name "${CONTAINER_NAME}" "${SQSH_PATH}"
fi

# ─── Run inference ───────────────────────────────────────────────────────
enroot start --rw \
    --mount "${OPENH_ROOT}:/data" \
    --mount "${WEIGHTS_DIR}:/weights" \
    --mount "${COSMOS_OUT}:/out" \
    --mount "${HF_CACHE}:/hf" \
    --mount "${SCRIPTS_DIR}:/scripts" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env HF_HOME=/hf \
    --env PYTHONPATH=/workspace \
    --env PATH=/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin \
    --env EMBODIMENTS="jhu_dvrk_mono:${SUTUREBOT}" \
    --env EPISODE_IDS="${EPISODE_IDS}" \
    --env DATA_SPLIT="full" \
    --env MAX_CHUNKS="20" \
    --env GUIDANCE="0" \
    --env SEED="42" \
    --env SAVE_FPS="10" \
    --env SAVE_COMPARISON="0" \
    "${CONTAINER_NAME}" \
    bash /scripts/run_inference.sh

# ─── Check output ────────────────────────────────────────────────────────
PRED="${COSMOS_OUT}/jhu_dvrk_mono/predicted"
EXPECTED=0
FOUND=0
for (( ep=START_EP; ep<=END_EP; ep++ )); do
    EXPECTED=$(( EXPECTED + 1 ))
    if [[ -f "${PRED}/episode_${ep}.mp4" ]]; then
        FOUND=$(( FOUND + 1 ))
    else
        echo "MISSING: ${PRED}/episode_${ep}.mp4" >&2
    fi
done

echo ""
echo "Task ${TASK_ID}: ${FOUND}/${EXPECTED} episodes completed"
ls -lh "${PRED}"/episode_*.mp4 2>/dev/null | tail -20
if (( FOUND < EXPECTED )); then
    echo "WARN: ${EXPECTED - FOUND} episode(s) missing — check logs" >&2
    exit 1
fi
echo "Task ${TASK_ID} DONE"
