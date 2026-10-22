#!/bin/bash
#SBATCH --job-name=cosmos-suturebot
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/tjaspers/codex/logs/%x_%A_%a.out
#SBATCH --error=/home/tjaspers/codex/logs/%x_%A_%a.err
#
# Cosmos-H SutureBot full inference — all 146 test episodes (1306-1451),
# generating K rollout samples per episode for best-of-K IDM probing.
#
# Array layout: 15 tasks (one per episode slice of 10 episodes).
# Each task loads the model once and generates all K samples for its
# episodes sequentially.  GT data is loaded once per episode and reused
# across samples.
#
# Output structure (for IDM probe):
#   <out_root>/<embodiment>/predicted/episode_<id>/sample_<idx>.mp4
#   <out_root>/manifests/manifest_task_<TASK_ID>.json
#
# After all tasks complete, merge manifests:
#   jq -s 'add' <out_root>/manifests/manifest_task_*.json > <out_root>/manifest.json
#
# Overridable env vars (pass via --export or env before sbatch):
#   K             -- samples per episode  (default: 8)
#   BASE_SEED     -- base random seed     (default: 42)
#
# Timing estimate (H200, K=8):
#   ~2-5 min/chunk, ~5-10 chunks/episode, 8 samples → ~2-7 h/episode
#   10 episodes/task → fits within 24 h wall limit
#
# Prerequisites:
#   1. cosmos-h.sqsh at /storage/tjaspers/containers/cosmos-h.sqsh
#   2. Model weights at /home/tjaspers/codex/weights/model_ema_bf16.pt
#   3. Suturebot dataset at /nfs/data/public/Open-H/Surgical/jhu/imerse/suturebot/
#
# Usage:
#   mkdir -p ~/codex/logs
#   sbatch cluster/run_all_suturebot.sbatch                           # default K=8
#   K=4 sbatch cluster/run_all_suturebot.sbatch                       # override K
#   sbatch --array=0 cluster/run_all_suturebot.sbatch                 # first 10 episodes only
#
# For a quick smoke test (1 episode, 2 samples):
#   K=2 sbatch --array=0 cluster/run_all_suturebot.sbatch
#
set -euo pipefail

# ─── Best-of-K parameters ────────────────────────────────────────────────
K="${K:-8}"                  # number of rollout samples per episode
BASE_SEED="${BASE_SEED:-42}" # base random seed; sample i uses BASE_SEED + i*1000

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

echo "Array task ${TASK_ID}: K=${K}, episodes ${START_EP}-${END_EP} → ${EPISODE_IDS}, base_seed=${BASE_SEED}"

# ─── Paths ───────────────────────────────────────────────────────────────
SQSH_PATH="/storage/tjaspers/containers/cosmos-h.sqsh"
CONTAINER_NAME="cosmos-h-${TASK_ID}"     # per-task container name avoids race
WEIGHTS_DIR="/home/tjaspers/codex/weights"
OPENH_ROOT="/nfs/data/public/Open-H"
COSMOS_OUT="/storage/tjaspers/cosmos_output/suturebot_test"
REPO_DIR="${HOME}/Cosmos-H-Surgical-Simulator"
SCRIPTS_DIR="${REPO_DIR}/cluster/scripts"
HF_CACHE="${HOME}/.cache/huggingface"

# Per-task manifest path (merged after all tasks complete)
MANIFEST_DIR="${COSMOS_OUT}/manifests"
MANIFEST_PATH="${MANIFEST_DIR}/manifest_task_${TASK_ID}.json"

# Dataset inside container (OPENH_ROOT mounted at /data)
SUTUREBOT="/data/Surgical/jhu/imerse/suturebot"

# ─── Setup ───────────────────────────────────────────────────────────────
mkdir -p "${COSMOS_OUT}" "${HF_CACHE}" "${MANIFEST_DIR}"

echo "============================================================"
echo "COSMOS-H SUTUREBOT FULL INFERENCE (BEST-OF-K)"
echo "============================================================"
echo "Job:        ${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
echo "Node:       $(hostname)"
echo "K:          ${K}"
echo "Base seed:  ${BASE_SEED}"
echo "Episodes:   ${START_EP}-${END_EP}  (${EPISODE_IDS})"
echo "Container:  ${SQSH_PATH}"
echo "Weights:    ${WEIGHTS_DIR}/model_ema_bf16.pt"
echo "Output:     ${COSMOS_OUT}"
echo "Manifest:   ${MANIFEST_PATH}"
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
# Container-internal manifest path (COSMOS_OUT is mounted at /out)
CONTAINER_MANIFEST="/out/manifests/manifest_task_${TASK_ID}.json"

enroot start --rw \
    --mount "${OPENH_ROOT}:/data" \
    --mount "${WEIGHTS_DIR}:/weights" \
    --mount "${COSMOS_OUT}:/out" \
    --mount "${HF_CACHE}:/hf" \
    --mount "${SCRIPTS_DIR}:/scripts" \
    --mount "${REPO_DIR}/cosmos_predict2:/workspace/cosmos_predict2" \
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
    --env SEED="${BASE_SEED}" \
    --env NUM_SAMPLES="${K}" \
    --env SAVE_FPS="10" \
    --env SAVE_COMPARISON="0" \
    --env MANIFEST_PATH="${CONTAINER_MANIFEST}" \
    "${CONTAINER_NAME}" \
    bash /scripts/run_inference.sh

# ─── Check output ────────────────────────────────────────────────────────
PRED="${COSMOS_OUT}/jhu_dvrk_mono/predicted"
EXPECTED=0
FOUND=0
for (( ep=START_EP; ep<=END_EP; ep++ )); do
    for (( s=0; s<K; s++ )); do
        EXPECTED=$(( EXPECTED + 1 ))
        SAMPLE_FILE="${PRED}/episode_${ep}/sample_$(printf '%02d' ${s}).mp4"
        if [[ -f "${SAMPLE_FILE}" ]]; then
            FOUND=$(( FOUND + 1 ))
        else
            echo "MISSING: ${SAMPLE_FILE}" >&2
        fi
    done
done

echo ""
echo "Task ${TASK_ID}: ${FOUND}/${EXPECTED} videos completed (${K} samples × episodes ${START_EP}-${END_EP})"
find "${PRED}" -path "*/episode_*/sample_*.mp4" 2>/dev/null | sort | tail -20
if (( FOUND < EXPECTED )); then
    echo "WARN: $(( EXPECTED - FOUND )) video(s) missing — check logs" >&2
    exit 1
fi

# ─── Check manifest ─────────────────────────────────────────────────────
if [[ -f "${MANIFEST_PATH}" ]]; then
    MANIFEST_ROWS=$(python3 -c "import json; print(len(json.load(open('${MANIFEST_PATH}'))))" 2>/dev/null || echo 0)
    echo "Manifest: ${MANIFEST_ROWS} rows in ${MANIFEST_PATH}"
else
    echo "WARN: Manifest not found at ${MANIFEST_PATH}" >&2
fi

echo "Task ${TASK_ID} DONE"
