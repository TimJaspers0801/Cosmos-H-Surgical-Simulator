#!/usr/bin/env bash
# run_inference.sh -- In-container wrapper for Cosmos-H-Surgical-Simulator inference.
#
# Runs INSIDE the container (called by run_cosmos.sbatch via enroot start).
# Loops over embodiment/dataset pairs in $EMBODIMENTS and calls
# inference_open_h.py for each one.
#
# Mount layout (set by the sbatch):
#   /data      <- $OPENH_ROOT  (read-only)   Open-H datasets in LeRobot layout
#   /weights   <- $WEIGHTS_DIR (read-only)   model_ema_bf16.pt
#   /out       <- $COSMOS_OUT  (read-write)  generated mp4s
#   /hf        <- $HF_CACHE   (read-write)  HF hub cache
#
# Required env vars:
#   EMBODIMENTS   -- space-separated "embodiment:dataset_path" pairs
#                    dataset_path must be an absolute container path (e.g. /data/...)
#
# Optional env vars:
#   EPISODE_IDS     -- comma-separated episode indices  (default: "0,1,2")
#   DATA_SPLIT      -- train / test / full              (default: "test")
#   GUIDANCE        -- classifier-free guidance scale   (default: 0)
#   SEED            -- random seed                      (default: 42)
#   SAVE_FPS        -- output video FPS                 (default: 10)
#   MAX_CHUNKS      -- max autoregressive chunks/ep     (default: 20)
#   SAVE_COMPARISON -- set to "1" for GT-vs-pred video  (default: 0)
#   EXPERIMENT      -- experiment config name           (default: release config)
#   EXCLUDE_SPLITS_<EMBODIMENT> -- per-embodiment split exclusions
set -euo pipefail

# ===========================================================================
# Defaults
# ===========================================================================
EXPERIMENT="${EXPERIMENT:-cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss}"

# Checkpoint: MUST be a .pt file.
# model_loader.py uses ckpt_path.endswith(".pt") to select single-file loading.
# A .safetensors path gets misrouted to the DCP loader and fails.
CKPT_PATH="/weights/model_ema_bf16.pt"

DATA_SPLIT="${DATA_SPLIT:-test}"
EPISODE_IDS="${EPISODE_IDS:-0,1,2}"
GUIDANCE="${GUIDANCE:-0}"
SEED="${SEED:-42}"
SAVE_FPS="${SAVE_FPS:-10}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
SAVE_COMPARISON="${SAVE_COMPARISON:-0}"

# PYTHONPATH must include the repo root (/workspace in STANDALONE image).
export PYTHONPATH="${PYTHONPATH:-/workspace}"

# ===========================================================================
# Validate
# ===========================================================================
if [ -z "${EMBODIMENTS:-}" ]; then
    cat <<'EOF'
ERROR: EMBODIMENTS is not set.
Set it as space-separated 'embodiment:dataset_path' pairs, e.g.:
  EMBODIMENTS="jhu_dvrk_mono:/data/jhu/suturebot_2 cmr_versius:/data/cmr-surgical-60hz-fixed/cholecystectomy_360p"

Dataset paths are under /data (the mount of $OPENH_ROOT).

Required meta files per embodiment:
  jhu_dvrk_mono (and other Open-H):
    meta/modality.json  +  meta/stats_cosmos.json
  cmr_versius:
    meta/modality-44D.json  +  meta/stats_cosmos-44D.json
    meta/cmr_filter_cache_<split>_<hash>-44D.json  (clutch filter cache)
    Generate cache: python scripts/compute_cmr_filtered_episodes_cache.py \
        --dataset-path <path> --split <split> --num-frames 12
EOF
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    echo "       Is \$WEIGHTS_DIR mounted at /weights with model_ema_bf16.pt?"
    exit 1
fi

# ===========================================================================
# Patch tokenizer to use local weights instead of HF download
# ===========================================================================
# The HF repo nvidia/Cosmos-Predict2.5-2B no longer has tokenizer.pth at the
# expected revision, causing a 404.  If we have tokenizer.pth mounted locally
# (via /weights), rewrite the S3 URI defaults in the tokenizer source so that
# get_checkpoint_path() treats it as a local file and skips the HF download.
TOKENIZER_LOCAL="${TOKENIZER_PATH:-/weights/tokenizer.pth}"
if [ -f "$TOKENIZER_LOCAL" ]; then
    echo "Patching tokenizer defaults to use local: $TOKENIZER_LOCAL"
    WAN1_S3="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth"
    sed -i "s|${WAN1_S3}|${TOKENIZER_LOCAL}|g" \
        /workspace/cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py \
        /workspace/cosmos_predict2/_src/predict2/tokenizers/cosmos.py \
        2>/dev/null || true
fi

# ===========================================================================
# Build optional flags
# ===========================================================================
COMPARISON_FLAG=""
if [ "${SAVE_COMPARISON}" = "1" ]; then
    COMPARISON_FLAG="--save_comparison"
fi

echo "=== Cosmos-H-Surgical-Simulator Inference ==="
echo "Checkpoint:  $CKPT_PATH"
echo "Output root: /out"
echo "Experiment:  $EXPERIMENT"
echo "Data split:  $DATA_SPLIT"
echo "Episode IDs: $EPISODE_IDS"
echo "Guidance:    $GUIDANCE"
echo "Seed:        $SEED"
echo "Max chunks:  $MAX_CHUNKS"
echo "Save FPS:    $SAVE_FPS"
echo ""

# ===========================================================================
# Loop over embodiments
# ===========================================================================
FAILED=0

for PAIR in $EMBODIMENTS; do
    EMBODIMENT="${PAIR%%:*}"
    DATASET_PATH="${PAIR#*:}"

    echo "============================================================"
    echo "Embodiment:   $EMBODIMENT"
    echo "Dataset path: $DATASET_PATH"
    echo "============================================================"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Dataset path does not exist inside the container: $DATASET_PATH"
        echo "       Check that OPENH_ROOT is mounted at /data and the sub-path is correct."
        FAILED=1
        continue
    fi

    # Per-embodiment --exclude_splits (e.g. EXCLUDE_SPLITS_dvrk_stanford_real="fail bad_frames")
    EXCLUDE_VAR="EXCLUDE_SPLITS_${EMBODIMENT}"
    EXCLUDE_FLAG=""
    if [ -n "${!EXCLUDE_VAR:-}" ]; then
        EXCLUDE_FLAG="--exclude_splits ${!EXCLUDE_VAR}"
    fi

    # inference_open_h.py writes to <save_root>/<embodiment>/predicted/episode_XXXX.mp4
    # We pass /out as --save_root so output lands at /out/<embodiment>/predicted/
    echo ""
    echo "Running: python3 inference_open_h.py --embodiment $EMBODIMENT --save_root /out ..."
    python3 /workspace/cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
        --experiment "$EXPERIMENT" \
        --ckpt_path "$CKPT_PATH" \
        --dataset_path "$DATASET_PATH" \
        --embodiment "$EMBODIMENT" \
        --data_split "$DATA_SPLIT" \
        --episode_ids "$EPISODE_IDS" \
        --guidance "$GUIDANCE" \
        --seed "$SEED" \
        --save_root /out \
        --save_fps "$SAVE_FPS" \
        --max_chunks "$MAX_CHUNKS" \
        $COMPARISON_FLAG \
        $EXCLUDE_FLAG

    # ---------------------------------------------------------------
    # Verify output was written
    # ---------------------------------------------------------------
    PRED_DIR="/out/${EMBODIMENT}/predicted"
    MP4_COUNT=$(find "$PRED_DIR" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
    echo ""
    echo "Output directory: $PRED_DIR"
    if [ "$MP4_COUNT" -eq 0 ]; then
        echo "FAIL: No .mp4 files found in $PRED_DIR after inference."
        echo "      The requested episode may not exist in the '${DATA_SPLIT}' split."
        echo "      Check the log above for 'Episode X not found' or 'doesn't start at base_index=0'."
        FAILED=1
    else
        echo "OK: $MP4_COUNT mp4 file(s) written to $PRED_DIR"
        ls -lh "$PRED_DIR"/*.mp4
    fi
    echo ""
done

if [ "$FAILED" -ne 0 ]; then
    echo "=== SOME EMBODIMENTS FAILED — see errors above ==="
    exit 1
fi

echo "=== All inference runs completed successfully ==="
echo "Results at: /out/<embodiment>/predicted/episode_XXXX.mp4"
