# Cluster Inference Pipeline — Cosmos-H-Surgical-Simulator

Run action-conditioned surgical video generation on H200 GPUs via **enroot** (no
Docker, no Apptainer on the cluster).

## Cluster Environment

| Component | Value |
|-----------|-------|
| GPUs | H200 (Hopper, sm_90) |
| CUDA | 12.8.1 (baked into the container) |
| Driver | >= 570.124.06 |
| Container runtime | **enroot 3.5.0** (`enroot create` + `enroot start`) |
| Docker / Apptainer | **NOT available** on the cluster |
| Pyxis | Available but we use raw enroot |
| Build host | Laptop (Docker rootful, Apple Silicon → cross-build linux/amd64) |

## Corrections vs. Task Assumptions

| Assumption | Reality |
|------------|---------|
| Checkpoint is `.safetensors` (fallback `.pt`) | **Must be `.pt`.** `model_loader.py` line 205 checks `ckpt_path.endswith(".pt")` for single-file loading. `.safetensors` gets misrouted to the DCP loader and **fails**. |
| Dockerfile has `APP_UID`/`APP_GID` args | **No.** No such args exist. Container runs as root; enroot maps the job's UID. |
| `--save_root /out/<embodiment>` | **Wrong — would double-nest.** The code does `save_root = os.path.join(args.save_root, embodiment)` (line 272), so pass `--save_root /out` and output lands at `/out/<embodiment>/predicted/`. |
| CMR clutch cache only for training | **No.** `_all_steps` is populated at dataset init even during inference, triggering `_get_all_steps_cmr_filtered()` which requires the pre-computed cache. |

---

## Environment Variables

### Required (set before `sbatch`)

| Variable | Description |
|----------|-------------|
| `SQSH_DIR` | Directory containing `cosmos-h.sqsh` (shared cluster storage) |
| `OPENH_ROOT` | Root of Open-H datasets on cluster storage |
| `WEIGHTS_DIR` | Directory containing `model_ema_bf16.pt` |
| `COSMOS_OUT` | Output directory for generated mp4s |
| `SCRIPTS_DIR` | Directory containing `run_inference.sh` (e.g. `$(pwd)/cluster/scripts`) |
| `EMBODIMENTS` | Space-separated `embodiment:/data/sub/path` pairs (container paths) |
| `HF_TOKEN` | HuggingFace token (for weight download; optional at inference if model is cached) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `EPISODE_IDS` | `0,1,2` | Comma-separated episode indices |
| `DATA_SPLIT` | `test` | `train` / `test` / `full` |
| `GUIDANCE` | `0` | Classifier-free guidance scale |
| `SEED` | `42` | Random seed |
| `SAVE_FPS` | `10` | Output video FPS |
| `MAX_CHUNKS` | `20` | Max autoregressive chunks per episode |
| `SAVE_COMPARISON` | `0` | Set to `1` for side-by-side GT vs predicted |
| `EXCLUDE_SPLITS_<emb>` | *(none)* | Per-embodiment split exclusions |
| `HF_CACHE` | `~/.cache/huggingface` | HF hub cache directory |

---

## End-to-End Sequence

### Step 0 — Verify the cluster environment

```bash
ssh <cluster>
cd Cosmos-H-Surgical-Simulator
bash cluster/step0_verify.sh
# Copy output and send it back.
```

### Step 1 — Convert Docker tar → enroot .sqsh (login node, once)

The `docker save` tar is already on the cluster at `~/docker-images/<IMAGE>.tar`.

```bash
export TAR_PATH=~/docker-images/<IMAGE>.tar   # exact filename from step0
export SQSH_DIR=/scratch/containers            # shared storage
bash cluster/import_sqsh.sh
```

This tries (in order):
1. `enroot import docker-archive://<tar>` (if enroot supports it)
2. `skopeo` → OCI layout → `enroot import`
3. **Stops with fallback instructions** if neither works

Verify: `ls -lh ${SQSH_DIR}/cosmos-h.sqsh`

### Step 2 — Download model weights (login node, once)

```bash
export HF_TOKEN="hf_..."
bash cluster/fetch_weights.sh /data/weights
```

Downloads `model_ema_bf16.pt` (~5 GB). You must accept the
[NVIDIA license](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator) first.

Verify: `ls -lh /data/weights/model_ema_bf16.pt`

### Step 3 — Smoke test (one episode, one chunk)

```bash
mkdir -p logs

export SQSH_DIR=/scratch/containers
export OPENH_ROOT=/data/openh
export WEIGHTS_DIR=/data/weights
export COSMOS_OUT=/scratch/output/smoke_test
export SCRIPTS_DIR=$(pwd)/cluster/scripts
export HF_TOKEN="hf_..."

# Use jhu_dvrk_mono (simpler meta requirements, no clutch cache needed):
export EMBODIMENTS="jhu_dvrk_mono:/data/jhu/suturebot_2"

sbatch cluster/smoke_test.sbatch
```

Check the log: `cat logs/cosmos-smoke-<jobid>.out`

Expected output: `$COSMOS_OUT/jhu_dvrk_mono/predicted/episode_0000.mp4`
(13 frames: 1 context + 12 predicted).

The script fails loudly if no mp4 is produced. If episode 0 doesn't exist in
the `test` split, try `export DATA_SPLIT=full`.

### Step 4 — Full inference run

```bash
export COSMOS_OUT=/scratch/output/cosmos
export EMBODIMENTS="jhu_dvrk_mono:/data/jhu/suturebot_2 cmr_versius:/data/cmr-surgical-60hz-fixed/cholecystectomy_360p"
export EPISODE_IDS="0,1,2,3,4"

sbatch cluster/run_cosmos.sbatch
```

---

## Dataset Requirements

Each dataset directory must be in LeRobot format with these meta files:

| Embodiment | Required meta files |
|------------|-------------------|
| `jhu_dvrk_mono` (and other Open-H) | `meta/modality.json` + `meta/stats_cosmos.json` |
| `cmr_versius` | `meta/modality-44D.json` + `meta/stats_cosmos-44D.json` + `meta/cmr_filter_cache_<split>_<hash>-44D.json` |

Generate stats/cache files:
```bash
# Open-H (non-CMR):
python scripts/compute_openh_action_stats.py

# CMR Versius:
python scripts/compute_cmr_action_stats.py --dataset-path <path>
python scripts/compute_cmr_filtered_episodes_cache.py \
    --dataset-path <path> --split <split> --num-frames 12
```

## Output Layout

```
$COSMOS_OUT/
  <embodiment>/
    predicted/
      episode_0000.mp4
      episode_0001.mp4
      ...
    comparison/              # only if SAVE_COMPARISON=1
      episode_0000.mp4
```

The benchmark reads mp4s from `$COSMOS_OUT/<embodiment>/predicted/`.

## Container Mount Layout

| Host path | Container path | Mode |
|-----------|---------------|------|
| `$OPENH_ROOT` | `/data` | read-only |
| `$WEIGHTS_DIR` | `/weights` | read-only |
| `$COSMOS_OUT` | `/out` | read-write |
| `$HF_CACHE` | `/hf` | read-write |
| `$SCRIPTS_DIR` | `/scripts` | read-only |

## Confirmed Argparse Flags

From `inference_open_h.py` `parse_arguments()` (lines 88-143):

| Flag | Type | Default | Required |
|------|------|---------|----------|
| `--experiment` | str | — | yes |
| `--ckpt_path` | str | — | yes |
| `--dataset_path` | str | — | yes |
| `--embodiment` | str | — | yes |
| `--data_split` | str | `test` | no |
| `--episode_ids` | str | — | yes |
| `--exclude_splits` | str (nargs+) | None | no |
| `--guidance` | float | `0` | no |
| `--seed` | int | `42` | no |
| `--save_root` | str | `results/open_h_eval` | no |
| `--save_fps` | int | `10` | no |
| `--save_comparison` | flag | off | no |
| `--max_chunks` | int | `20` | no |

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `step0_verify.sh` | Login node | Environment verification checklist |
| `import_sqsh.sh` | Login node | Convert `docker save` tar → enroot `.sqsh` |
| `fetch_weights.sh` | Login node | Download `model_ema_bf16.pt` from HuggingFace |
| `run_cosmos.sbatch` | Login node | Slurm sbatch using `enroot create` + `enroot start` |
| `smoke_test.sbatch` | Login node | Minimal test (1 episode, 1 chunk) |
| `scripts/run_inference.sh` | Container | Loops over embodiments, calls `inference_open_h.py` |
| `Dockerfile.corporate` | Laptop | Build wrapper injecting Netskope CA certs |
| `build_and_push.sh` | Laptop | Build Docker image (cross-compile linux/amd64) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `enroot import` fails on docker-archive | See fallback options in `import_sqsh.sh` output |
| `uv sync` TLS error during build | Use `Dockerfile.corporate` (Netskope CA injection) |
| `invalid peer certificate` on `docker push` | Netskope blocks ghcr.io; use `docker save` + tar transfer instead |
| `CUDA error: no kernel image` | Wrong GPU arch. Ensure H200 (sm_90). |
| `FileNotFoundError: stats_cosmos` | Missing normalization stats; see [Dataset Requirements](#dataset-requirements) |
| `CMR VERSIUS FILTER CACHE NOT FOUND` | Run `scripts/compute_cmr_filtered_episodes_cache.py` |
| Episode missing from split | Try `--data_split full` or check which episodes exist |
| Container doesn't see GPU | enroot NVIDIA hook should auto-expose; check `/etc/enroot/enroot.conf` |
| model_loader treats ckpt as DCP | Checkpoint doesn't end in `.pt`. Must use `model_ema_bf16.pt`. |
