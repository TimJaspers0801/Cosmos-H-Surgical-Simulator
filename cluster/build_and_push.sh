#!/usr/bin/env bash
# build_and_push.sh -- Build Cosmos-H-Surgical-Simulator image and push to a registry.
#
# Runs on your LAPTOP (rootful Docker required).
# The cluster has NO Docker and NO Apptainer -- only enroot + pyxis -- so the
# image must reach the cluster via a registry that enroot can pull from.
#
# Usage:
#   ./cluster/build_and_push.sh          # build + push
#   PUSH=0 ./cluster/build_and_push.sh   # build only (dry-run)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =========================================================================
# Registry / tag — edit these to match your private registry.
# =========================================================================
# Default: GitHub Container Registry (ghcr.io).  You MUST be logged in:
#   echo $GHCR_PAT | docker login ghcr.io -u <github-user> --password-stdin
REG="${REG:-ghcr.io/<your-github-user>}"      # <-- EDIT THIS
TAG="${TAG:-latest}"
IMAGE="${REG}/cosmos-h:${TAG}"

# Alternatives (uncomment one and comment the ghcr.io default above):
# REG="docker.io/<your-dockerhub-user>"       # Docker Hub
# REG="nvcr.io/<your-ngc-org>"                # NGC private registry

PUSH="${PUSH:-1}"   # set PUSH=0 to skip the push step

echo "=== Cosmos-H-Surgical-Simulator — Build & Push ==="
echo "Repo root:  $REPO_ROOT"
echo "Image:      $IMAGE"
echo "Push:       $([ "$PUSH" = "1" ] && echo yes || echo no)"
echo ""

# =========================================================================
# 1. Pull Git LFS assets
# =========================================================================
# The Dockerfile's STANDALONE mode copies the whole tree into the image.
# LFS-tracked files (assets/*, *.mp4, *.png …) must be real files, not
# pointer stubs, or inference will fail at runtime.
cd "$REPO_ROOT"
if command -v git-lfs &>/dev/null || git lfs env &>/dev/null 2>&1; then
    echo "--- Pulling Git LFS objects ---"
    git lfs pull
else
    echo "WARNING: git-lfs not found. LFS pointers will be copied as-is."
    echo "         Install git-lfs and run 'git lfs pull' before building."
fi

# =========================================================================
# 2. Build the Docker image
# =========================================================================
# * Standard Dockerfile (Ampere–Hopper, base nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04)
# * STANDALONE=true  → code + deps baked into /workspace (no bind-mount needed)
# * CUDA_NAME=cu128  → PyTorch wheels for CUDA 12.8
#
# NOTE: The Dockerfile does NOT define APP_UID / APP_GID build args.
#       The container runs as root.  Under pyxis the job's UID is mapped in
#       automatically, so bind-mounted output dirs are writable by us.
echo ""
echo "--- Building Docker image (linux/amd64) ---"
# Force linux/amd64 even on Apple Silicon -- the cluster runs x86_64 H200 nodes.
# This uses QEMU emulation via Docker Desktop, so it's slower than a native build.
#
# If the corporate CA bundle exists (Netskope), use the wrapper Dockerfile that
# injects the CAs so uv/curl can verify TLS through the proxy.  Otherwise use
# the upstream Dockerfile directly.
CORPORATE_CA="$SCRIPT_DIR/corporate-ca-bundle.pem"
if [ -f "$CORPORATE_CA" ]; then
    echo "(Using cluster/Dockerfile.corporate — Netskope CA injection)"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.corporate"
else
    DOCKERFILE="$REPO_ROOT/Dockerfile"
fi

docker build \
    --platform linux/amd64 \
    --build-arg STANDALONE=true \
    --build-arg CUDA_NAME=cu128 \
    -f "$DOCKERFILE" \
    -t "$IMAGE" \
    "$REPO_ROOT"

echo ""
echo "Build complete: $IMAGE"

# =========================================================================
# 3. Push to registry
# =========================================================================
if [ "$PUSH" = "1" ]; then
    echo ""
    echo "--- Pushing to $REG ---"
    docker push "$IMAGE"
    echo ""
    echo "=== Push complete.  On the cluster run: ==="
    echo "  enroot import -o /path/to/cosmos-h.sqsh 'docker://${REG}#cosmos-h:${TAG}'"
else
    echo ""
    echo "PUSH=0 — skipping push.  To push later:"
    echo "  docker push $IMAGE"
fi

# =========================================================================
# Fallback: save-as-tar (last resort if registry is unreachable)
# =========================================================================
# enroot 3.5 can import from a local docker-daemon URI on the same host but
# the cluster has NO Docker daemon, so a docker:// pull is the only option.
# If you truly cannot push, save a tarball and transfer it:
#
#   docker save "$IMAGE" | gzip > cosmos-h.tar.gz
#   scp cosmos-h.tar.gz <cluster>:/scratch/images/
#
# Then on the cluster (enroot >= 3.4.1 supports docker-archive):
#   enroot import -o /path/to/cosmos-h.sqsh 'docker-archive:///scratch/images/cosmos-h.tar.gz'
#
# This is slower and uses 2× disk (tar + sqsh), so the registry path is
# strongly preferred.
