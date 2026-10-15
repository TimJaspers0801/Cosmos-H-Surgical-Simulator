#!/usr/bin/env bash
# step0_verify.sh -- Run on the CLUSTER LOGIN NODE to verify the environment.
# Produces a pass/fail checklist.  Copy the output and send it back.
set -uo pipefail

echo "=========================================="
echo "STEP 0 — Cluster Environment Verification"
echo "=========================================="
echo ""

# --- 1. Docker image tar ---
echo "--- 1. Docker image tar ---"
TAR=$(ls -1 ~/docker-images/*.tar ~/docker-images/*.tar.gz 2>/dev/null | head -1)
if [ -n "$TAR" ]; then
    ls -lh "$TAR"
    echo "PASS: tar found at $TAR"
else
    echo "FAIL: no .tar or .tar.gz found in ~/docker-images/"
fi
echo ""

# --- 2. enroot version + import transports ---
echo "--- 2. enroot ---"
if command -v enroot &>/dev/null; then
    enroot version 2>&1
    echo ""
    echo "enroot import --help:"
    enroot import --help 2>&1 | head -30
    echo ""
    echo "enroot create --help:"
    enroot create --help 2>&1 | head -20
    echo ""
    echo "PASS: enroot available"
else
    echo "FAIL: enroot not found in PATH"
fi
echo ""

# --- 3. skopeo ---
echo "--- 3. skopeo ---"
if command -v skopeo &>/dev/null; then
    skopeo --version 2>&1
    echo "PASS: skopeo available"
else
    echo "INFO: skopeo not found (not required if enroot supports docker-archive)"
fi
echo ""

# --- 4. Disk space ---
echo "--- 4. Disk space ---"
echo "Home / docker-images filesystem:"
df -h ~/docker-images/ 2>/dev/null || df -h ~/
echo ""
echo "Candidate shared-storage filesystems (/scratch, /data, /shared, /mnt):"
df -h /scratch 2>/dev/null; df -h /data 2>/dev/null; df -h /shared 2>/dev/null
echo ""

# --- 5. NVIDIA / GPU check (login node may not have GPUs) ---
echo "--- 5. NVIDIA ---"
nvidia-smi 2>/dev/null | head -5 || echo "INFO: nvidia-smi not available on login node (expected)"
echo ""

# --- 6. Slurm + pyxis ---
echo "--- 6. Slurm ---"
sinfo --version 2>/dev/null || echo "FAIL: sinfo not found"
echo ""
echo "Available partitions:"
sinfo -o "%P %l %D %G" 2>/dev/null | head -10 || true
echo ""
echo "pyxis check (srun --container-image help):"
srun --help 2>&1 | grep -i "container" | head -5 || echo "INFO: no --container flags found (pyxis may not be loaded)"
echo ""

# --- 7. HuggingFace CLI ---
echo "--- 7. HuggingFace ---"
if command -v hf &>/dev/null; then
    echo "PASS: hf CLI found"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "PASS: huggingface_hub Python package available"
else
    echo "INFO: huggingface_hub not found (install with: pip install huggingface_hub)"
fi
echo ""

echo "=========================================="
echo "Done.  Copy everything above and send it back."
echo "=========================================="
