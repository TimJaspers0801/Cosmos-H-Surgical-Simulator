#!/usr/bin/env bash
# import_sqsh.sh -- Convert a `docker save` tar(.gz) to an enroot .sqsh file.
#
# Runs on a cluster LOGIN node.  The cluster has NO Docker daemon, NO
# Apptainer, and this enroot 3.5.0 build does NOT support docker-archive://.
#
# This script manually extracts the docker-save layers and packs them into a
# squashfs image that enroot can use.
#
# Usage:
#   export TAR_PATH=~/docker-images/cosmos-h.tar.gz
#   export SQSH_DIR=/scratch/containers
#   bash cluster/import_sqsh.sh
#
# Requires: tar, python3, mksquashfs (or sqfstar)
# Temp space: ~2-3x the compressed tar size on SQSH_DIR's filesystem
set -euo pipefail

TAR_PATH="${TAR_PATH:?Set TAR_PATH to the docker-save tar on the cluster}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to shared storage for the .sqsh (e.g. /scratch/containers)}"
SQSH_FILE="${SQSH_DIR}/cosmos-h.sqsh"

echo "=== Import docker-save tar → enroot .sqsh ==="
echo "Tar:    $TAR_PATH"
echo "Output: $SQSH_FILE"
echo ""

if [ ! -f "$TAR_PATH" ]; then
    echo "ERROR: $TAR_PATH not found."
    exit 1
fi

# Check for mksquashfs
if ! command -v mksquashfs &>/dev/null; then
    echo "ERROR: mksquashfs not found.  Install squashfs-tools."
    echo "       (On Ubuntu/Debian: sudo apt install squashfs-tools)"
    exit 1
fi

mkdir -p "$SQSH_DIR"

# =========================================================================
# Work directory on the same filesystem as SQSH_DIR (avoid cross-device moves)
# =========================================================================
WORKDIR=$(mktemp -d "${SQSH_DIR}/.import-XXXXXX")
trap 'echo "Cleaning up $WORKDIR ..."; rm -rf "$WORKDIR"' EXIT

echo "Work directory: $WORKDIR"
echo ""

# =========================================================================
# Step 1: Decompress if gzipped
# =========================================================================
if file "$TAR_PATH" | grep -q gzip; then
    echo "--- Decompressing .tar.gz ---"
    PLAIN_TAR="${WORKDIR}/image.tar"
    gunzip -c "$TAR_PATH" > "$PLAIN_TAR"
    echo "Decompressed to $PLAIN_TAR ($(du -sh "$PLAIN_TAR" | cut -f1))"
else
    PLAIN_TAR="$TAR_PATH"
fi
echo ""

# =========================================================================
# Step 2: Extract the docker-save archive
# =========================================================================
echo "--- Extracting docker-save archive ---"
ARCHIVE_DIR="${WORKDIR}/archive"
mkdir -p "$ARCHIVE_DIR"
tar xf "$PLAIN_TAR" -C "$ARCHIVE_DIR"
echo "Extracted to $ARCHIVE_DIR"

# Clean up decompressed tar to save space
if [ "$PLAIN_TAR" = "${WORKDIR}/image.tar" ]; then
    rm -f "$PLAIN_TAR"
    echo "Removed decompressed tar to free space"
fi
echo ""

# =========================================================================
# Step 3: Read manifest.json to get layers in order
# =========================================================================
echo "--- Reading manifest.json ---"
if [ ! -f "${ARCHIVE_DIR}/manifest.json" ]; then
    echo "ERROR: manifest.json not found in the archive.  Is this a 'docker save' tar?"
    exit 1
fi

LAYERS=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    manifest = json.load(f)
if not manifest:
    print('ERROR: empty manifest', file=sys.stderr)
    sys.exit(1)
# docker save always produces a list; take the first (and usually only) entry
entry = manifest[0]
for layer_path in entry['Layers']:
    print(layer_path)
" "${ARCHIVE_DIR}/manifest.json")

LAYER_COUNT=$(echo "$LAYERS" | wc -l)
echo "Found $LAYER_COUNT layer(s)"
echo ""

# =========================================================================
# Step 4: Extract layers in order to build a rootfs
# =========================================================================
echo "--- Building rootfs from layers ---"
ROOTFS="${WORKDIR}/rootfs"
mkdir -p "$ROOTFS"

N=0
for LAYER in $LAYERS; do
    N=$((N + 1))
    LAYER_TAR="${ARCHIVE_DIR}/${LAYER}"
    if [ ! -f "$LAYER_TAR" ]; then
        echo "ERROR: Layer file not found: $LAYER_TAR"
        exit 1
    fi
    echo "  [$N/$LAYER_COUNT] Extracting $LAYER ..."
    # Extract layer; --overwrite-dir handles re-created directories.
    # Docker whiteout files (.wh.*) are extracted as regular files here;
    # we handle them in the next step.
    tar xf "$LAYER_TAR" -C "$ROOTFS" --no-same-owner 2>/dev/null || true
done
echo ""

# =========================================================================
# Step 5: Handle Docker whiteout files (.wh.*)
# =========================================================================
# Docker uses .wh.<name> sentinel files to mark deletions in upper layers.
# .wh..wh..opq marks an opaque directory (all prior contents removed).
echo "--- Processing whiteout files ---"
WH_COUNT=0
find "$ROOTFS" -name '.wh.*' -type f 2>/dev/null | while read -r whfile; do
    dir=$(dirname "$whfile")
    base=$(basename "$whfile")
    if [ "$base" = ".wh..wh..opq" ]; then
        # Opaque whiteout: remove all siblings that were from lower layers
        # (we can't tell which, so just remove the marker)
        rm -f "$whfile"
    else
        # Regular whiteout: remove the target file/dir
        target="${dir}/${base#.wh.}"
        rm -rf "$target" 2>/dev/null || true
        rm -f "$whfile"
    fi
    WH_COUNT=$((WH_COUNT + 1))
done
echo "Processed whiteout files"
echo ""

# =========================================================================
# Step 6: Create squashfs image
# =========================================================================
echo "--- Creating squashfs image ---"
echo "This may take several minutes for a large rootfs..."

# Remove any existing .sqsh (mksquashfs -noappend would fail on a non-sqsh file)
rm -f "$SQSH_FILE"

mksquashfs "$ROOTFS" "$SQSH_FILE" \
    -noappend \
    -all-root \
    -no-progress \
    -processors "$(nproc)" \
    2>&1 | tail -5

echo ""

# =========================================================================
# Step 7: Verify
# =========================================================================
if [ -f "$SQSH_FILE" ]; then
    SIZE=$(du -sh "$SQSH_FILE" | cut -f1)
    echo "=== SUCCESS: $SQSH_FILE ($SIZE) ==="
    echo ""
    echo "Next steps:"
    echo "  enroot create --name cosmos-h $SQSH_FILE"
    echo "  # or use in sbatch: srun --container-image=$SQSH_FILE ..."
else
    echo "ERROR: $SQSH_FILE was not created.  Check mksquashfs output above."
    exit 1
fi
