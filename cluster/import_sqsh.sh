#!/usr/bin/env bash
# import_sqsh.sh -- Convert a `docker save` tar to an enroot .sqsh file.
#
# Runs on a cluster LOGIN node.  The cluster has NO Docker daemon and NO
# Apptainer, only enroot 3.5.0.
#
# Usage:
#   export TAR_PATH=~/docker-images/cosmos-h.tar    # or .tar.gz
#   export SQSH_DIR=/scratch/containers              # shared storage
#   ./cluster/import_sqsh.sh
#
# Strategy (tried in order):
#   1. enroot import docker-archive://<tar>   (enroot >= 3.4.1 with docker-archive support)
#   2. skopeo copy docker-archive:<tar> → OCI layout, then enroot import oci://<dir>
#   3. STOP with fallback instructions for the laptop
set -euo pipefail

TAR_PATH="${TAR_PATH:?Set TAR_PATH to the docker-save tar on the cluster (e.g. ~/docker-images/cosmos-h.tar)}"
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

mkdir -p "$SQSH_DIR"

# =========================================================================
# Route 1: enroot import docker-archive://
# =========================================================================
# enroot >= 3.4.1 can import directly from a docker-archive tar.
if enroot import --help 2>&1 | grep -qi "docker-archive"; then
    echo "--- Route 1: enroot import docker-archive:// ---"
    enroot import -o "$SQSH_FILE" "docker-archive://${TAR_PATH}"

    if [ -f "$SQSH_FILE" ]; then
        SIZE=$(du -sh "$SQSH_FILE" | cut -f1)
        echo ""
        echo "=== SUCCESS: $SQSH_FILE ($SIZE) ==="
        exit 0
    else
        echo "WARN: enroot import docker-archive did not produce $SQSH_FILE, trying next route."
    fi
fi

# =========================================================================
# Route 2: skopeo → OCI layout → enroot import
# =========================================================================
if command -v skopeo &>/dev/null; then
    echo "--- Route 2: skopeo copy → OCI layout → enroot import ---"
    OCI_DIR=$(mktemp -d "${SQSH_DIR}/cosmos-oci-XXXXXX")
    echo "Intermediate OCI dir: $OCI_DIR"

    skopeo copy "docker-archive:${TAR_PATH}" "oci:${OCI_DIR}:latest"

    if enroot import --help 2>&1 | grep -qi "oci"; then
        enroot import -o "$SQSH_FILE" "oci://${OCI_DIR}#latest"
    else
        # Some enroot builds don't support oci:// but can import from a
        # docker-daemon-like layout.  Try converting OCI → docker:
        echo "WARN: enroot doesn't support oci:// transport."
        echo "      Trying to convert OCI → docker-archive via skopeo..."
        DOCKER_TAR="${SQSH_DIR}/cosmos-docker-archive.tar"
        skopeo copy "oci:${OCI_DIR}:latest" "docker-archive:${DOCKER_TAR}"
        enroot import -o "$SQSH_FILE" "docker-archive://${DOCKER_TAR}"
        rm -f "$DOCKER_TAR"
    fi

    rm -rf "$OCI_DIR"

    if [ -f "$SQSH_FILE" ]; then
        SIZE=$(du -sh "$SQSH_FILE" | cut -f1)
        echo ""
        echo "=== SUCCESS: $SQSH_FILE ($SIZE) ==="
        exit 0
    fi
fi

# =========================================================================
# Route 3: STOP — need the laptop
# =========================================================================
echo ""
echo "============================================================"
echo "COULD NOT CONVERT TAR → SQSH ON THIS NODE"
echo "============================================================"
echo ""
echo "Neither 'enroot import docker-archive://' nor 'skopeo' worked."
echo "You have two fallback options (run these on your LAPTOP which has Docker):"
echo ""
echo "  OPTION A — import on the laptop, scp the .sqsh:"
echo "    docker load < ${TAR_PATH}            # reload the tar into Docker"
echo "    # Install enroot on macOS (brew) or use a Linux VM"
echo "    enroot import -o cosmos-h.sqsh 'dockerd://cosmos-h:latest'"
echo "    scp cosmos-h.sqsh <cluster>:${SQSH_DIR}/"
echo ""
echo "  OPTION B — push to a registry, then enroot import on the cluster:"
echo "    docker load < ${TAR_PATH}"
echo "    docker tag cosmos-h:latest ghcr.io/<user>/cosmos-h:latest"
echo "    docker push ghcr.io/<user>/cosmos-h:latest"
echo "    # On cluster login node:"
echo "    enroot import -o ${SQSH_DIR}/cosmos-h.sqsh 'docker://ghcr.io/<user>#cosmos-h:latest'"
echo ""
exit 1
