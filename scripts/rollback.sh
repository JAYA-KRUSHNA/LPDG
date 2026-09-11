#!/usr/bin/env bash
# Rollback to a previous model version.
# Usage: ./scripts/rollback.sh v1.0.0
set -euo pipefail

VERSION="${1:?Usage: rollback.sh VERSION (e.g., v1.0.0)}"
MODEL_DIR="${MODEL_DIR:-./models}"

echo "→ Rolling back to model version ${VERSION}..."

# Check version exists
if [ ! -d "${MODEL_DIR}/${VERSION}" ]; then
    echo "✗ Error: Version ${VERSION} not found in ${MODEL_DIR}"
    echo "  Available versions:"
    ls -d "${MODEL_DIR}"/v* 2>/dev/null || echo "  (none)"
    exit 1
fi

# Record current version
CURRENT=""
if [ -L "${MODEL_DIR}/current" ]; then
    CURRENT=$(readlink "${MODEL_DIR}/current")
    echo "  Current version: ${CURRENT}"
fi

# Set new version
rm -f "${MODEL_DIR}/current"
ln -s "${VERSION}" "${MODEL_DIR}/current"

echo "  New current version: ${VERSION}"

# Verify the rollback
python -c "
from src.model.registry import load_model
booster, meta, features = load_model('${MODEL_DIR}')
print(f'  Loaded model: {meta.get(\"version\", \"unknown\")} ({booster.num_trees()} trees)')
"

echo "✓ Rollback complete. Run 'make predict' to regenerate predictions."
