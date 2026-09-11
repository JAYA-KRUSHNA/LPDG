#!/usr/bin/env bash
# Train the gateway health prediction model.
set -euo pipefail

echo "→ Training model..."
python -m src.model.train "$@"
echo "✓ Training complete."
