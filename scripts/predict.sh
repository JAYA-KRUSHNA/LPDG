#!/usr/bin/env bash
# Generate predictions.csv from the trained model.
set -euo pipefail

echo "→ Generating predictions..."
python -m src.model.predict --out predictions.csv "$@"
echo "✓ predictions.csv generated."
