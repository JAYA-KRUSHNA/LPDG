#!/usr/bin/env bash
# Evaluate model predictions vs baseline.
set -euo pipefail

echo "→ Generating baseline predictions..."
python baseline_3sigma.py --data "${DATA_DIR:-./data}" --out predictions_baseline.csv

echo "→ Evaluating model vs baseline..."
python -m src.model.evaluate --model predictions.csv --baseline predictions_baseline.csv
