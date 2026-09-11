#!/usr/bin/env bash
# Validate predictions.csv format.
set -euo pipefail

PREDICTIONS="${1:-predictions.csv}"
echo "→ Validating ${PREDICTIONS}..."
python validate_submission.py "${PREDICTIONS}"
