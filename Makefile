.PHONY: all train predict validate evaluate test rollback clean help

# Default target: train, predict, and validate
all: train predict validate
	@echo "✓ Pipeline complete. predictions.csv is ready."

train:
	@echo "→ Training model..."
	python -m src.model.train
	@echo "✓ Training complete."

predict:
	@echo "→ Generating predictions..."
	python -m src.model.predict --out predictions.csv
	@echo "✓ predictions.csv generated."

validate:
	@echo "→ Validating predictions.csv..."
	python validate_submission.py predictions.csv
	@echo "✓ Validation passed."

evaluate:
	@echo "→ Evaluating model vs baseline..."
	python baseline_3sigma.py --data $${DATA_DIR:-./data} --out predictions_baseline.csv
	python -m src.model.evaluate --model predictions.csv --baseline predictions_baseline.csv
	@echo "✓ Evaluation complete."

test:
	@echo "→ Running tests..."
	python -m pytest tests/ -v --tb=short
	@echo "✓ All tests passed."

rollback:
	@echo "→ Rolling back to previous model version..."
	@if [ -z "$(VERSION)" ]; then echo "Usage: make rollback VERSION=v1.0.0"; exit 1; fi
	bash scripts/rollback.sh $(VERSION)
	@echo "✓ Rolled back to $(VERSION)."

drift:
	@echo "→ Running drift detection..."
	python -m src.data.drift
	@echo "✓ Drift report generated."

clean:
	rm -f predictions.csv predictions_baseline.csv predictions_model.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned."

help:
	@echo "LPDG Gateway Health Prediction System"
	@echo ""
	@echo "Usage:"
	@echo "  make all        Train, predict, and validate (default)"
	@echo "  make train      Train the model"
	@echo "  make predict    Generate predictions.csv"
	@echo "  make validate   Check predictions.csv format"
	@echo "  make evaluate   Compare model vs baseline on cost"
	@echo "  make test       Run all tests"
	@echo "  make rollback VERSION=v1.0.0  Rollback to a specific model version"
	@echo "  make drift      Run data drift detection"
	@echo "  make clean      Remove generated files"
