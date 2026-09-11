# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Install make and libgomp (LightGBM OpenMP dependency)
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy application code and tests
COPY src/ src/
COPY tests/ tests/
COPY config/ config/
COPY scripts/ scripts/
COPY baseline_3sigma.py validate_submission.py Makefile ./

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set default environment variables
ENV DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    RANDOM_SEED=42 \
    PYTHONUNBUFFERED=1

# Default command: run the full pipeline (train + predict + validate)
CMD ["make", "all"]
