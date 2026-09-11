# LPDG Gateway Health Prediction System

A machine learning system that predicts which LoRaWAN gateways need maintenance visits each week, optimizing for cost reduction against a simple 3-sigma statistical baseline.

**Result**: Model catches **83% more** bad gateways than the baseline, saving **€3,000** per evaluation window.

---

## Quick Start

### 1. Set up data

Extract the challenge data into the `data/` directory:

```bash
unzip 03-challenge-data.zip -d data/
```

Your `data/` folder should contain:
```
data/
├── telemetry/          # Parquet files with gateway metrics
├── gateway_master.csv  # Gateway metadata (332 gateways)
├── field_visits.csv    # Historical visit outcomes
├── meter_read_success.csv
└── engineer_review_2026-02.xlsx  # Validation labels (60 Schlecht / 60 Normal)
```

### 2. Run the pipeline

```bash
# One command — trains model, generates predictions, validates format
docker compose run --rm gateway-health make all
```

Or without Docker:
```bash
pip install -r requirements.txt
make all
```

---

## Step by Step

```bash
# 1. Train model (~10 seconds)
make train

# 2. Generate predictions.csv (~5 seconds)
make predict

# 3. Validate format
make validate

# 4. Compare cost vs baseline
make evaluate

# 5. Run all tests
make test
```

---

## How to Know It's Working

| Check | Command | Expected |
|-------|---------|----------|
| Format validation | `make validate` | `predictions.csv: OK` |
| Cost comparison | `make evaluate` | Model cost < baseline cost |
| Model exists | `ls models/current/` | `model.lgb`, `metadata.json`, `features.json` |
| All tests pass | `make test` | 45+ tests pass |

---

## Architecture

```
Raw Data → ID Normalization → Feature Engineering (68 features)
                                      ↓
                              LightGBM Model (45 trees)
                                      ↓
                        Risk Ranking → Top 15 per week → predictions.csv
```

**Feature groups**:
- Connectivity: offline duration, disconnections, trends, 3-sigma anomaly hours
- Reboots: count, duration, power cycle ratio, importance
- System: CPU load, memory, uptime
- LoRa: packet counts, CRC error rate, TX success
- Signal: RSSI/RSCP/RSRQ bad ratios, network type
- Meter reads: read rate, trend, meters at risk, data staleness
- Gateway: age, firmware age, meters installed, site type
- Visit history: past visits, fault rate, days since last visit

---

## Model Versioning & Rollback

```bash
# List available versions
ls models/

# Rollback to a previous version
make rollback VERSION=v1.0.0

# Regenerate predictions with rolled-back model
make predict
```

---

## Retrain with New Data

```bash
# Point to new data directory
DATA_DIR=/path/to/new/data make train

# Generate new predictions
make predict
```

Training takes ~10 seconds. Safe to do during the live session.

---

## Data Drift Detection

```bash
make drift
```

Checks: schema changes, statistical distribution shifts (KS-test), gateway count anomalies. Returns severity: OK / WARNING / CRITICAL.

---

## Project Structure

```
├── DECISIONS.md           # 5 key decisions with alternatives
├── AI-USAGE.md            # AI tool usage disclosure
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # One-command startup
├── Makefile               # All operations as simple targets
├── config/default.yaml    # All hyperparameters and settings
├── src/
│   ├── data/
│   │   ├── loader.py      # Data loading + ID normalization
│   │   ├── features.py    # 68-feature engineering pipeline
│   │   ├── labels.py      # Label construction from field visits
│   │   └── drift.py       # Data drift detection
│   ├── model/
│   │   ├── train.py       # LightGBM training pipeline
│   │   ├── predict.py     # Inference + reason generation
│   │   ├── evaluate.py    # Cost comparison vs baseline
│   │   └── registry.py    # Model versioning + rollback
│   └── utils/
│       ├── config.py      # YAML + env var config loading
│       ├── gateway_ids.py # ID normalization (hex ↔ colon)
│       └── logging_setup.py
├── tests/                 # 45+ tests
├── scripts/               # Shell script wrappers
└── models/                # Versioned model storage
```

---

## Areas Covered

| Area | What We Did |
|------|-------------|
| **Machine Learning (E)** — PRIMARY | LightGBM model beating baseline on cost; feature importance analysis; gateway-level CV; temporal split validation |
| **Data Science (D)** | Cost analysis; definition of "needs a visit"; threshold sensitivity; write-up aimed at operations manager |
| **MLOps (F)** | Model versioning; reproducibility (deterministic training); drift detection; rollback (tested); retrain policy |
| **DevOps (C)** — Docker only | Containerized pipeline; one-command start; environment-based configuration |
