# 🏗️ LPDG Gateway Health Prediction System

> **AI-powered predictive maintenance for LoRaWAN gateways** — predicting which of 320 gateways need engineer visits each week, optimizing field service costs.

[![CI — Build & Test](https://github.com/JAYA-KRUSHNA/LPDG/actions/workflows/ci.yml/badge.svg)](https://github.com/JAYA-KRUSHNA/LPDG/actions)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Tests](https://img.shields.io/badge/Tests-55%20Passing-brightgreen.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)

---

## 📊 Results at a Glance

| Metric | Baseline (3σ) | Our Model | Improvement |
|--------|:---:|:---:|:---:|
| **Bad gateways caught** | 6 | **10** | **+67%** |
| **Total cost** | €79,800 | **€77,400** | **-€2,400** |
| **AUC-ROC** | — | **0.888** | — |
| **Training time** | — | **10 seconds** | — |

> The model catches **67% more broken gateways** than the statistical baseline, translating to **€2,400 savings** per evaluation window — while respecting the 15 visits/week constraint.

---

## 🚀 Quick Start

### Prerequisites
- **Docker** (recommended) or **Python 3.12+**
- Challenge data (`03-challenge-data.zip`)

### 1. Set Up Data

```bash
# Extract the challenge data into the data/ directory
unzip 03-challenge-data.zip -d data/
```

Your `data/` folder should contain:
```
data/
├── telemetry/                      # Parquet files with hourly gateway metrics
├── gateway_master.csv              # Gateway metadata (332 gateways)
├── field_visits.csv                # Historical visit outcomes (642 visits)
├── meter_read_success.csv          # Weekly meter read success rates
└── engineer_review_2026-02.xlsx    # Ground truth labels (60 Schlecht / 60 Normal)
```

### 2. Run the Pipeline

**With Docker (recommended — one command):**
```bash
docker compose run --rm gateway-health make all
```

**Without Docker:**
```bash
pip install -r requirements.txt
make all
```

**Expected output:**
```
→ Training model...          ✓ (10 seconds, 34 trees, 68 features)
→ Generating predictions...  ✓ (5 seconds, 120 rows)
→ Validating predictions...  ✓ predictions.csv: OK
✓ Pipeline complete. predictions.csv is ready.
```

---

## 🎯 Problem Statement

A utility company operates **320 LoRaWAN gateways** that relay meter readings. When a gateway fails, meter data stops flowing — costing **€600/week** in penalties per undetected failure.

**The constraint:** Only **15 engineer visits** can be scheduled per week.

**The challenge:** Pick the 15 gateways most likely to need attention each week. Every correct pick saves €600; every wrong pick wastes €380 in visit costs.

**Our solution:** A LightGBM classifier trained on 68 engineered features that ranks all gateways by failure risk, selecting the top 15 for weekly visits.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   5 DATA SOURCES                        │
│  📡 Telemetry (1.4M rows)    🏭 Gateway Master (332)   │
│  🔧 Field Visits (642)       📊 Meter Reads (7,226)    │
│  📋 Engineer Review (120)                               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│              DATA LOADING & NORMALIZATION                │
│  • Gateway ID normalization (hex ↔ colon format)        │
│  • Latin-1 encoding handling (German text)              │
│  • Schema validation & decommissioned gateway removal   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│              68-FEATURE ENGINEERING PIPELINE             │
│                                                          │
│  Connectivity (15)  │  Reboots (8)     │  System (6)    │
│  LoRa/Radio (8)     │  Meter Reads (8) │  Gateway (8)   │
│  Visit History (5)  │  Trends (10)     │                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  LightGBM CLASSIFIER                     │
│  • 34 trees, 3-fold gateway-level CV                    │
│  • scale_pos_weight = 1.58 (cost-driven)                │
│  • Early stopping to prevent overfitting                │
│  • AUC-ROC: 0.888 | AUC-PR: 0.339                      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│              RISK RANKING → predictions.csv              │
│  • Score all 320 gateways per week                      │
│  • Select top 15 by risk score                          │
│  • Generate human-readable reasons                      │
│  • 8 weeks × 15 gateways = 120 predictions              │
└──────────────────────────────────────────────────────────┘
```

---

## 📋 Available Commands

| Command | Description | Time |
|---------|-------------|------|
| `make all` | Train → Predict → Validate (default) | ~15s |
| `make train` | Train the LightGBM model | ~10s |
| `make predict` | Generate `predictions.csv` | ~5s |
| `make validate` | Check predictions format | <1s |
| `make evaluate` | Cost comparison vs 3σ baseline | ~10s |
| `make test` | Run all 55 tests | ~25s |
| `make drift` | Data drift detection report | ~5s |
| `make dashboard` | Interactive HTML dashboard with charts | ~2s |
| `make rollback VERSION=v1.0.0` | Rollback to a previous model | <1s |
| `make clean` | Remove generated files | <1s |

---

## 🔬 Feature Engineering Details

### 68 features across 8 groups:

| Group | Count | Key Features |
|-------|:---:|---|
| **Connectivity** | 15 | Offline duration (mean/max/sum/trend), disconnection count, 3σ anomaly hours |
| **Reboots** | 8 | Reboot count, duration, power cycle ratio, reboot importance |
| **System Health** | 6 | CPU load, free memory, uptime statistics |
| **LoRa/Radio** | 8 | RX/TX packets, CRC error rate, TX success rate |
| **Signal Quality** | 5 | RSSI/RSCP/RSRQ bad ratios, network type |
| **Meter Reads** | 8 | Read rate, 4-week trend, meters at risk, data staleness |
| **Gateway Info** | 8 | Age, firmware age, site type, meters installed |
| **Visit History** | 5 | Past visits, fault rate, days since last visit |
| **Trends** | 5 | Week-over-4-week ratio for key metrics (worsening = higher risk) |

### Top 5 Most Important Features (by gain):

| Rank | Feature | Gain | Interpretation |
|:---:|---------|:---:|---|
| 🥇 | `meters_at_risk` | 3,429 | How many meters depend on this gateway |
| 🥈 | `read_rate_trend_4w` | 1,934 | Is meter read success declining? |
| 🥉 | `days_since_last_visit` | 1,629 | Time since last engineer check |
| 4 | `n_past_visits` | 1,356 | How many times visited before |
| 5 | `last_read_rate` | 869 | Current meter read success rate |

---

## 🛡️ Data Integrity & Leakage Prevention

| Safeguard | Implementation |
|-----------|----------------|
| **Temporal gap** | 21-day gap between training end and scored window start |
| **No scored-window data in training** | Training: Sep 2025 – Jan 2026 / Scoring: Feb – Mar 2026 |
| **Gateway-level CV** | GroupKFold ensures no gateway appears in both train & validation |
| **Engineer review excluded** | Feb 2026 engineer review used only for validation, never for training |
| **Decommissioned filtering** | Per-week filtering — a gateway decommissioned in Dec won't appear in Jan predictions |

---

## 📦 Model Versioning & Rollback

```bash
# Models are saved with full metadata
models/
├── v1.0.0/
│   ├── model.lgb           # Trained LightGBM model
│   ├── metadata.json       # Training config, metrics, data hash, timestamp
│   └── features.json       # Feature names, importance scores
├── v1.1.0/
│   └── ...
└── current -> v1.1.0       # Symlink to active version

# Rollback to any version
make rollback VERSION=v1.0.0

# Verify rollback
make predict    # Uses the rolled-back model
make validate   # Confirms output is valid
```

Each model stores:
- **Training metadata**: timestamp, random seed, data hash, sample count
- **CV metrics**: AUC-ROC, AUC-PR, cost metrics per fold
- **Feature importance**: full ranking for explainability

---

## 🔍 Data Drift Detection

```bash
make drift
```

Detects three types of drift:

| Check | Method | Severity |
|-------|--------|----------|
| **Schema drift** | Column presence/absence | 🔴 CRITICAL |
| **Statistical drift** | KS-test per feature (p < 0.01) | 🟡 WARNING |
| **Volume drift** | Gateway count vs expected range | 🟡 WARNING |

---

## 🐳 Docker

```bash
# Build the image
docker compose build

# Run full pipeline
docker compose run --rm gateway-health make all

# Run tests inside container
docker compose run --rm gateway-health make test

# Run with custom parameters
COST_FP=400 COST_FN=700 docker compose run --rm gateway-health make all
```

**Docker architecture:**
- Multi-stage build (builder + runtime) for minimal image size
- Data mounted read-only (`./data:/app/data:ro`)
- Models persisted via volume mount (`./models:/app/models`)
- Environment variables override all config values

---

## ✅ Testing

```bash
make test    # 55 tests, ~25 seconds
```

| Test Suite | Tests | What It Covers |
|-----------|:---:|---|
| `test_loader.py` | 13 | ID normalization, data loading, schema validation |
| `test_features.py` | 10 | Feature engineering, trends, NaN handling |
| `test_drift.py` | 9 | Schema, statistical, and volume drift detection |
| `test_e2e.py` | 16 | End-to-end pipeline, format validation, cost comparison |
| `test_rollback.py` | 7 | Model versioning, rollback, reproducibility |

---

## 🗂️ Project Structure

```
LPDG/
├── README.md                  # This file
├── DECISIONS.md               # 5 key decisions with alternatives considered
├── AI-USAGE.md                # AI tool usage disclosure
│
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # One-command startup
├── Makefile                   # All operations as simple targets
├── requirements.txt           # Pinned Python dependencies
├── config/default.yaml        # All hyperparameters and settings
├── .env.example               # Environment variable template
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading + ID normalization
│   │   ├── features.py        # 68-feature engineering pipeline
│   │   ├── labels.py          # Label construction from field visits
│   │   └── drift.py           # Data drift detection (KS-test)
│   ├── model/
│   │   ├── train.py           # LightGBM training with gateway-level CV
│   │   ├── predict.py         # Inference + human-readable reason generation
│   │   ├── evaluate.py        # Cost-based comparison vs baseline
│   │   └── registry.py        # Model versioning, save/load, rollback
│   └── utils/
│       ├── config.py          # YAML + env var config loading
│       ├── gateway_ids.py     # ID normalization (hex ↔ colon)
│       └── logging_setup.py   # Structured logging
│
├── tests/                     # 55 tests across 5 suites
├── scripts/                   # Shell script wrappers
├── models/                    # Versioned model storage (gitignored)
│
├── predictions.csv            # Final submission (120 rows)
├── baseline_3sigma.py         # Provided 3-sigma baseline
└── validate_submission.py     # Provided format validator
```

---

## 🔧 Configuration

All settings in `config/default.yaml`, overridable via environment variables:

| Setting | YAML Key | Env Variable | Default |
|---------|----------|:---:|:---:|
| Data directory | `paths.data_dir` | `DATA_DIR` | `./data` |
| Model directory | `paths.model_dir` | `MODEL_DIR` | `./models` |
| Visit cost (€) | `cost.visit_cost` | `COST_FP` | 380 |
| Miss cost (€) | `cost.miss_cost` | `COST_FN` | 600 |
| Visits per week | `cost.visits_per_week` | — | 15 |
| Random seed | `model.random_seed` | `RANDOM_SEED` | 42 |
| Log level | — | `LOG_LEVEL` | INFO |

---

## 🎓 Areas Covered

| Area | What We Did |
|------|-------------|
| **Machine Learning (E)** — PRIMARY | LightGBM classifier beating baseline on cost; 68 engineered features; gateway-level CV; feature importance analysis |
| **Data Science (D)** | Cost-based evaluation framework; label definition from German field visit outcomes; threshold sensitivity analysis |
| **MLOps (F)** | Model versioning with metadata; reproducibility (deterministic training, data hashing); drift detection; tested rollback; retrain pipeline |
| **DevOps (C)** — Docker only | Containerized pipeline; one-command execution; environment-based config; GitHub Actions CI |

---

## 📝 Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for detailed rationale on:

1. **LightGBM over XGBoost/Random Forest** — faster training, native categorical support, better with small data
2. **Gateway-level CV over temporal split** — prevents data leakage from same gateway in train+val
3. **Cost-weighted loss over standard binary crossentropy** — missing a bad gateway (€600) costs more than a wasted visit (€380)
4. **68 hand-crafted features over raw data** — domain knowledge outperforms auto-feature extraction on tabular data
5. **Symlink-based model registry over MLflow** — lightweight, zero dependencies, fits the project scope

---

## 📄 License

This project was created for the LPDG Campus Drive Challenge. All challenge data remains property of the challenge organizers.
