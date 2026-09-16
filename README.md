<p align="center">
  <h1 align="center">🏗️ LPDG — Gateway Health Prediction</h1>
  <p align="center">
    <em>AI-powered predictive maintenance for LoRaWAN gateways</em><br>
    <em>Predicting which of 320 gateways need engineer visits each week</em>
  </p>
</p>

<p align="center">
  <a href="https://github.com/JAYA-KRUSHNA/LPDG/actions/workflows/ci.yml">
    <img src="https://github.com/JAYA-KRUSHNA/LPDG/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Tests-55%20Passing-22c55e?logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/Model-LightGBM-9ACD32?logo=lightgbm" alt="LightGBM">
</p>

<p align="center">
  🎬 <strong><a href="https://drive.google.com/file/d/1t16ZdhUe7f13UEZwGnr8QrenM8gmgXX2/view?usp=sharing">▶️ Watch the Video Explanation (6 min)</a></strong><br>
  <em>Full project walkthrough — problem statement, feature engineering, model design, results, and live demo</em>
</p>

---

## 📊 Results at a Glance

| Metric | 3σ Baseline | Our Model | Delta |
|--------|:-----------:|:---------:|:-----:|
| **Bad gateways caught** | 6 | **10** | **+67%** ✅ |
| **Bad gateways missed** | 54 | **50** | −4 |
| **Total cost (2 weeks)** | €79,800 | **€77,400** | **−€2,400** |
| **AUC-ROC** | — | **0.888** | — |
| **Training time** | — | **~10 seconds** | — |

> **Bottom line:** Our model catches **67% more broken gateways** than the statistical baseline, saving **€2,400** per evaluation window — while staying within the 15 visits/week constraint.

---

## 🚀 Quick Start (2 minutes)

### Prerequisites
- **Docker** (recommended) **OR** Python 3.12+
- Challenge data file (`03-challenge-data.zip`)

### Step 1 — Set Up Data

```bash
git clone https://github.com/JAYA-KRUSHNA/LPDG.git
cd LPDG
# Extract challenge data into data/ folder
unzip 03-challenge-data.zip -d data/
```

Your `data/` folder should look like this:
```
data/
├── telemetry/                      # Parquet files — hourly gateway metrics
├── gateway_master.csv              # 332 gateways (metadata)
├── field_visits.csv                # 642 historical visit outcomes
├── meter_read_success.csv          # Weekly meter read success rates
└── engineer_review_2026-02.xlsx    # Ground truth (60 Schlecht / 60 Normal)
```

### Step 2 — Run the Pipeline

**Option A — Docker (one command, zero setup):**
```bash
docker compose run --rm gateway-health
```

**Option B — Local Python:**
```bash
pip install -r requirements.txt
make all
```

**What happens:**
```
→ Training model...          ✓  10s — 34 trees, 68 features
→ Generating predictions...  ✓   5s — 120 rows (8 weeks × 15 gateways)
→ Validating predictions...  ✓  predictions.csv: OK
✓ Pipeline complete. predictions.csv is ready.
```

### Step 3 — Explore Results

```bash
make dashboard    # Opens interactive HTML dashboard in browser
make evaluate     # Prints cost comparison: Model vs Baseline
```

---

## 🎯 Problem Statement

A utility company operates **320 LoRaWAN gateways** that relay smart meter readings. When a gateway fails silently, meter data stops flowing — costing **€600/week** in penalties per undetected failure.

**The constraint:** Only **15 engineer visits** can be scheduled per week.

**The challenge:** Each week, pick the 15 gateways most likely to need attention.

| Outcome | Cost |
|---------|------|
| ✅ Visit a truly broken gateway | €380 (visit) — but **saves €600/week** in penalties |
| ❌ Visit a healthy gateway | €380 wasted |
| ❌ Miss a broken gateway | **€600/week** penalty continues |

**Our approach:** Train a LightGBM classifier on 68 engineered features to rank all gateways by failure risk, then select the top 15 each week.

---

## 🏗️ How It Works — Data Flow

```
                         RAW DATA
    ┌────────────────────────────────────────────────────┐
    │  📡 Telemetry       1.4M rows (hourly metrics)     │
    │  🏭 Gateway Master  332 gateways (metadata)        │
    │  🔧 Field Visits    642 visits (outcomes)           │
    │  📊 Meter Reads     7,226 rows (weekly success %)   │
    │  📋 Engineer Review  120 labels (ground truth)      │
    └──────────────────────┬─────────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │    DATA LOADING         │
              │  • ID normalization     │
              │  • Latin-1 decoding     │
              │  • Schema validation    │
              │  • Decommissioned       │
              │    gateway removal      │
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
              │  68 ENGINEERED FEATURES │
              │                         │
              │  Connectivity    (15)   │
              │  Reboots          (8)   │
              │  System Health    (6)   │
              │  LoRa/Radio       (8)   │
              │  Meter Reads      (8)   │
              │  Gateway Info     (8)   │
              │  Visit History    (5)   │
              │  Trends          (10)   │
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
              │   LightGBM CLASSIFIER   │
              │  • 34 trees             │
              │  • 3-fold gateway CV    │
              │  • Cost-weighted loss   │
              │  • AUC-ROC: 0.888       │
              └──────────┬──────────────┘
                         ▼
              ┌─────────────────────────┐
              │   predictions.csv       │
              │  • 8 weeks × 15/week    │
              │  • Ranked by risk       │
              │  • Human-readable       │
              │    reasons included     │
              └─────────────────────────┘
```

---

## 📋 All Available Commands

| Command | What It Does | Time |
|---------|-------------|:----:|
| `make all` | Train → Predict → Validate **(default)** | ~15s |
| `make train` | Train the LightGBM model | ~10s |
| `make predict` | Generate `predictions.csv` | ~5s |
| `make validate` | Check `predictions.csv` format | <1s |
| `make evaluate` | Cost comparison vs 3σ baseline | ~10s |
| `make dashboard` | **Interactive HTML dashboard** with charts | ~2s |
| `make test` | Run all 55 tests | ~25s |
| `make drift` | Data drift detection report | ~5s |
| `make rollback VERSION=v1.0.0` | Rollback to a previous model version | <1s |
| `make clean` | Remove generated files | <1s |

**All commands work both locally and inside Docker:**
```bash
# Local
make dashboard

# Docker
docker compose run --rm gateway-health make dashboard --no-open
```

---

## 📊 Interactive Dashboard

Run `make dashboard` to generate a **self-contained HTML dashboard** that opens in your browser:

| Section | What It Shows |
|---------|--------------|
| **KPI Cards** | AUC-ROC, cost savings, catches, predictions count — animated counters |
| **Cost Waterfall** | How our model saves €2,400 vs the 3σ baseline |
| **Feature Importance** | Top 15 features ranked by information gain |
| **Cross-Validation** | AUC-ROC, AUC-PR, Recall@15 with error bars |
| **Risk Score Curves** | Per-week risk profiles showing score decay by rank |
| **Score Distribution** | Violin plots per week showing outliers |
| **Gateway Heatmap** | Which gateways are selected each week (color = risk) |
| **Predictions Table** | All 120 predictions — filterable by week, color-coded scores |

> All charts are **fully interactive**: zoom, pan, hover for details, download as PNG.

---

## 🔬 Feature Engineering (68 Features)

| Group | Count | Key Features | Why It Matters |
|-------|:-----:|---|---|
| **Connectivity** | 15 | Offline duration, disconnection count, 3σ anomaly hours | Direct signal of gateway availability |
| **Reboots** | 8 | Reboot count, duration, power cycle ratio | Hardware instability indicator |
| **System Health** | 6 | CPU load, free memory, uptime | Resource exhaustion signals |
| **LoRa/Radio** | 8 | RX/TX packets, CRC error rate, TX success rate | Communication quality |
| **Meter Reads** | 8 | Read rate, 4-week trend, meters at risk, staleness | **Core business metric** |
| **Gateway Info** | 8 | Age, firmware age, site type, meters installed | Static risk factors |
| **Visit History** | 5 | Past visits, fault rate, days since last visit | Repeat offender detection |
| **Trends** | 10 | Week-over-4-week ratio for all major metrics | Detects deterioration |

### Top 5 Most Predictive Features

| Rank | Feature | Why |
|:----:|---------|-----|
| 🥇 | `meters_at_risk` | Gateways serving more meters have higher impact when failing |
| 🥈 | `read_rate_trend_4w` | A declining meter read rate is the strongest early warning |
| 🥉 | `days_since_last_visit` | Longer time since last check = higher accumulated risk |
| 4 | `n_past_visits` | Repeat visitors have chronic underlying issues |
| 5 | `last_read_rate` | Current data quality directly measures gateway health |

---

## 🛡️ Data Integrity & Leakage Prevention

| Safeguard | How We Implement It |
|-----------|-------------------|
| **Temporal gap** | 21-day gap between last training week and first scored week |
| **No future data** | Training uses Sep 2025–Jan 2026; scoring uses Feb–Mar 2026 |
| **Gateway-level CV** | `GroupKFold` — same gateway never appears in both train & validation |
| **Labels excluded** | Engineer review (Feb 2026) used only for evaluation, never training |
| **Per-week filtering** | Decommissioned gateways excluded based on each week's active list |
| **Deterministic** | Fixed seed (42), data hash tracked — same input = same output |

---

## 📦 Model Versioning & Rollback

Every trained model is saved with full metadata for audit and reproducibility:

```
models/
├── v1.0.0/
│   ├── model.lgb           # Trained LightGBM booster
│   ├── metadata.json       # Config, metrics, data hash, timestamp
│   └── features.json       # Feature names + importance scores
├── v1.1.0/
│   └── ...
└── current -> v1.1.0       # Symlink to active version
```

```bash
# Rollback to any previous version
make rollback VERSION=v1.0.0

# Verify the rollback
make predict      # Uses the rolled-back model
make validate     # Confirms output format
```

Each `metadata.json` includes:
- Training timestamp & random seed
- SHA256 hash of training data
- CV metrics (AUC-ROC, AUC-PR, cost per fold)
- Feature importance ranking

---

## 🔍 Data Drift Detection

```bash
make drift
```

Detects three types of drift to alert when the model may need retraining:

| Check | Method | What It Catches |
|-------|--------|----------------|
| **Schema drift** | Column presence/absence | Missing or renamed telemetry fields |
| **Statistical drift** | KS-test per feature (p < 0.01) | Distribution shifts in key metrics |
| **Volume drift** | Gateway count vs expected range | Mass decommissioning or data loss |

---

## 🐳 Docker

```bash
# Build the image
docker compose build

# Run full pipeline (default)
docker compose run --rm gateway-health

# Run specific commands
docker compose run --rm gateway-health make test
docker compose run --rm gateway-health make evaluate
docker compose run --rm gateway-health make dashboard --no-open

# Override cost parameters
COST_FP=400 COST_FN=700 docker compose run --rm gateway-health
```

**How it works:**
- Multi-stage build (builder + runtime) for minimal image size
- `data/` mounted read-only — never copied into the image
- `models/` mounted read-write — persisted between runs
- All environment variables are configurable

---

## ✅ Testing (55 Tests)

```bash
make test
```

| Test Suite | Tests | What It Validates |
|-----------|:-----:|-------------------|
| `test_loader.py` | 13 | Data loading, ID normalization, schema validation |
| `test_features.py` | 10 | Feature engineering, trends, NaN handling |
| `test_drift.py` | 9 | Schema, statistical, and volume drift detection |
| `test_e2e.py` | 16 | Full pipeline, format validation, cost comparison vs baseline |
| `test_rollback.py` | 7 | Model versioning, rollback, reproducibility |

---

## 🗂️ Project Structure

```
LPDG/
│
├── README.md                      ← You are here
├── DECISIONS.md                   ← 5 key design decisions with rationale
├── AI-USAGE.md                    ← AI tool usage disclosure
│
├── Dockerfile                     ← Multi-stage Docker build
├── docker-compose.yml             ← One-command startup
├── Makefile                       ← All operations as simple targets
├── requirements.txt               ← Pinned Python dependencies
├── config/default.yaml            ← All hyperparameters & settings
├── .env.example                   ← Environment variable template
├── .github/workflows/ci.yml      ← GitHub Actions CI pipeline
│
├── src/
│   ├── data/
│   │   ├── loader.py              ← Data loading + ID normalization
│   │   ├── features.py            ← 68-feature engineering pipeline
│   │   ├── labels.py              ← Label construction from field visits
│   │   └── drift.py               ← Data drift detection (KS-test)
│   ├── model/
│   │   ├── train.py               ← LightGBM training with GroupKFold
│   │   ├── predict.py             ← Inference + reason generation
│   │   ├── evaluate.py            ← Cost-based model evaluation
│   │   └── registry.py            ← Model versioning, save/load, rollback
│   ├── dashboard/
│   │   └── report.py              ← Interactive HTML dashboard generator
│   └── utils/
│       ├── config.py              ← YAML + env var config loading
│       ├── gateway_ids.py         ← ID normalization (hex ↔ colon)
│       └── logging_setup.py       ← Structured logging
│
├── tests/                         ← 55 tests across 5 suites
├── scripts/                       ← Shell script wrappers
├── models/                        ← Versioned model storage (gitignored)
│
├── predictions.csv                ← Final submission (120 rows)
├── baseline_3sigma.py             ← Provided 3-sigma baseline
└── validate_submission.py         ← Provided format validator
```

---

## ⚙️ Configuration

All settings live in `config/default.yaml` and can be overridden via environment variables:

| Setting | Config Key | Env Variable | Default |
|---------|-----------|:------------:|:-------:|
| Data directory | `paths.data_dir` | `DATA_DIR` | `./data` |
| Model directory | `paths.model_dir` | `MODEL_DIR` | `./models` |
| Visit cost (€) | `cost.visit_cost` | `COST_FP` | 380 |
| Miss cost (€) | `cost.miss_cost` | `COST_FN` | 600 |
| Visits per week | `cost.visits_per_week` | — | 15 |
| Random seed | `model.random_seed` | `RANDOM_SEED` | 42 |
| Log level | — | `LOG_LEVEL` | INFO |

---

## 🎓 Competency Areas Covered

| Area | What We Demonstrate |
|------|-------------------|
| **E — Machine Learning** (Primary) | LightGBM classifier, 68 engineered features, cost-weighted loss, gateway-level CV, feature importance analysis, beats baseline by 67% |
| **D — Data Science** | Cost-based evaluation framework, label construction from German field visit data, threshold analysis, interactive dashboard with 7 chart types |
| **F — MLOps** | Model versioning with metadata, deterministic training (data hashing), drift detection, tested rollback, automated retrain pipeline |
| **C — DevOps** (Docker) | Containerized pipeline, one-command execution, environment-based config, multi-stage build, GitHub Actions CI |

---

## 📝 Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for detailed rationale. Summary:

| Decision | Why |
|----------|-----|
| **LightGBM** over XGBoost | Faster training, native categorical support, built-in early stopping |
| **Gateway-level CV** over temporal split | Prevents data leakage from same gateway in train + validation |
| **Cost-weighted loss** (scale_pos_weight=1.58) | Missing a bad gateway (€600) costs more than a wasted visit (€380) |
| **68 hand-crafted features** over auto-extraction | Domain knowledge outperforms automated methods on tabular data |
| **Symlink model registry** over MLflow | Lightweight, zero dependencies, fits the project scope |

---

## 📄 License

This project was created for the LPDG Campus Drive Challenge. All challenge data remains property of the challenge organizers.
