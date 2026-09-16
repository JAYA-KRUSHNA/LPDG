"""Prediction pipeline — generates predictions.csv from a saved model.

Completely separate from training (MLOps requirement F1).
Loads the saved model, computes features for scored weeks, ranks gateways,
and writes predictions.csv with human-readable reasons.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import time

import numpy as np
import pandas as pd

from src.data.loader import load_all, get_active_gateways
from src.data.features import build_features_for_week
from src.model.registry import load_model
from src.model.validation_gate import run_pre_prediction_checks
from src.utils.config import load_config, get_data_dir, get_model_dir
from src.utils.audit import log_pipeline_event
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)

# Scored weeks (from the brief)
SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]
VISITS_PER_WEEK = 15


def _generate_reason(
    gateway_id: str,
    risk_score: float,
    features: pd.Series,
    feature_importance: dict[str, float],
    max_chars: int = 300,
) -> str:
    """Generate a human-readable reason for why a gateway was selected.

    Aimed at the operations manager, not at engineers.
    Max 300 characters per the submission format.
    """
    # Find the top contributing features for this gateway
    contributions = {}
    for feat_name, importance in feature_importance.items():
        if feat_name in features.index:
            val = features[feat_name]
            if val != 0 and importance > 0:
                contributions[feat_name] = abs(val) * importance

    # Sort by contribution
    top_features = sorted(contributions.items(), key=lambda x: -x[1])[:3]

    # Human-friendly feature name mapping
    friendly = {
        "offline_duration_sec_mean": "offline time",
        "offline_duration_sec_max": "peak offline time",
        "offline_duration_sec_trend": "worsening offline trend",
        "offline_duration_sec_sum": "total offline time",
        "disconnection_cnt_mean": "disconnections",
        "disconnection_cnt_max": "peak disconnections",
        "disconnection_cnt_trend": "worsening disconnections",
        "disconnection_cnt_sum": "total disconnections",
        "reboot_cnt_sum": "reboots",
        "reboot_cnt_max": "peak reboots",
        "reboot_cnt_trend": "increasing reboots",
        "reboot_importance_max": "reboot severity",
        "reboot_duration_sum": "reboot downtime",
        "power_cycle_ratio": "hardware reboot ratio",
        "last_read_rate": "low meter read rate",
        "read_rate_trend_4w": "declining meter reads",
        "meters_at_risk": "meters at risk",
        "n_past_visits": "repeated visits needed",
        "fault_rate": "high historical fault rate",
        "days_since_last_visit": "long time since last check",
        "no_conn_importance_max": "connection severity score",
        "rssi_bad_ratio": "poor signal quality",
        "crc_error_rate": "radio errors",
        "memfree_min": "low memory",
        "uptime_min": "low uptime (frequent restarts)",
        "offline_duration_sec_3sigma_hours": "anomalous offline hours",
        "disconnection_cnt_3sigma_hours": "anomalous disconnection hours",
        "reboot_cnt_3sigma_hours": "anomalous reboot hours",
    }

    if not top_features:
        return f"Risk score {risk_score:.2f}; elevated anomaly indicators in trailing 28-day window"

    reasons = []
    for feat_name, _ in top_features:
        name = friendly.get(feat_name, feat_name.replace("_", " "))
        val = features[feat_name]
        # Format value based on feature type
        if "trend" in feat_name:
            if val > 1.1:
                reasons.append(f"{name} ({val:.1f}x)")
            elif val < 0.9:
                reasons.append(f"{name} (declining)")
            else:
                reasons.append(name)
        elif feat_name in ("fault_rate", "power_cycle_ratio", "crc_error_rate",
                          "rssi_bad_ratio", "rscp_bad_ratio", "ecio_bad_ratio",
                          "last_read_rate", "read_rate"):
            reasons.append(f"{name} ({val:.0%})")
        elif feat_name in ("meters_at_risk", "n_past_visits", "n_faults_found",
                          "reboot_cnt_sum", "disconnection_cnt_sum"):
            reasons.append(f"{name} ({int(val)})")
        elif feat_name == "days_since_last_visit":
            reasons.append(f"{name} ({int(val)} days)")
        elif feat_name == "firmware_age_days" or feat_name == "gateway_age_days":
            reasons.append(f"{name} ({int(val)} days)")
        elif abs(val) > 10000:
            reasons.append(name)  # Don't show huge raw values
        elif isinstance(val, float) and val != int(val):
            reasons.append(f"{name} ({val:.1f})")
        else:
            reasons.append(f"{name} ({int(val)})")

    reason = f"Risk {risk_score:.2f}: " + "; ".join(reasons)

    # Truncate to max_chars
    if len(reason) > max_chars:
        reason = reason[:max_chars - 3] + "..."

    return reason


def predict_week(
    booster,
    feature_names: list[str],
    feature_importance: dict[str, float],
    features: pd.DataFrame,
    monday: dt.date,
    visits_per_week: int = VISITS_PER_WEEK,
) -> pd.DataFrame:
    """Generate ranked predictions for a single week.

    Args:
        booster: Trained LightGBM model.
        feature_names: Ordered feature names the model expects.
        feature_importance: Feature importance dict for reason generation.
        features: Per-gateway feature DataFrame (gateway_id as index).
        monday: The Monday of the prediction week.
        visits_per_week: Number of gateways to select.

    Returns:
        DataFrame with columns [week_start, rank, gateway_id, score, reason].
    """
    if features.empty:
        logger.warning("No features for week %s — cannot predict", monday)
        return pd.DataFrame(columns=["week_start", "rank", "gateway_id", "score", "reason"])

    # Ensure features are in the correct order, fill missing with 0
    X = features.reindex(columns=feature_names, fill_value=0).astype(float)

    # Predict risk scores
    risk_scores = booster.predict(X)

    # Rank by risk score (descending) and take top-N
    features_with_score = features.copy()
    features_with_score["risk_score"] = risk_scores

    ranked = features_with_score.sort_values("risk_score", ascending=False)

    if len(ranked) < visits_per_week:
        logger.warning("Only %d gateways available for week %s (need %d)",
                       len(ranked), monday, visits_per_week)

    top_n = ranked.head(visits_per_week)

    # Build prediction rows
    rows = []
    for rank, (gateway_id, row) in enumerate(top_n.iterrows(), 1):
        reason = _generate_reason(
            gateway_id=gateway_id,
            risk_score=row["risk_score"],
            features=row.drop("risk_score"),
            feature_importance=feature_importance,
        )
        rows.append({
            "week_start": monday.isoformat(),
            "rank": rank,
            "gateway_id": gateway_id,
            "score": float(row["risk_score"]),
            "reason": reason,
        })

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Main prediction entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate gateway health predictions")
    parser.add_argument("--data", type=pathlib.Path, default=None,
                        help="Path to data directory")
    parser.add_argument("--model-dir", type=pathlib.Path, default=None,
                        help="Path to model directory")
    parser.add_argument("--version", type=str, default=None,
                        help="Specific model version to use (default: current)")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions.csv"),
                        help="Output predictions file path")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Path to config YAML")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip pre-prediction validation checks")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_dir = args.data or get_data_dir(config)
    model_dir = args.model_dir or get_model_dir(config)

    logger.info("=== Prediction pipeline started ===")
    start_time = time.time()

    # 1. Load model
    booster, metadata, feature_names = load_model(model_dir, args.version)

    # Load feature importance
    import json
    feat_path = (model_dir / "current" / "features.json")
    if feat_path.exists():
        with open(feat_path) as f:
            feat_info = json.load(f)
        feature_importance = feat_info.get("feature_importance", {})
    else:
        feature_importance = {}

    logger.info("Loaded model %s (%d trees, %d features)",
                metadata.get("version", "unknown"),
                booster.num_trees(), len(feature_names))

    # 1b. Pre-prediction validation gate
    if not args.skip_checks:
        passed, report = run_pre_prediction_checks(data_dir, model_dir, config)
        logger.info("Validation gate:\n%s", report)
        if not passed:
            logger.error("Pre-prediction validation FAILED. Use --skip-checks to bypass.")
            return 1
    else:
        logger.info("Skipping pre-prediction validation checks")

    # 2. Load data
    bundle = load_all(data_dir)

    # 3. Generate predictions for each scored week
    all_predictions = []
    for monday in SCORED_WEEKS:
        # Get active gateways as of this Monday
        active_ids = get_active_gateways(bundle.gateway_master, as_of=monday)

        features = build_features_for_week(
            bundle.telemetry,
            bundle.gateway_master,
            bundle.field_visits,
            bundle.meter_reads,
            monday,
            active_gateway_ids=active_ids,
        )

        week_predictions = predict_week(
            booster, feature_names, feature_importance,
            features, monday,
        )
        all_predictions.append(week_predictions)

    # 4. Combine and save
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.out, index=False)

    elapsed = time.time() - start_time
    logger.info("=== Predictions written to %s — %d rows over %d weeks (%.1fs) ===",
                args.out, len(predictions),
                predictions["week_start"].nunique(), elapsed)

    # Audit trail
    log_pipeline_event("predict", {
        "model_version": metadata.get("version", "unknown"),
        "n_rows": len(predictions),
        "n_weeks": int(predictions["week_start"].nunique()),
        "output_file": str(args.out),
        "elapsed_seconds": round(elapsed, 1),
        "status": "success",
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
