"""Training pipeline — trains a LightGBM model for gateway health prediction.

Orchestrates: data loading -> feature engineering -> label construction ->
model training -> validation -> model saving with versioning.

Training and prediction are genuinely separate steps (MLOps requirement F1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.loader import load_all, get_active_gateways
from src.data.features import build_features_for_week
from src.data.labels import build_training_labels, get_training_weeks
from src.model.registry import save_model, get_next_version
from src.model.experiment_tracker import log_experiment
from src.utils.config import load_config, get_data_dir, get_model_dir
from src.utils.audit import log_pipeline_event
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)

# Scored weeks (from the brief)
SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]


def _compute_data_hash(features: pd.DataFrame, labels: pd.DataFrame) -> str:
    """Compute a hash of the training data for reproducibility tracking."""
    content = features.head(100).to_string() + labels.head(100).to_string()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def prepare_training_data(
    config: dict,
    data_dir: pathlib.Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load data, build features and labels for training.

    Returns:
        X: Feature matrix
        y: Binary labels
        groups: Gateway IDs (for group cross-validation)
        feature_names: List of feature column names
    """
    bundle = load_all(data_dir)

    # Get training weeks (before the scored window)
    training_mondays = get_training_weeks(bundle.field_visits)
    logger.info("Using %d training weeks", len(training_mondays))

    # Build features for each training week
    all_features = []
    all_labels_list = []

    # Get active gateways (exclude decommissioned)
    # For training, use all gateways that were active at some point
    active_ids = bundle.active_gateway_ids

    # Build labels from field visits
    labels_df = build_training_labels(
        bundle.field_visits,
        active_ids,
        training_mondays,
        lookahead_days=14,
    )

    # Build features for each training week
    for monday in training_mondays:
        # Get active gateways as of this Monday
        week_active = get_active_gateways(bundle.gateway_master, as_of=monday)

        feat = build_features_for_week(
            bundle.telemetry,
            bundle.gateway_master,
            bundle.field_visits,
            bundle.meter_reads,
            monday,
            active_gateway_ids=week_active,
        )
        feat["week_start"] = monday.isoformat()
        all_features.append(feat)

    if not all_features:
        raise ValueError("No training features could be built")

    # Combine features
    features_df = pd.concat(all_features).reset_index()

    # Merge with labels
    merged = features_df.merge(
        labels_df,
        on=["week_start", "gateway_id"],
        how="inner",
    )

    # Separate X, y, groups
    feature_cols = [c for c in merged.columns
                    if c not in ("week_start", "gateway_id", "label")]
    X = merged[feature_cols].astype(float)
    y = merged["label"].astype(int)
    groups = merged["gateway_id"]

    logger.info("Training data: %d samples, %d features, %.1f%% positive",
                len(X), len(feature_cols), 100 * y.mean())

    return X, y, groups, feature_cols


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: dict,
) -> tuple[lgb.Booster, dict]:
    """Train a LightGBM model with gateway-level cross-validation.

    Args:
        X: Feature matrix.
        y: Binary labels.
        groups: Gateway IDs for group-based splitting.
        config: Model configuration.

    Returns:
        Trained booster and validation metrics dict.
    """
    model_config = config["model"]
    seed = model_config["random_seed"]

    # LightGBM parameters
    params = {
        "objective": model_config["objective"],
        "metric": model_config["metric"],
        "boosting_type": model_config["boosting_type"],
        "num_leaves": model_config["num_leaves"],
        "learning_rate": model_config["learning_rate"],
        "min_child_samples": model_config["min_child_samples"],
        "subsample": model_config["subsample"],
        "colsample_bytree": model_config["colsample_bytree"],
        "reg_alpha": model_config["reg_alpha"],
        "reg_lambda": model_config["reg_lambda"],
        "scale_pos_weight": model_config["scale_pos_weight"],
        "seed": seed,
        "verbose": model_config["verbose"],
        "deterministic": True,
        "force_row_wise": True,
    }

    n_estimators = model_config["n_estimators"]
    early_stopping = model_config["early_stopping_rounds"]

    # Gateway-level cross-validation (requirement E3: test on unseen gateways)
    gkf = GroupKFold(n_splits=3)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Evaluate on validation fold
        y_pred = booster.predict(X_val)
        fold_metrics = _evaluate_fold(y_val, y_pred, config)
        fold_metrics["fold"] = fold
        fold_metrics["n_train"] = len(X_train)
        fold_metrics["n_val"] = len(X_val)
        fold_metrics["best_iteration"] = booster.best_iteration
        cv_metrics.append(fold_metrics)

        logger.info("Fold %d: %s", fold, {k: f"{v:.4f}" if isinstance(v, float) else v
                                           for k, v in fold_metrics.items()})

    # Train final model on all data
    train_set = lgb.Dataset(X, label=y)
    final_booster = lgb.train(
        params,
        train_set,
        num_boost_round=max(m["best_iteration"] for m in cv_metrics),
    )

    # Aggregate CV metrics
    avg_metrics = {}
    numeric_keys = [k for k in cv_metrics[0] if isinstance(cv_metrics[0][k], (int, float))]
    for key in numeric_keys:
        values = [m[key] for m in cv_metrics]
        avg_metrics[f"cv_{key}_mean"] = float(np.mean(values))
        avg_metrics[f"cv_{key}_std"] = float(np.std(values))

    avg_metrics["cv_folds"] = len(cv_metrics)
    avg_metrics["final_num_trees"] = final_booster.num_trees()

    logger.info("Final model: %d trees, CV metrics: %s",
                final_booster.num_trees(),
                {k: f"{v:.4f}" if isinstance(v, float) else v
                 for k, v in avg_metrics.items()})

    return final_booster, avg_metrics


def _evaluate_fold(y_true: pd.Series, y_pred: np.ndarray, config: dict) -> dict:
    """Evaluate a fold using cost-based and standard metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

    metrics = {}

    # Standard ML metrics
    metrics["logloss"] = float(log_loss(y_true, y_pred))

    if y_true.nunique() > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_pred))
    else:
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    # Cost-based evaluation: simulate picking top-15
    visits_per_week = config["cost"]["visits_per_week"]
    miss_cost = config["cost"]["miss_cost"]

    # Sort by predicted score descending, take top-15
    sorted_idx = np.argsort(-y_pred)
    selected = set(sorted_idx[:visits_per_week])
    true_positives = set(np.where(y_true == 1)[0])

    caught = len(selected & true_positives)
    missed = len(true_positives - selected)
    false_alarms = len(selected - true_positives)

    metrics["caught"] = caught
    metrics["missed"] = missed
    metrics["false_alarms"] = false_alarms
    metrics["missed_cost"] = missed * miss_cost
    metrics["recall_at_15"] = caught / max(len(true_positives), 1)

    return metrics


def main(argv: list[str] | None = None) -> int:
    """Main training entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Train gateway health model")
    parser.add_argument("--data", type=pathlib.Path, default=None,
                        help="Path to data directory")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Path to config YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_dir = args.data or get_data_dir(config)

    logger.info("=== Training pipeline started ===")
    start_time = time.time()

    # 1. Prepare training data
    X, y, groups, feature_names = prepare_training_data(config, data_dir)

    # 2. Train model
    booster, cv_metrics = train_model(X, y, groups, config)

    # 3. Compute feature importance
    importance = dict(zip(
        feature_names,
        booster.feature_importance(importance_type="gain").tolist(),
    ))
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    top_10 = dict(list(importance.items())[:10])
    logger.info("Top 10 features by gain: %s",
                {k: f"{v:.1f}" for k, v in top_10.items()})

    # 4. Save model with versioning
    model_dir = get_model_dir(config)
    version = get_next_version(model_dir)

    metadata = {
        "version": version,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_samples": len(X),
        "n_features": len(feature_names),
        "positive_rate": float(y.mean()),
        "config": config["model"],
        "cv_metrics": cv_metrics,
        "data_hash": _compute_data_hash(X, pd.DataFrame({"label": y})),
        "random_seed": config["model"]["random_seed"],
    }

    save_model(
        booster=booster,
        version=version,
        model_dir=model_dir,
        metadata=metadata,
        feature_names=feature_names,
        feature_importance=importance,
        config=config,
    )

    elapsed = time.time() - start_time
    logger.info("=== Training complete in %.1f seconds. Model saved as %s ===",
                elapsed, version)

    # Log experiment for tracking model evolution
    log_experiment(model_dir, version, metadata, elapsed)

    # Audit trail
    log_pipeline_event("train", {
        "version": version,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "data_hash": metadata["data_hash"],
        "elapsed_seconds": round(elapsed, 1),
        "status": "success",
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
