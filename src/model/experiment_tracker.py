"""Experiment tracker — logs every training run for model evolution tracking.

Records training metadata, metrics, and config snapshots in a JSONL file.
Each line is a complete JSON record of one training experiment.

This enables:
  - Comparing model versions over time
  - Understanding what changed between versions
  - Auditing the model development process

MLOps requirement: experiment reproducibility and model lineage tracking.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def log_experiment(
    model_dir: pathlib.Path,
    version: str,
    metadata: dict,
    elapsed_seconds: float,
) -> None:
    """Append an experiment record to the experiment log.

    Args:
        model_dir: Root model directory (where experiment_log.jsonl lives).
        version: Model version string (e.g., "v1.6.0").
        metadata: Full training metadata dict (from train.py).
        elapsed_seconds: Total training time in seconds.
    """
    model_dir = pathlib.Path(model_dir)
    log_path = model_dir / "experiment_log.jsonl"

    # Extract the most important metrics for quick scanning
    cv = metadata.get("cv_metrics", {})

    record = {
        "version": version,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_seconds": round(elapsed_seconds, 1),
        "n_samples": metadata.get("training_samples", 0),
        "n_features": metadata.get("n_features", 0),
        "positive_rate": round(metadata.get("positive_rate", 0), 4),
        "data_hash": metadata.get("data_hash", ""),
        "random_seed": metadata.get("random_seed", 42),
        "metrics": {
            "auc_roc": round(cv.get("cv_auc_roc_mean", 0), 4),
            "auc_pr": round(cv.get("cv_auc_pr_mean", 0), 4),
            "logloss": round(cv.get("cv_logloss_mean", 0), 4),
            "caught_mean": round(cv.get("cv_caught_mean", 0), 1),
            "missed_cost_mean": round(cv.get("cv_missed_cost_mean", 0), 0),
            "recall_at_15": round(cv.get("cv_recall_at_15_mean", 0), 4),
            "num_trees": cv.get("final_num_trees", 0),
        },
        "config_snapshot": {
            "learning_rate": metadata.get("config", {}).get("learning_rate"),
            "num_leaves": metadata.get("config", {}).get("num_leaves"),
            "scale_pos_weight": metadata.get("config", {}).get("scale_pos_weight"),
            "min_child_samples": metadata.get("config", {}).get("min_child_samples"),
            "subsample": metadata.get("config", {}).get("subsample"),
        },
    }

    # Append to JSONL file
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("Experiment logged: %s (AUC-ROC=%.4f, %d trees, %.1fs)",
                version, record["metrics"]["auc_roc"],
                record["metrics"]["num_trees"], elapsed_seconds)


def load_experiment_log(model_dir: pathlib.Path) -> list[dict]:
    """Load all experiments from the experiment log.

    Returns:
        List of experiment records, oldest first.
    """
    model_dir = pathlib.Path(model_dir)
    log_path = model_dir / "experiment_log.jsonl"

    if not log_path.exists():
        return []

    experiments = []
    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in experiment log",
                               line_num)

    return experiments


def format_experiment_history(experiments: list[dict]) -> str:
    """Format experiment history as a readable table.

    Returns:
        Multi-line string with experiment summary.
    """
    if not experiments:
        return "No experiments recorded yet. Run 'make train' to create one."

    lines = []
    lines.append("=" * 90)
    lines.append("EXPERIMENT HISTORY")
    lines.append("=" * 90)
    lines.append(
        f"{'Version':<10} {'AUC-ROC':>8} {'Trees':>6} {'Caught':>7} "
        f"{'Features':>9} {'Time':>7} {'Date':>12}"
    )
    lines.append("-" * 90)

    for exp in experiments:
        m = exp.get("metrics", {})
        ts = exp.get("timestamp", "")[:10]  # Just the date part
        lines.append(
            f"{exp.get('version', '?'):<10} "
            f"{m.get('auc_roc', 0):>8.4f} "
            f"{m.get('num_trees', 0):>6} "
            f"{m.get('caught_mean', 0):>7.1f} "
            f"{exp.get('n_features', 0):>9} "
            f"{exp.get('training_seconds', 0):>6.1f}s "
            f"{ts:>12}"
        )

    lines.append("=" * 90)
    lines.append(f"Total experiments: {len(experiments)}")

    return "\n".join(lines)
