"""Model comparison — side-by-side analysis of model versions.

Enables comparing any two model versions on metrics, config, and features.
Also provides a chronological experiment history view.

Usage:
    python -m src.model.compare v1.0.0 v1.6.0   # Compare two versions
    python -m src.model.compare --history         # Show experiment history
"""

from __future__ import annotations

import argparse
import json
import pathlib

from src.model.experiment_tracker import load_experiment_log, format_experiment_history
from src.utils.config import get_model_dir
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)


def _load_version_metadata(model_dir: pathlib.Path, version: str) -> dict:
    """Load metadata and features for a specific model version."""
    version_dir = model_dir / version

    if not version_dir.exists():
        raise FileNotFoundError(f"Model version not found: {version}")

    result = {"version": version}

    # Load metadata
    meta_path = version_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["metadata"] = json.load(f)

    # Load features
    feat_path = version_dir / "features.json"
    if feat_path.exists():
        with open(feat_path) as f:
            result["features"] = json.load(f)

    return result


def compare_versions(
    model_dir: pathlib.Path,
    version_a: str,
    version_b: str,
) -> dict:
    """Compare two model versions and compute deltas.

    Args:
        model_dir: Root model directory.
        version_a: First version (typically older).
        version_b: Second version (typically newer).

    Returns:
        Dict with both versions' data and computed deltas.
    """
    a = _load_version_metadata(model_dir, version_a)
    b = _load_version_metadata(model_dir, version_b)

    # Extract CV metrics
    cv_a = a.get("metadata", {}).get("cv_metrics", {})
    cv_b = b.get("metadata", {}).get("cv_metrics", {})

    # Compute metric deltas
    metric_keys = ["cv_auc_roc_mean", "cv_auc_pr_mean", "cv_logloss_mean",
                   "cv_caught_mean", "cv_missed_cost_mean", "cv_recall_at_15_mean",
                   "final_num_trees"]
    deltas = {}
    for key in metric_keys:
        val_a = cv_a.get(key, 0)
        val_b = cv_b.get(key, 0)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            deltas[key] = val_b - val_a

    # Config differences
    config_a = a.get("metadata", {}).get("config", {})
    config_b = b.get("metadata", {}).get("config", {})
    config_changes = {}
    all_keys = set(list(config_a.keys()) + list(config_b.keys()))
    for key in sorted(all_keys):
        va = config_a.get(key)
        vb = config_b.get(key)
        if va != vb:
            config_changes[key] = {"from": va, "to": vb}

    # Feature count comparison
    n_feat_a = a.get("features", {}).get("n_features", 0)
    n_feat_b = b.get("features", {}).get("n_features", 0)

    # Top feature importance shifts
    imp_a = a.get("features", {}).get("feature_importance", {})
    imp_b = b.get("features", {}).get("feature_importance", {})
    top_a = list(imp_a.keys())[:5] if imp_a else []
    top_b = list(imp_b.keys())[:5] if imp_b else []

    return {
        "version_a": version_a,
        "version_b": version_b,
        "metrics_a": cv_a,
        "metrics_b": cv_b,
        "deltas": deltas,
        "config_changes": config_changes,
        "n_features": {"a": n_feat_a, "b": n_feat_b},
        "top_features_a": top_a,
        "top_features_b": top_b,
    }


def format_comparison(comparison: dict) -> str:
    """Format a comparison result as a readable report."""
    va = comparison["version_a"]
    vb = comparison["version_b"]
    ma = comparison["metrics_a"]
    mb = comparison["metrics_b"]
    deltas = comparison["deltas"]

    lines = []
    lines.append("=" * 70)
    lines.append(f"MODEL COMPARISON: {va} vs {vb}")
    lines.append("=" * 70)

    # Metrics table
    lines.append("")
    lines.append(f"{'Metric':<25} {va:>12} {vb:>12} {'Delta':>10}")
    lines.append("-" * 70)

    metric_display = [
        ("AUC-ROC", "cv_auc_roc_mean", ".4f"),
        ("AUC-PR", "cv_auc_pr_mean", ".4f"),
        ("Log Loss", "cv_logloss_mean", ".4f"),
        ("Caught (avg/fold)", "cv_caught_mean", ".1f"),
        ("Missed Cost (avg)", "cv_missed_cost_mean", ",.0f"),
        ("Recall@15", "cv_recall_at_15_mean", ".4f"),
        ("Num Trees", "final_num_trees", "d"),
    ]

    for label, key, fmt in metric_display:
        val_a = ma.get(key, 0)
        val_b = mb.get(key, 0)
        delta = deltas.get(key, 0)
        delta_str = f"{delta:+{fmt}}" if isinstance(delta, (int, float)) else "—"

        # Mark improvement with arrow
        if key == "cv_logloss_mean" or key == "cv_missed_cost_mean":
            marker = " ↓✓" if delta < 0 else " ↑✗" if delta > 0 else ""
        elif key == "final_num_trees":
            marker = ""
        else:
            marker = " ↑✓" if delta > 0 else " ↓✗" if delta < 0 else ""

        lines.append(
            f"{label:<25} {val_a:>12{fmt}} {val_b:>12{fmt}} {delta_str:>10}{marker}"
        )

    # Config changes
    if comparison["config_changes"]:
        lines.append("")
        lines.append("Config Changes:")
        for key, change in comparison["config_changes"].items():
            lines.append(f"  {key}: {change['from']} → {change['to']}")
    else:
        lines.append("")
        lines.append("Config: No changes between versions")

    # Feature counts
    nf = comparison["n_features"]
    lines.append("")
    lines.append(f"Features: {nf['a']} ({va}) → {nf['b']} ({vb})")

    # Top features
    if comparison["top_features_b"]:
        lines.append("")
        lines.append(f"Top 5 features ({vb}):")
        for i, feat in enumerate(comparison["top_features_b"], 1):
            lines.append(f"  {i}. {feat}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for model comparison."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Compare model versions")
    parser.add_argument("versions", nargs="*",
                        help="Two versions to compare (e.g., v1.0.0 v1.6.0)")
    parser.add_argument("--history", action="store_true",
                        help="Show experiment history")
    parser.add_argument("--model-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    model_dir = args.model_dir or get_model_dir()

    if args.history or not args.versions:
        experiments = load_experiment_log(model_dir)
        print(format_experiment_history(experiments))
        return 0

    if len(args.versions) != 2:
        print("Error: Provide exactly two versions to compare.")
        print("Usage: python -m src.model.compare v1.0.0 v1.6.0")
        return 1

    try:
        comparison = compare_versions(model_dir, args.versions[0], args.versions[1])
        print(format_comparison(comparison))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
