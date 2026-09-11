"""Cost-based model evaluation — compares model predictions vs baseline.

Uses the cost structure from the brief:
  - Every visit costs €380 (fixed, 15 per week)
  - Every missed broken gateway costs €600 per week
  - Optimization: maximize broken gateways caught in top-15

Evaluates against engineer review (Feb 15, 2026) as ground truth.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from src.utils.config import load_config, get_data_dir
from src.utils.gateway_ids import normalize_gateway_id
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)


def load_engineer_truth(data_dir: pathlib.Path | str) -> dict[str, int]:
    """Load engineer review and return gateway_id -> is_bad mapping."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / "engineer_review_2026-02.xlsx"
    if not path.exists():
        logger.warning("Engineer review not found — cannot evaluate")
        return {}

    df = pd.read_excel(path)
    truth = {}
    for _, row in df.iterrows():
        gid = normalize_gateway_id(row["gateway_id"])
        truth[gid] = 1 if row["Kategorie"] == "Schlecht" else 0
    return truth


def evaluate_predictions(
    predictions: pd.DataFrame,
    truth: dict[str, int],
    visit_cost: int = 380,
    miss_cost: int = 600,
    eval_weeks: list[str] | None = None,
) -> dict:
    """Evaluate predictions against ground truth.

    Args:
        predictions: DataFrame with [week_start, rank, gateway_id, score, reason].
        truth: gateway_id -> is_bad (1=broken, 0=healthy).
        visit_cost: Cost per visit (€380).
        miss_cost: Cost per missed broken gateway per week (€600).
        eval_weeks: Weeks to evaluate (defaults to all weeks near engineer review).

    Returns:
        Dict with evaluation metrics.
    """
    # Normalize prediction gateway IDs
    predictions = predictions.copy()
    predictions["gateway_id"] = predictions["gateway_id"].apply(normalize_gateway_id)

    # If no specific weeks given, evaluate weeks near the engineer review (Feb 15)
    if eval_weeks is None:
        eval_weeks = ["2026-02-09", "2026-02-16"]

    all_bad_ids = {gid for gid, label in truth.items() if label == 1}
    all_known_ids = set(truth.keys())

    total_caught = 0
    total_missed = 0
    total_visits = 0
    total_false_alarms = 0
    week_results = []

    for week in eval_weeks:
        week_pred = predictions[predictions["week_start"] == week]
        if week_pred.empty:
            continue

        selected_ids = set(week_pred["gateway_id"])
        total_visits += len(selected_ids)

        # How many of the selected gateways are truly bad?
        caught = len(selected_ids & all_bad_ids)
        # How many bad gateways did we miss?
        missed = len(all_bad_ids - selected_ids)
        # How many selected gateways are healthy (false alarms)?
        false_alarms = len(selected_ids & (all_known_ids - all_bad_ids))
        # Unknown: selected gateways not in the review
        unknown = len(selected_ids - all_known_ids)

        total_caught += caught
        total_missed += missed
        total_false_alarms += false_alarms

        week_cost = len(selected_ids) * visit_cost + missed * miss_cost

        week_results.append({
            "week": week,
            "selected": len(selected_ids),
            "caught": caught,
            "missed": missed,
            "false_alarms": false_alarms,
            "unknown": unknown,
            "week_cost": week_cost,
        })

    n_eval_weeks = len(week_results)
    total_cost = sum(r["week_cost"] for r in week_results)
    avg_weekly_cost = total_cost / max(n_eval_weeks, 1)
    recall = total_caught / max(total_caught + total_missed, 1)

    return {
        "total_cost": total_cost,
        "avg_weekly_cost": avg_weekly_cost,
        "total_caught": total_caught,
        "total_missed": total_missed,
        "total_false_alarms": total_false_alarms,
        "recall": recall,
        "n_bad_gateways": len(all_bad_ids),
        "n_eval_weeks": n_eval_weeks,
        "weeks": week_results,
    }


def compare_model_vs_baseline(
    model_path: pathlib.Path,
    baseline_path: pathlib.Path,
    data_dir: pathlib.Path,
    config: dict | None = None,
) -> dict:
    """Compare model predictions against baseline predictions.

    Returns a comparison dict with both evaluation results and the delta.
    """
    if config is None:
        config = load_config()

    visit_cost = config["cost"]["visit_cost"]
    miss_cost = config["cost"]["miss_cost"]

    truth = load_engineer_truth(data_dir)
    if not truth:
        return {"error": "No engineer review data available for evaluation"}

    model_pred = pd.read_csv(model_path)
    baseline_pred = pd.read_csv(baseline_path)

    model_eval = evaluate_predictions(model_pred, truth, visit_cost, miss_cost)
    baseline_eval = evaluate_predictions(baseline_pred, truth, visit_cost, miss_cost)

    cost_delta = model_eval["total_cost"] - baseline_eval["total_cost"]
    cost_improvement_pct = (
        (baseline_eval["total_cost"] - model_eval["total_cost"])
        / max(baseline_eval["total_cost"], 1) * 100
    )

    comparison = {
        "model": model_eval,
        "baseline": baseline_eval,
        "cost_delta": cost_delta,
        "cost_improvement_pct": cost_improvement_pct,
        "model_beats_baseline": cost_delta < 0,
        "caught_delta": model_eval["total_caught"] - baseline_eval["total_caught"],
    }

    return comparison


def main(argv: list[str] | None = None) -> int:
    """Evaluate model vs baseline and print comparison."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate model vs baseline")
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path("predictions.csv"))
    parser.add_argument("--baseline", type=pathlib.Path,
                        default=pathlib.Path("predictions_baseline.csv"))
    parser.add_argument("--data", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    data_dir = args.data or get_data_dir(config)

    if not args.baseline.exists():
        logger.info("Generating baseline predictions...")
        import subprocess
        subprocess.run([
            "python", "baseline_3sigma.py",
            "--data", str(data_dir),
            "--out", str(args.baseline),
        ], check=True)

    result = compare_model_vs_baseline(args.model, args.baseline, data_dir, config)

    if "error" in result:
        logger.error(result["error"])
        return 1

    # Print comparison
    print("\n" + "=" * 60)
    print("COST COMPARISON: MODEL vs BASELINE")
    print("=" * 60)

    for name, eval_data in [("Baseline", result["baseline"]), ("Model", result["model"])]:
        print(f"\n{name}:")
        print(f"  Bad gateways caught: {eval_data['total_caught']} / {eval_data['n_bad_gateways']}")
        print(f"  Bad gateways missed: {eval_data['total_missed']}")
        print(f"  False alarms:        {eval_data['total_false_alarms']}")
        print(f"  Recall:              {eval_data['recall']:.1%}")
        print(f"  Total cost:          €{eval_data['total_cost']:,.0f}")
        print(f"  Avg weekly cost:     €{eval_data['avg_weekly_cost']:,.0f}")

    print(f"\n{'=' * 60}")
    if result["model_beats_baseline"]:
        print(f"✓ Model BEATS baseline by €{abs(result['cost_delta']):,.0f} "
              f"({result['cost_improvement_pct']:.1f}% improvement)")
        print(f"  Caught {result['caught_delta']} more bad gateways")
    else:
        print(f"✗ Model is WORSE than baseline by €{result['cost_delta']:,.0f}")
        print(f"  Caught {result['caught_delta']} fewer bad gateways")
    print("=" * 60)

    return 0 if result["model_beats_baseline"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
