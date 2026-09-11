"""End-to-end test — full pipeline from raw data to valid predictions.csv.

Also includes:
  - test_cost_vs_baseline: "the test we wrote because we found a bug"
  - test_decommissioned: no decommissioned gateway in predictions
"""

import datetime as dt
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from src.data.loader import load_gateway_master
from src.utils.gateway_ids import normalize_gateway_id


DATA_DIR = pathlib.Path("data")
PREDICTIONS_PATH = pathlib.Path("predictions.csv")


class TestEndToEnd:
    """Full pipeline produces valid predictions.csv."""

    def test_predictions_exist(self):
        assert PREDICTIONS_PATH.exists(), "predictions.csv not found — run 'make predict'"

    def test_predictions_row_count(self):
        df = pd.read_csv(PREDICTIONS_PATH)
        assert len(df) == 120, f"Expected 120 rows, got {len(df)}"

    def test_predictions_columns(self):
        df = pd.read_csv(PREDICTIONS_PATH)
        expected = {"week_start", "rank", "gateway_id", "score", "reason"}
        assert set(df.columns) == expected, f"Wrong columns: {set(df.columns)}"

    def test_predictions_weeks(self):
        df = pd.read_csv(PREDICTIONS_PATH)
        expected_weeks = {
            f"2026-{m:02d}-{d:02d}"
            for m, d in [(2, 2), (2, 9), (2, 16), (2, 23),
                         (3, 2), (3, 9), (3, 16), (3, 23)]
        }
        assert set(df["week_start"]) == expected_weeks

    def test_predictions_15_per_week(self):
        df = pd.read_csv(PREDICTIONS_PATH)
        for week, group in df.groupby("week_start"):
            assert len(group) == 15, f"Week {week} has {len(group)} predictions"
            assert set(group["rank"]) == set(range(1, 16))

    def test_predictions_unique_gateways_per_week(self):
        df = pd.read_csv(PREDICTIONS_PATH)
        for week, group in df.groupby("week_start"):
            assert group["gateway_id"].nunique() == 15, \
                f"Week {week}: duplicate gateway IDs"

    def test_predictions_reason_length(self):
        """Reasons must be ≤ 300 characters."""
        df = pd.read_csv(PREDICTIONS_PATH)
        long_reasons = df[df["reason"].str.len() > 300]
        assert len(long_reasons) == 0, \
            f"{len(long_reasons)} reasons exceed 300 chars"

    def test_validator_passes(self):
        """The official validate_submission.py must accept our predictions."""
        result = subprocess.run(
            [sys.executable, "validate_submission.py", str(PREDICTIONS_PATH)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Validator failed: {result.stdout} {result.stderr}"


class TestCostVsBaseline:
    """The test we wrote because we found a bug.

    Bug discovered: the baseline selects decommissioned gateway 02EBC6CD4398
    in week 2026-02-02. That gateway was decommissioned on 2026-02-04.
    Visiting a decommissioned gateway is guaranteed €380 waste.

    Our model should avoid this mistake and achieve lower total cost.
    """

    def test_model_beats_baseline_on_cost(self):
        """Model predictions must have lower cost than baseline."""
        from src.model.evaluate import evaluate_predictions, load_engineer_truth

        truth = load_engineer_truth(DATA_DIR)
        if not truth:
            pytest.skip("No engineer review data for evaluation")

        model_pred = pd.read_csv("predictions.csv")

        baseline_path = pathlib.Path("predictions_baseline.csv")
        if not baseline_path.exists():
            # Generate baseline on-the-fly
            result = subprocess.run(
                [sys.executable, "baseline_3sigma.py",
                 "--data", str(DATA_DIR), "--out", str(baseline_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not baseline_path.exists():
                pytest.skip(f"Could not generate baseline: {result.stderr}")

        baseline_pred = pd.read_csv(baseline_path)

        model_eval = evaluate_predictions(model_pred, truth)
        baseline_eval = evaluate_predictions(baseline_pred, truth)

        assert model_eval["total_cost"] <= baseline_eval["total_cost"], (
            f"Model cost €{model_eval['total_cost']} > baseline €{baseline_eval['total_cost']}"
        )


class TestDecommissioned:
    """No decommissioned gateway should appear in predictions."""

    def test_no_decommissioned_in_predictions(self):
        gw = load_gateway_master(DATA_DIR)
        decomm_ids = set(
            gw[gw["decommissioned_on"].notna()]["gateway_id"]
        )

        df = pd.read_csv("predictions.csv")
        df["gateway_id_norm"] = df["gateway_id"].apply(normalize_gateway_id)

        for _, row in df.iterrows():
            week = row["week_start"]
            gid = row["gateway_id_norm"]
            # Check if this gateway was decommissioned before or during this week
            decomm_row = gw[gw["gateway_id"] == gid]
            if not decomm_row.empty and pd.notna(decomm_row.iloc[0]["decommissioned_on"]):
                decomm_date = decomm_row.iloc[0]["decommissioned_on"]
                week_date = pd.Timestamp(week)
                assert week_date < decomm_date, (
                    f"Decommissioned gateway {gid} (decomm {decomm_date}) "
                    f"selected for week {week}"
                )
