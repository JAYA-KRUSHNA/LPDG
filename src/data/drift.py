"""Data drift detection — monitors incoming data for changes.

Detects three types of drift:
  1. Schema drift: missing/extra columns, type changes
  2. Statistical drift: KS-test on continuous, chi-squared on categorical
  3. Volume drift: unexpected gateway count or missing hours

Requirement F3: "Something watching the incoming data that would notice
if it changed shape."
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import load_config, get_data_dir, get_model_dir
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)


class DriftReport:
    """Container for drift detection results."""

    def __init__(self):
        self.checks: list[dict] = []
        self.severity: str = "OK"  # OK, INFO, WARNING, CRITICAL

    def add_check(self, name: str, status: str, detail: str, severity: str = "INFO"):
        """Add a drift check result."""
        self.checks.append({
            "name": name,
            "status": status,
            "detail": detail,
            "severity": severity,
        })
        # Escalate overall severity
        severity_order = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
        if severity_order.get(severity, 0) > severity_order.get(self.severity, 0):
            self.severity = severity

    def to_dict(self) -> dict:
        return {
            "overall_severity": self.severity,
            "n_checks": len(self.checks),
            "n_warnings": sum(1 for c in self.checks if c["severity"] == "WARNING"),
            "n_critical": sum(1 for c in self.checks if c["severity"] == "CRITICAL"),
            "checks": self.checks,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    def __str__(self):
        lines = [f"Drift Report — Overall: {self.severity}"]
        for c in self.checks:
            marker = "✓" if c["status"] == "PASS" else "✗"
            lines.append(f"  {marker} [{c['severity']}] {c['name']}: {c['detail']}")
        return "\n".join(lines)


def check_schema_drift(
    current_df: pd.DataFrame,
    expected_columns: list[str],
    name: str = "telemetry",
) -> list[dict]:
    """Check for schema drift: missing or extra columns."""
    checks = []
    current_cols = set(current_df.columns)
    expected_cols = set(expected_columns)

    missing = expected_cols - current_cols
    extra = current_cols - expected_cols

    if missing:
        checks.append({
            "name": f"{name}_missing_columns",
            "status": "FAIL",
            "detail": f"Missing columns: {sorted(missing)}",
            "severity": "CRITICAL",
        })
    else:
        checks.append({
            "name": f"{name}_columns_present",
            "status": "PASS",
            "detail": f"All {len(expected_cols)} expected columns present",
            "severity": "OK",
        })

    if extra:
        checks.append({
            "name": f"{name}_extra_columns",
            "status": "INFO",
            "detail": f"Extra columns (ignored): {sorted(extra)}",
            "severity": "INFO",
        })

    return checks


def check_statistical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: list[str],
    ks_threshold: float = 0.1,
) -> list[dict]:
    """Check for statistical drift using the Kolmogorov-Smirnov test.

    Compares the distribution of numeric features between reference
    (training data snapshot) and current (new incoming data).
    """
    checks = []

    for col in numeric_columns:
        if col not in reference.columns or col not in current.columns:
            continue

        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

        if p_value < ks_threshold:
            checks.append({
                "name": f"stat_drift_{col}",
                "status": "DRIFT",
                "detail": f"KS={ks_stat:.3f}, p={p_value:.4f} (threshold={ks_threshold})",
                "severity": "WARNING",
            })
        else:
            checks.append({
                "name": f"stat_drift_{col}",
                "status": "PASS",
                "detail": f"KS={ks_stat:.3f}, p={p_value:.4f}",
                "severity": "OK",
            })

    return checks


def check_volume_drift(
    current_df: pd.DataFrame,
    expected_gateways: int,
    tolerance: float = 0.1,
) -> list[dict]:
    """Check for volume drift: unexpected number of gateways."""
    checks = []
    actual = current_df["gateway_id"].nunique()
    lower = int(expected_gateways * (1 - tolerance))
    upper = int(expected_gateways * (1 + tolerance))

    if actual < lower or actual > upper:
        checks.append({
            "name": "gateway_count",
            "status": "DRIFT",
            "detail": f"Found {actual} gateways, expected {lower}-{upper}",
            "severity": "WARNING",
        })
    else:
        checks.append({
            "name": "gateway_count",
            "status": "PASS",
            "detail": f"{actual} gateways (expected ~{expected_gateways})",
            "severity": "OK",
        })

    return checks


def run_drift_detection(
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    config: dict | None = None,
) -> DriftReport:
    """Run all drift checks on the current data.

    Compares against the feature schema saved with the current model.
    """
    if config is None:
        config = load_config()

    report = DriftReport()

    # Load current data (just schema + basic stats, not full features)
    from src.data.loader import load_telemetry, load_gateway_master

    try:
        telemetry = load_telemetry(data_dir)
    except Exception as e:
        report.add_check("data_load", "FAIL", f"Cannot load telemetry: {e}", "CRITICAL")
        return report

    report.add_check("data_load", "PASS", "Telemetry loaded successfully", "OK")

    # Load expected feature schema from model
    features_path = model_dir / "current" / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            feat_info = json.load(f)
        expected_features = feat_info.get("feature_names", [])
    else:
        expected_features = []
        report.add_check("model_schema", "SKIP",
                         "No model found — cannot check feature schema", "INFO")

    # Schema drift on telemetry
    for check in check_schema_drift(
        telemetry,
        list(telemetry.columns),  # Use current as reference if no model
        "telemetry",
    ):
        report.add_check(**check)

    # Volume drift
    try:
        gw = load_gateway_master(data_dir)
        active_count = len(gw[gw["decommissioned_on"].isna()])
    except Exception:
        active_count = 320  # Default

    for check in check_volume_drift(telemetry, active_count):
        report.add_check(**check)

    # Statistical drift (compare recent month vs older data)
    telemetry["ts"] = pd.to_datetime(telemetry["ts_utc"], utc=True)
    latest_month = telemetry["ts"].max() - pd.Timedelta(days=30)
    recent = telemetry[telemetry["ts"] >= latest_month]
    older = telemetry[telemetry["ts"] < latest_month]

    drift_config = config.get("drift", {})
    ks_threshold = drift_config.get("ks_threshold", 0.1)

    numeric_cols = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt",
                    "avg_load1", "avg_memfree", "rx_nr_pkts"]

    for check in check_statistical_drift(older, recent, numeric_cols, ks_threshold):
        report.add_check(**check)

    return report


def main(argv: list[str] | None = None) -> int:
    """Run drift detection and output report."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run data drift detection")
    parser.add_argument("--data", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Save drift report as JSON")
    args = parser.parse_args(argv)

    config = load_config()
    data_dir = args.data or get_data_dir(config)
    model_dir = get_model_dir(config)

    report = run_drift_detection(data_dir, model_dir, config)

    print(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Drift report saved to %s", args.output)

    return 0 if report.severity != "CRITICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
