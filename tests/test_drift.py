"""Tests for data drift detection."""

import pandas as pd
import numpy as np
import pytest

from src.data.drift import (
    DriftReport,
    check_schema_drift,
    check_statistical_drift,
    check_volume_drift,
)


class TestSchemaDrift:
    """Schema drift detection catches missing/extra columns."""

    def test_no_drift_when_matching(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        checks = check_schema_drift(df, ["a", "b", "c"], "test")
        assert all(c["status"] == "PASS" for c in checks)

    def test_detects_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        checks = check_schema_drift(df, ["a", "b", "c", "d"], "test")
        critical = [c for c in checks if c["severity"] == "CRITICAL"]
        assert len(critical) > 0
        assert "Missing" in critical[0]["detail"]

    def test_detects_extra_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "extra": [4]})
        checks = check_schema_drift(df, ["a", "b", "c"], "test")
        info = [c for c in checks if c["severity"] == "INFO"]
        assert len(info) > 0


class TestStatisticalDrift:
    """Statistical drift detection using KS-test."""

    def test_no_drift_same_distribution(self):
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        cur = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        checks = check_statistical_drift(ref, cur, ["x"], ks_threshold=0.01)
        passed = [c for c in checks if c["status"] == "PASS"]
        assert len(passed) > 0

    def test_detects_distribution_shift(self):
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        cur = pd.DataFrame({"x": np.random.normal(5, 1, 1000)})  # Shifted!
        checks = check_statistical_drift(ref, cur, ["x"], ks_threshold=0.1)
        drifted = [c for c in checks if c["status"] == "DRIFT"]
        assert len(drifted) > 0


class TestVolumeDrift:
    """Volume drift detection — unexpected gateway count."""

    def test_no_drift_expected_count(self):
        df = pd.DataFrame({"gateway_id": [f"GW{i}" for i in range(320)]})
        checks = check_volume_drift(df, expected_gateways=320)
        passed = [c for c in checks if c["status"] == "PASS"]
        assert len(passed) > 0

    def test_detects_missing_gateways(self):
        df = pd.DataFrame({"gateway_id": [f"GW{i}" for i in range(100)]})
        checks = check_volume_drift(df, expected_gateways=320)
        drifted = [c for c in checks if c["status"] == "DRIFT"]
        assert len(drifted) > 0


class TestDriftReport:
    """Drift report aggregation."""

    def test_severity_escalation(self):
        report = DriftReport()
        report.add_check("check1", "PASS", "ok", "OK")
        assert report.severity == "OK"
        report.add_check("check2", "DRIFT", "warning", "WARNING")
        assert report.severity == "WARNING"
        report.add_check("check3", "FAIL", "critical", "CRITICAL")
        assert report.severity == "CRITICAL"

    def test_to_dict(self):
        report = DriftReport()
        report.add_check("test", "PASS", "ok", "OK")
        d = report.to_dict()
        assert "overall_severity" in d
        assert "checks" in d
        assert len(d["checks"]) == 1
