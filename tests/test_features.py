"""Tests for feature engineering pipeline."""

import datetime as dt

import pandas as pd
import pytest

from src.data.loader import load_all
from src.data.features import (
    compute_telemetry_features,
    compute_meter_features,
    compute_gateway_features,
    compute_visit_features,
    build_features_for_week,
)


DATA_DIR = "data"


@pytest.fixture(scope="module")
def data_bundle():
    """Load all data once for the feature test module."""
    return load_all(DATA_DIR)


class TestTelemetryFeatures:
    """Telemetry feature computation."""

    def test_produces_features(self, data_bundle):
        feat = compute_telemetry_features(data_bundle.telemetry, dt.date(2026, 1, 6))
        assert len(feat) > 0
        assert len(feat.columns) > 20  # Should have many features

    def test_no_nan_in_output(self, data_bundle):
        feat = compute_telemetry_features(data_bundle.telemetry, dt.date(2026, 1, 6))
        assert not feat.isna().any().any(), f"NaN found in: {feat.columns[feat.isna().any()].tolist()}"

    def test_trend_features_exist(self, data_bundle):
        feat = compute_telemetry_features(data_bundle.telemetry, dt.date(2026, 1, 6))
        trend_cols = [c for c in feat.columns if "trend" in c]
        assert len(trend_cols) >= 3  # At least connectivity trends


class TestMeterFeatures:
    """Meter read feature computation."""

    def test_produces_features(self, data_bundle):
        feat = compute_meter_features(data_bundle.meter_reads, dt.date(2026, 1, 6))
        assert len(feat) > 0
        assert "last_read_rate" in feat.columns
        assert "meters_at_risk" in feat.columns

    def test_staleness_computed(self, data_bundle):
        feat = compute_meter_features(data_bundle.meter_reads, dt.date(2026, 2, 15))
        assert "meter_data_staleness_days" in feat.columns
        # By Feb 15, data is at least 2 weeks stale (last data: Jan 26)
        assert feat["meter_data_staleness_days"].mean() > 14


class TestGatewayFeatures:
    """Static gateway feature computation."""

    def test_produces_features(self, data_bundle):
        feat = compute_gateway_features(data_bundle.gateway_master, dt.date(2026, 2, 2))
        assert len(feat) == len(data_bundle.gateway_master)
        assert "gateway_age_days" in feat.columns
        assert "n_meters_installed" in feat.columns

    def test_categorical_encoding(self, data_bundle):
        feat = compute_gateway_features(data_bundle.gateway_master, dt.date(2026, 2, 2))
        code_cols = [c for c in feat.columns if "_code" in c]
        assert len(code_cols) >= 3  # hw_model, site_type, region at minimum


class TestVisitFeatures:
    """Field visit history features."""

    def test_produces_features(self, data_bundle):
        feat = compute_visit_features(data_bundle.field_visits, dt.date(2026, 2, 2))
        assert len(feat) > 0
        assert "n_past_visits" in feat.columns
        assert "fault_rate" in feat.columns

    def test_no_future_leakage(self, data_bundle):
        """Visit features should not use data from after the prediction date."""
        cutoff = dt.date(2025, 10, 1)
        feat = compute_visit_features(data_bundle.field_visits, cutoff)
        # days_since_last_visit should be > 0 (no future visits)
        assert (feat["days_since_last_visit"] >= 0).all()


class TestCombinedFeatures:
    """Full feature pipeline."""

    def test_build_features_for_week(self, data_bundle):
        feat = build_features_for_week(
            data_bundle.telemetry,
            data_bundle.gateway_master,
            data_bundle.field_visits,
            data_bundle.meter_reads,
            dt.date(2026, 2, 2),
            active_gateway_ids=data_bundle.active_gateway_ids,
        )
        assert len(feat) > 0
        assert len(feat.columns) >= 50  # Should have many features
        assert not feat.isna().any().any()

    def test_decommissioned_excluded(self, data_bundle):
        """Decommissioned gateways should not appear in features."""
        feat = build_features_for_week(
            data_bundle.telemetry,
            data_bundle.gateway_master,
            data_bundle.field_visits,
            data_bundle.meter_reads,
            dt.date(2026, 3, 1),
            active_gateway_ids=data_bundle.active_gateway_ids,
        )
        # Should have max 320 gateways (332 - 12 decommissioned)
        assert len(feat) <= 320

    def test_feature_schema_consistent(self, data_bundle):
        """Two different weeks should produce the same feature columns."""
        feat1 = build_features_for_week(
            data_bundle.telemetry, data_bundle.gateway_master,
            data_bundle.field_visits, data_bundle.meter_reads,
            dt.date(2026, 1, 6), data_bundle.active_gateway_ids,
        )
        feat2 = build_features_for_week(
            data_bundle.telemetry, data_bundle.gateway_master,
            data_bundle.field_visits, data_bundle.meter_reads,
            dt.date(2026, 2, 2), data_bundle.active_gateway_ids,
        )
        assert set(feat1.columns) == set(feat2.columns)
