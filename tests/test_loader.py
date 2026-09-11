"""Tests for data loading, schema validation, and ID normalization."""

import pathlib

import pytest
import pandas as pd

from src.utils.gateway_ids import normalize_gateway_id, normalize_id_column
from src.data.loader import (
    load_all, load_telemetry, load_gateway_master,
    load_field_visits, load_meter_reads, load_engineer_review,
    get_active_gateways,
)


# ── ID Normalization ──────────────────────────────────────────────────

class TestGatewayIDNormalization:
    """Gateway ID normalization across formats."""

    def test_bare_hex_passthrough(self):
        assert normalize_gateway_id("0639EA5602C1") == "0639EA5602C1"

    def test_colon_separated_to_bare(self):
        assert normalize_gateway_id("06:39:EA:56:02:C1") == "0639EA5602C1"

    def test_lowercase_to_upper(self):
        assert normalize_gateway_id("0639ea5602c1") == "0639EA5602C1"

    def test_colon_lowercase_to_upper(self):
        assert normalize_gateway_id("06:39:ea:56:02:c1") == "0639EA5602C1"

    def test_whitespace_stripped(self):
        assert normalize_gateway_id(" 0639EA5602C1 ") == "0639EA5602C1"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            normalize_gateway_id("not-a-gateway-id")

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            normalize_gateway_id("0639EA")

    def test_series_normalization(self):
        s = pd.Series(["06:39:EA:56:02:C1", "0E5DFCF65AD4"])
        result = normalize_id_column(s)
        assert result.iloc[0] == "0639EA5602C1"
        assert result.iloc[1] == "0E5DFCF65AD4"


# ── Data Loading ──────────────────────────────────────────────────────

DATA_DIR = pathlib.Path("data")


class TestDataLoading:
    """Integration tests for data loading."""

    def test_load_telemetry_shape(self):
        df = load_telemetry(DATA_DIR)
        assert len(df) > 1_000_000
        assert "gateway_id" in df.columns
        assert "ts" in df.columns
        # IDs should be normalized (no colons)
        assert ":" not in df["gateway_id"].iloc[0]

    def test_load_gateway_master(self):
        df = load_gateway_master(DATA_DIR)
        assert len(df) == 332
        assert ":" not in df["gateway_id"].iloc[0]  # Normalized
        assert df["decommissioned_on"].notna().sum() == 12

    def test_load_field_visits(self):
        df = load_field_visits(DATA_DIR)
        assert len(df) == 642
        assert ":" not in df["gateway_id"].iloc[0]  # Normalized
        # Check outcomes are German text
        outcomes = set(df["outcome"].unique())
        assert "Fehler behoben" in outcomes or "Kein Fehler gefunden" in outcomes

    def test_load_meter_reads(self):
        df = load_meter_reads(DATA_DIR)
        assert len(df) == 7226
        assert "read_rate" in df.columns
        assert df["read_rate"].between(0, 1.01).all()  # Allow tiny float overflow

    def test_load_engineer_review(self):
        df = load_engineer_review(DATA_DIR)
        assert df is not None
        assert len(df) == 120
        assert df["is_bad"].sum() == 60  # 60 Schlecht

    def test_load_all_bundle(self):
        bundle = load_all(DATA_DIR)
        assert len(bundle.active_gateway_ids) == 320
        assert bundle.telemetry is not None
        assert bundle.gateway_master is not None
        assert bundle.field_visits is not None
        assert bundle.meter_reads is not None

    def test_gateway_ids_consistent_across_datasets(self):
        """All datasets use the same normalized ID format after loading."""
        bundle = load_all(DATA_DIR)
        tel_ids = set(bundle.telemetry["gateway_id"].unique())
        gw_ids = set(bundle.gateway_master["gateway_id"])
        mr_ids = set(bundle.meter_reads["gateway_id"].unique())

        # Telemetry gateways should be a subset of master
        assert tel_ids <= gw_ids

        # All IDs should be 12-char hex (no colons)
        for gid in list(tel_ids)[:10]:
            assert len(gid) == 12
            assert ":" not in gid


# ── Active Gateway Filtering ─────────────────────────────────────────

class TestActiveGateways:
    """Decommissioned gateway filtering."""

    def test_active_gateways_excludes_decommissioned(self):
        gw = load_gateway_master(DATA_DIR)
        active = get_active_gateways(gw)
        assert len(active) == 320  # 332 - 12 = 320

    def test_active_gateways_as_of_date(self):
        """Gateways decommissioned after the date should still be active."""
        import datetime as dt
        gw = load_gateway_master(DATA_DIR)
        # Before any decommissioning (Aug 2025)
        active_early = get_active_gateways(gw, as_of=dt.date(2025, 8, 1))
        assert len(active_early) == 332  # All active
        # During scored window (Feb 2026)
        active_scored = get_active_gateways(gw, as_of=dt.date(2026, 2, 15))
        assert len(active_scored) < 332  # Some decommissioned
