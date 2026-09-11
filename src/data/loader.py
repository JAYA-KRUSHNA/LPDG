"""Data loader — loads, validates, and normalizes all datasets.

Every downstream module uses this as the single entry point for data access.
Handles encoding differences, ID normalization, and schema validation.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from dataclasses import dataclass

import pandas as pd

from src.utils.config import load_config, get_data_dir
from src.utils.gateway_ids import normalize_id_column
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Expected minimum column sets for schema validation
_TELEMETRY_REQUIRED = {"gateway_id", "ts_utc", "offline_duration_sec",
                       "disconnection_cnt", "reboot_cnt"}
_GATEWAY_MASTER_REQUIRED = {"gateway_id", "hw_model", "site_type", "region",
                            "installed_on", "n_meters_installed"}
_FIELD_VISITS_REQUIRED = {"gateway_id", "requested_on", "outcome"}
_METER_READS_REQUIRED = {"gateway_id", "week_start", "meters_expected", "meters_read"}
_ENGINEER_REVIEW_REQUIRED = {"gateway_id", "Kategorie"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise if expected columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


@dataclass
class DataBundle:
    """Container for all loaded and normalized datasets."""
    telemetry: pd.DataFrame
    gateway_master: pd.DataFrame
    field_visits: pd.DataFrame
    meter_reads: pd.DataFrame
    engineer_review: pd.DataFrame | None  # May not exist in new data
    active_gateway_ids: set[str]          # Non-decommissioned gateway IDs


def load_telemetry(data_dir: pathlib.Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load telemetry from partitioned Parquet files.

    Args:
        data_dir: Path to the data/ directory.
        columns: Optional list of columns to load (for memory efficiency).

    Returns:
        DataFrame with normalized gateway_id and parsed timestamps.
    """
    data_dir = pathlib.Path(data_dir)
    telemetry_path = data_dir / "telemetry"
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_path}")

    # Always load gateway_id and ts_utc even if not in user's column list
    must_have = {"gateway_id", "ts_utc"}
    if columns is not None:
        load_cols = list(must_have | set(columns))
    else:
        load_cols = None

    df = pd.read_parquet(telemetry_path, columns=load_cols)
    _validate_columns(df, _TELEMETRY_REQUIRED if columns is None else must_have, "telemetry")

    df["gateway_id"] = normalize_id_column(df["gateway_id"])
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)

    logger.info("Loaded telemetry: %d rows, %d gateways, %s to %s",
                len(df), df["gateway_id"].nunique(),
                df["ts_utc"].min(), df["ts_utc"].max())
    return df


def load_gateway_master(data_dir: pathlib.Path | str) -> pd.DataFrame:
    """Load gateway master with latin-1 encoding and normalize IDs."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / "gateway_master.csv"
    df = pd.read_csv(path, encoding="latin-1")
    _validate_columns(df, _GATEWAY_MASTER_REQUIRED, "gateway_master")

    df["gateway_id"] = normalize_id_column(df["gateway_id"])

    # Parse dates
    for col in ["installed_on", "decommissioned_on", "fw_updated_on"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Loaded gateway_master: %d gateways (%d decommissioned)",
                len(df), df["decommissioned_on"].notna().sum())
    return df


def load_field_visits(data_dir: pathlib.Path | str) -> pd.DataFrame:
    """Load field visits with latin-1 encoding and normalize IDs."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / "field_visits.csv"
    df = pd.read_csv(path, encoding="latin-1")
    _validate_columns(df, _FIELD_VISITS_REQUIRED, "field_visits")

    df["gateway_id"] = normalize_id_column(df["gateway_id"])
    df["requested_on"] = pd.to_datetime(df["requested_on"])
    if "visited_on" in df.columns:
        df["visited_on"] = pd.to_datetime(df["visited_on"])

    logger.info("Loaded field_visits: %d visits, %d unique gateways, %s to %s",
                len(df), df["gateway_id"].nunique(),
                df["requested_on"].min().date(), df["requested_on"].max().date())
    return df


def load_meter_reads(data_dir: pathlib.Path | str) -> pd.DataFrame:
    """Load meter read success data and normalize IDs."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / "meter_read_success.csv"
    df = pd.read_csv(path)
    _validate_columns(df, _METER_READS_REQUIRED, "meter_read_success")

    df["gateway_id"] = normalize_id_column(df["gateway_id"])
    df["week_start"] = pd.to_datetime(df["week_start"])

    # Compute read rate
    df["read_rate"] = df["meters_read"] / df["meters_expected"].replace(0, float("nan"))

    logger.info("Loaded meter_reads: %d rows, %d gateways, %s to %s",
                len(df), df["gateway_id"].nunique(),
                df["week_start"].min().date(), df["week_start"].max().date())
    return df


def load_engineer_review(data_dir: pathlib.Path | str) -> pd.DataFrame | None:
    """Load engineer review Excel file if it exists. Returns None if not found."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / "engineer_review_2026-02.xlsx"
    if not path.exists():
        logger.warning("Engineer review file not found at %s — skipping", path)
        return None

    df = pd.read_excel(path)
    _validate_columns(df, _ENGINEER_REVIEW_REQUIRED, "engineer_review")

    df["gateway_id"] = normalize_id_column(df["gateway_id"])

    # Binary label: Schlecht (bad) = 1, Normal = 0
    df["is_bad"] = (df["Kategorie"] == "Schlecht").astype(int)

    logger.info("Loaded engineer_review: %d gateways (%d Schlecht, %d Normal)",
                len(df),
                df["is_bad"].sum(),
                (1 - df["is_bad"]).sum())
    return df


def get_active_gateways(
    gateway_master: pd.DataFrame,
    as_of: dt.date | None = None,
) -> set[str]:
    """Return set of gateway IDs that are NOT decommissioned as of a given date.

    Args:
        gateway_master: The gateway master DataFrame.
        as_of: Date to check against. If None, returns gateways not yet decommissioned.
    """
    if as_of is None:
        mask = gateway_master["decommissioned_on"].isna()
    else:
        as_of_ts = pd.Timestamp(as_of)
        mask = (
            gateway_master["decommissioned_on"].isna()
            | (gateway_master["decommissioned_on"] > as_of_ts)
        )
    return set(gateway_master.loc[mask, "gateway_id"])


def load_all(data_dir: str | pathlib.Path | None = None) -> DataBundle:
    """Load all datasets from the data directory.

    Args:
        data_dir: Path to data directory. Defaults to config value / DATA_DIR env var.

    Returns:
        DataBundle with all datasets loaded, validated, and normalized.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    telemetry = load_telemetry(data_dir)
    gateway_master = load_gateway_master(data_dir)
    field_visits = load_field_visits(data_dir)
    meter_reads = load_meter_reads(data_dir)
    engineer_review = load_engineer_review(data_dir)

    active_ids = get_active_gateways(gateway_master)

    logger.info("All datasets loaded. %d active gateways (of %d total)",
                len(active_ids), len(gateway_master))

    return DataBundle(
        telemetry=telemetry,
        gateway_master=gateway_master,
        field_visits=field_visits,
        meter_reads=meter_reads,
        engineer_review=engineer_review,
        active_gateway_ids=active_ids,
    )
