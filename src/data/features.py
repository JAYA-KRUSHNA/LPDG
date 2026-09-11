"""Feature engineering pipeline.

Builds per-gateway-per-week feature vectors from all data sources.
The baseline uses only 3 metrics with a 3-sigma rule.  We engineer ~55 features
from telemetry, meter reads, gateway master, and field visit history.

Each feature function takes raw data + a reference Monday and returns
a DataFrame indexed by gateway_id.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_DAYS = 28
RECENT_DAYS = 7

# Core metrics the baseline tracks
CORE_METRICS = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt"]

# Extended telemetry columns we use
REBOOT_COLS = [
    "reboot_cnt", "reboot_duration_sec", "r_cnt_power_cycle",
    "r_cnt_reboot", "r_cnt_unknown", "reboot_importance",
]
CONNECTIVITY_COLS = [
    "offline_duration_sec", "disconnection_cnt",
    "online_duration_mins", "no_conn_importance",
]
SYSTEM_COLS = [
    "avg_load1", "load1_bigger1", "load1_bigger2",
    "avg_memfree", "avg_uptime",
]
LORA_COLS = [
    "rx_nr_pkts", "rx_crc_bad", "tx_success", "tx_busy",
]
SIGNAL_COLS = [
    "rssi_good", "rssi_normal", "rssi_bad",
    "rscp_rsrp_good", "rscp_rsrp_normal", "rscp_rsrp_bad",
    "ecio_rsrq_good", "ecio_rsrq_normal", "ecio_rsrq_bad",
]
NETWORK_COLS = ["network_2g", "network_3g", "network_4g", "network_unknown"]

ALL_TELEMETRY_FEATURE_COLS = (
    CORE_METRICS + REBOOT_COLS + CONNECTIVITY_COLS +
    SYSTEM_COLS + LORA_COLS + SIGNAL_COLS + NETWORK_COLS
)
# Deduplicate while preserving order
ALL_TELEMETRY_FEATURE_COLS = list(dict.fromkeys(ALL_TELEMETRY_FEATURE_COLS))


# ---------------------------------------------------------------------------
# Telemetry features
# ---------------------------------------------------------------------------

def _safe_div(a, b, fill=0.0):
    """Divide a by b, filling NaN/inf with fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return np.where(np.isfinite(result), result, fill)


def compute_telemetry_features(
    telemetry: pd.DataFrame,
    monday: dt.date,
    baseline_days: int = BASELINE_DAYS,
    recent_days: int = RECENT_DAYS,
) -> pd.DataFrame:
    """Compute per-gateway telemetry features for a given Monday.

    Uses a trailing window of `baseline_days` before `monday`.
    Computes trend as ratio of recent `recent_days` to prior period.
    """
    end = pd.Timestamp(monday, tz="UTC")
    start = end - dt.timedelta(days=baseline_days)
    recent_start = end - dt.timedelta(days=recent_days)

    window = telemetry[(telemetry["ts"] >= start) & (telemetry["ts"] < end)].copy()
    if window.empty:
        logger.warning("No telemetry data in window ending %s", monday)
        return pd.DataFrame()

    recent = window[window["ts"] >= recent_start]
    prior = window[window["ts"] < recent_start]

    features = {}

    # --- Connectivity features ---
    for col in ["offline_duration_sec", "disconnection_cnt"]:
        grp = window.groupby("gateway_id")[col]
        features[f"{col}_mean"] = grp.mean()
        features[f"{col}_max"] = grp.max()
        features[f"{col}_std"] = grp.std().fillna(0)
        features[f"{col}_p95"] = grp.quantile(0.95)
        features[f"{col}_sum"] = grp.sum()

        # Trend: recent mean / prior mean (>1 means getting worse)
        recent_mean = recent.groupby("gateway_id")[col].mean()
        prior_mean = prior.groupby("gateway_id")[col].mean()
        # Align indices: use pandas division for automatic index alignment
        aligned_prior = prior_mean.reindex(recent_mean.index, fill_value=0).replace(0, np.nan)
        features[f"{col}_trend"] = (recent_mean / aligned_prior).fillna(1.0)

    # Online duration
    grp = window.groupby("gateway_id")["online_duration_mins"]
    features["online_duration_mean"] = grp.mean()
    features["online_duration_min"] = grp.min()

    # Connection importance score
    if "no_conn_importance" in window.columns:
        grp = window.groupby("gateway_id")["no_conn_importance"]
        features["no_conn_importance_mean"] = grp.mean()
        features["no_conn_importance_max"] = grp.max()

    # --- Reboot features ---
    grp = window.groupby("gateway_id")["reboot_cnt"]
    features["reboot_cnt_sum"] = grp.sum()
    features["reboot_cnt_max"] = grp.max()

    recent_reboot = recent.groupby("gateway_id")["reboot_cnt"].sum()
    prior_reboot = prior.groupby("gateway_id")["reboot_cnt"].sum()
    aligned_prior_reboot = prior_reboot.reindex(recent_reboot.index, fill_value=0).replace(0, np.nan)
    features["reboot_cnt_trend"] = (recent_reboot / aligned_prior_reboot).fillna(1.0)

    grp = window.groupby("gateway_id")["reboot_duration_sec"]
    features["reboot_duration_sum"] = grp.sum()
    features["reboot_duration_max"] = grp.max()

    if "reboot_importance" in window.columns:
        grp = window.groupby("gateway_id")["reboot_importance"]
        features["reboot_importance_mean"] = grp.mean()
        features["reboot_importance_max"] = grp.max()

    # Power cycle ratio (hardware vs software reboots)
    total_reboots = window.groupby("gateway_id")["reboot_cnt"].sum()
    power_cycles = window.groupby("gateway_id")["r_cnt_power_cycle"].sum()
    features["power_cycle_ratio"] = (power_cycles / total_reboots.replace(0, np.nan)).fillna(0.0)

    # --- System health features ---
    for col in ["avg_load1"]:
        grp = window.groupby("gateway_id")[col]
        features[f"{col}_mean"] = grp.mean()
        features[f"{col}_max"] = grp.max()

    if "load1_bigger2" in window.columns:
        features["load1_bigger2_freq"] = (
            window.groupby("gateway_id")["load1_bigger2"].sum()
            / window.groupby("gateway_id")["load1_bigger2"].count()
        )

    if "avg_memfree" in window.columns:
        grp = window.groupby("gateway_id")["avg_memfree"]
        features["memfree_min"] = grp.min()
        features["memfree_mean"] = grp.mean()
        # Trend: declining memory is concerning
        recent_mem = recent.groupby("gateway_id")["avg_memfree"].mean()
        prior_mem = prior.groupby("gateway_id")["avg_memfree"].mean()
        aligned_prior_mem = prior_mem.reindex(recent_mem.index, fill_value=0).replace(0, np.nan)
        features["memfree_trend"] = (recent_mem / aligned_prior_mem).fillna(1.0)

    if "avg_uptime" in window.columns:
        grp = window.groupby("gateway_id")["avg_uptime"]
        features["uptime_min"] = grp.min()  # Low uptime = frequent restarts
        features["uptime_mean"] = grp.mean()

    # --- LoRa performance features ---
    if "rx_nr_pkts" in window.columns:
        grp = window.groupby("gateway_id")["rx_nr_pkts"]
        features["rx_pkts_mean"] = grp.mean()

        recent_rx = recent.groupby("gateway_id")["rx_nr_pkts"].mean()
        prior_rx = prior.groupby("gateway_id")["rx_nr_pkts"].mean()
        aligned_prior_rx = prior_rx.reindex(recent_rx.index, fill_value=0).replace(0, np.nan)
        features["rx_pkts_trend"] = (recent_rx / aligned_prior_rx).fillna(1.0)

    if "rx_crc_bad" in window.columns and "rx_nr_pkts" in window.columns:
        crc_bad = window.groupby("gateway_id")["rx_crc_bad"].sum()
        rx_total = window.groupby("gateway_id")["rx_nr_pkts"].sum()
        features["crc_error_rate"] = (crc_bad / rx_total.replace(0, np.nan)).fillna(0.0)

    if "tx_success" in window.columns:
        features["tx_success_mean"] = window.groupby("gateway_id")["tx_success"].mean()

    if "tx_busy" in window.columns:
        features["tx_busy_sum"] = window.groupby("gateway_id")["tx_busy"].sum()

    # --- Signal quality features ---
    for prefix, cols in [
        ("rssi", ["rssi_good", "rssi_normal", "rssi_bad"]),
        ("rscp", ["rscp_rsrp_good", "rscp_rsrp_normal", "rscp_rsrp_bad"]),
        ("ecio", ["ecio_rsrq_good", "ecio_rsrq_normal", "ecio_rsrq_bad"]),
    ]:
        if all(c in window.columns for c in cols):
            good = window.groupby("gateway_id")[cols[0]].sum()
            normal = window.groupby("gateway_id")[cols[1]].sum()
            bad = window.groupby("gateway_id")[cols[2]].sum()
            total = good + normal + bad
            features[f"{prefix}_bad_ratio"] = pd.Series(
                _safe_div(bad, total, fill=0.0), index=bad.index
            )

    # --- Network type features ---
    if all(c in window.columns for c in NETWORK_COLS):
        for col in NETWORK_COLS:
            grp_sum = window.groupby("gateway_id")[col].sum()
            features[f"{col}_frac"] = grp_sum  # Will normalize later

        # Compute fractions
        net_total = sum(features[f"{c}_frac"] for c in NETWORK_COLS)
        for col in NETWORK_COLS:
            features[f"{col}_frac"] = pd.Series(
                _safe_div(features[f"{col}_frac"], net_total, fill=0.0),
                index=features[f"{col}_frac"].index,
            )

    # --- Temporal pattern features ---
    # Weekend vs weekday issue ratio
    window_copy = window.copy()
    window_copy["is_weekend"] = window_copy["ts"].dt.dayofweek >= 5
    for col in ["offline_duration_sec", "disconnection_cnt"]:
        weekend_sum = (
            window_copy[window_copy["is_weekend"]]
            .groupby("gateway_id")[col].sum()
        )
        weekday_sum = (
            window_copy[~window_copy["is_weekend"]]
            .groupby("gateway_id")[col].sum()
        )
        features[f"{col}_weekend_ratio"] = (
            weekend_sum / weekday_sum.reindex(weekend_sum.index, fill_value=0).replace(0, np.nan)
        ).fillna(1.0)

    # Hours with any anomaly (like baseline's flag count)
    stats = window.groupby("gateway_id")[CORE_METRICS].agg(["mean", "std"])
    for metric in CORE_METRICS:
        mean_s = recent["gateway_id"].map(stats[(metric, "mean")])
        std_s = recent["gateway_id"].map(stats[(metric, "std")]).replace(0, np.nan)
        exceeded = ((recent[metric] - mean_s) > 3.0 * std_s).fillna(False)
        features[f"{metric}_3sigma_hours"] = (
            exceeded.astype(int).groupby(recent["gateway_id"]).sum()
        )

    # Combine all features into a DataFrame
    result = pd.DataFrame(features)
    result.index.name = "gateway_id"

    # Fill NaN with 0 (gateways with no recent data get zero features)
    result = result.fillna(0)

    logger.debug("Computed %d telemetry features for %d gateways (week %s)",
                 len(result.columns), len(result), monday)
    return result


# ---------------------------------------------------------------------------
# Meter read features
# ---------------------------------------------------------------------------

def compute_meter_features(
    meter_reads: pd.DataFrame,
    monday: dt.date,
) -> pd.DataFrame:
    """Compute per-gateway meter read features as of a given Monday.

    Uses the last available meter read data before the Monday.
    Data may be stale (meter reads end before the scored window).
    """
    cutoff = pd.Timestamp(monday)
    available = meter_reads[meter_reads["week_start"] < cutoff].copy()

    if available.empty:
        logger.warning("No meter read data available before %s", monday)
        return pd.DataFrame()

    features = {}

    # Last known read rate
    latest = available.sort_values("week_start").groupby("gateway_id").last()
    features["last_read_rate"] = latest["read_rate"]
    features["meters_expected"] = latest["meters_expected"]
    features["meters_at_risk"] = (
        latest["meters_expected"] * (1 - latest["read_rate"].fillna(0))
    )

    # Staleness: days since last meter read data
    features["meter_data_staleness_days"] = (
        (cutoff - latest["week_start"]).dt.days
    )

    # 4-week trend (slope of read_rate over last 4 available weeks)
    last_4w = (
        available.sort_values("week_start")
        .groupby("gateway_id")
        .tail(4)
    )
    def _slope(group):
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group), dtype=float)
        y = group["read_rate"].values.astype(float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return 0.0
        coef = np.polyfit(x[mask], y[mask], 1)
        return float(coef[0])

    trends = last_4w.groupby("gateway_id").apply(_slope, include_groups=False)
    features["read_rate_trend_4w"] = trends

    result = pd.DataFrame(features)
    result.index.name = "gateway_id"
    result = result.fillna(0)

    logger.debug("Computed %d meter features for %d gateways (week %s)",
                 len(result.columns), len(result), monday)
    return result


# ---------------------------------------------------------------------------
# Static gateway features
# ---------------------------------------------------------------------------

def compute_gateway_features(
    gateway_master: pd.DataFrame,
    monday: dt.date,
) -> pd.DataFrame:
    """Compute per-gateway static features.

    Encodes categorical variables and computes age-based features.
    """
    gw = gateway_master.set_index("gateway_id").copy()
    features = {}

    ref_date = pd.Timestamp(monday)

    # Gateway age
    if "installed_on" in gw.columns:
        features["gateway_age_days"] = (ref_date - gw["installed_on"]).dt.days

    # Firmware age
    if "fw_updated_on" in gw.columns:
        features["firmware_age_days"] = (ref_date - gw["fw_updated_on"]).dt.days
        # Gateways never updated: fill with gateway age
        if "gateway_age_days" in features:
            age_series = pd.Series(features["gateway_age_days"])
            fw_age = pd.Series(features["firmware_age_days"])
            features["firmware_age_days"] = fw_age.fillna(age_series)

    # Number of meters
    features["n_meters_installed"] = gw["n_meters_installed"]

    # Categorical features — label encode
    for col in ["hw_model", "site_type", "region", "antenna_type", "tenant"]:
        if col in gw.columns:
            codes, uniques = pd.factorize(gw[col])
            features[f"{col}_code"] = pd.Series(codes, index=gw.index)

    result = pd.DataFrame(features)
    result.index.name = "gateway_id"
    result = result.fillna(0)

    logger.debug("Computed %d gateway features for %d gateways",
                 len(result.columns), len(result))
    return result


# ---------------------------------------------------------------------------
# Field visit history features
# ---------------------------------------------------------------------------

def compute_visit_features(
    field_visits: pd.DataFrame,
    monday: dt.date,
) -> pd.DataFrame:
    """Compute per-gateway features from field visit history.

    Only uses visits strictly before the prediction Monday to avoid leakage.
    """
    cutoff = pd.Timestamp(monday)
    past = field_visits[field_visits["requested_on"] < cutoff].copy()

    if past.empty:
        return pd.DataFrame()

    features = {}

    grp = past.groupby("gateway_id")

    features["n_past_visits"] = grp.size()

    # Count faults found
    past["is_fault"] = (past["outcome"] == "Fehler behoben").astype(int)
    features["n_faults_found"] = past.groupby("gateway_id")["is_fault"].sum()

    # Fault rate
    features["fault_rate"] = (
        pd.Series(features["n_faults_found"])
        / pd.Series(features["n_past_visits"]).replace(0, np.nan)
    ).fillna(0)

    # Days since last visit
    last_visit = grp["requested_on"].max()
    features["days_since_last_visit"] = (cutoff - last_visit).dt.days

    # Last outcome was a fault?
    last_rows = past.sort_values("requested_on").groupby("gateway_id").last()
    features["last_outcome_was_fault"] = (
        (last_rows["outcome"] == "Fehler behoben").astype(int)
    )

    # Days since last confirmed fault
    faults_only = past[past["is_fault"] == 1]
    if not faults_only.empty:
        last_fault = faults_only.groupby("gateway_id")["requested_on"].max()
        features["days_since_last_fault"] = (cutoff - last_fault).dt.days
    else:
        features["days_since_last_fault"] = pd.Series(dtype=float)

    result = pd.DataFrame(features)
    result.index.name = "gateway_id"
    result = result.fillna(0)

    logger.debug("Computed %d visit features for %d gateways (week %s)",
                 len(result.columns), len(result), monday)
    return result


# ---------------------------------------------------------------------------
# Combined feature builder
# ---------------------------------------------------------------------------

def build_features_for_week(
    telemetry: pd.DataFrame,
    gateway_master: pd.DataFrame,
    field_visits: pd.DataFrame,
    meter_reads: pd.DataFrame,
    monday: dt.date,
    active_gateway_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for a single week.

    Args:
        telemetry: Full telemetry DataFrame (will be filtered internally).
        gateway_master: Gateway master DataFrame.
        field_visits: Field visits DataFrame.
        meter_reads: Meter reads DataFrame.
        monday: The Monday of the prediction week.
        active_gateway_ids: Set of non-decommissioned gateway IDs to include.

    Returns:
        DataFrame with gateway_id as index and all features as columns.
    """
    tel_feat = compute_telemetry_features(telemetry, monday)
    meter_feat = compute_meter_features(meter_reads, monday)
    gw_feat = compute_gateway_features(gateway_master, monday)
    visit_feat = compute_visit_features(field_visits, monday)

    # Join all feature groups on gateway_id
    # Start with gateway features (covers all gateways)
    combined = gw_feat.copy()

    for feat_df in [tel_feat, meter_feat, visit_feat]:
        if not feat_df.empty:
            combined = combined.join(feat_df, how="left")

    # Fill missing values (gateways with no telemetry/visits get 0)
    combined = combined.fillna(0)

    # Filter to active gateways only
    if active_gateway_ids is not None:
        combined = combined[combined.index.isin(active_gateway_ids)]

    logger.info("Built %d features for %d gateways (week %s)",
                len(combined.columns), len(combined), monday)
    return combined


def build_features_all_weeks(
    telemetry: pd.DataFrame,
    gateway_master: pd.DataFrame,
    field_visits: pd.DataFrame,
    meter_reads: pd.DataFrame,
    mondays: list[dt.date],
    active_gateway_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build features for multiple weeks, returning a multi-indexed DataFrame.

    Returns:
        DataFrame with (monday, gateway_id) as multi-index.
    """
    frames = []
    for monday in mondays:
        week_feat = build_features_for_week(
            telemetry, gateway_master, field_visits, meter_reads,
            monday, active_gateway_ids,
        )
        week_feat["week_start"] = monday.isoformat()
        frames.append(week_feat)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames)
    result = result.reset_index().set_index(["week_start", "gateway_id"])

    logger.info("Built features for %d weeks, %d total rows",
                len(mondays), len(result))
    return result
