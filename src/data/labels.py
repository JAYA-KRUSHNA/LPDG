"""Label construction from field visit outcomes.

Builds binary labels for the ML model:
  - Positive (1): gateway needed a visit (fault was found and fixed)
  - Negative (0): gateway was either fine, or was visited and nothing was wrong

Only uses data BEFORE the prediction date to avoid temporal leakage.
The engineer review (Feb 2026) is reserved for VALIDATION ONLY.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Outcome meanings (German):
#   "Fehler behoben"       = Fault fixed (positive label)
#   "Kein Fehler gefunden" = No fault found (negative label)
#   "Kein Zugang"          = No access (ambiguous - exclude)
POSITIVE_OUTCOMES = {"Fehler behoben"}
NEGATIVE_OUTCOMES = {"Kein Fehler gefunden"}
EXCLUDE_OUTCOMES = {"Kein Zugang"}


def build_training_labels(
    field_visits: pd.DataFrame,
    all_gateway_ids: set[str],
    mondays: list[dt.date],
    lookahead_days: int = 14,
) -> pd.DataFrame:
    """Build per-gateway-per-week binary labels from field visit outcomes.

    For each Monday in `mondays`, a gateway gets label=1 if a field visit
    with outcome "Fehler behoben" was requested within `lookahead_days`
    after that Monday. This acts as a proxy: "this gateway genuinely
    needed attention around this time."

    Gateways with no visit data are labeled 0 (assumed healthy).

    Args:
        field_visits: Field visits DataFrame with normalized gateway_id.
        all_gateway_ids: Set of all gateway IDs to generate labels for.
        mondays: List of Mondays to generate labels for.
        lookahead_days: How many days after Monday to look for visits.

    Returns:
        DataFrame with columns [week_start, gateway_id, label].
    """
    rows = []

    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        window_end = monday_ts + dt.timedelta(days=lookahead_days)

        # Get visits in the lookahead window
        window_visits = field_visits[
            (field_visits["requested_on"] >= monday_ts)
            & (field_visits["requested_on"] < window_end)
        ]

        # Positive: gateways with a fault-fix visit
        positive_ids = set(
            window_visits[
                window_visits["outcome"].isin(POSITIVE_OUTCOMES)
            ]["gateway_id"]
        )

        for gw_id in all_gateway_ids:
            rows.append({
                "week_start": monday.isoformat(),
                "gateway_id": gw_id,
                "label": 1 if gw_id in positive_ids else 0,
            })

    labels_df = pd.DataFrame(rows)

    n_positive = labels_df["label"].sum()
    n_total = len(labels_df)
    logger.info(
        "Built labels: %d total (%d positive = %.1f%%, %d negative) across %d weeks",
        n_total, n_positive, 100 * n_positive / max(n_total, 1),
        n_total - n_positive, len(mondays),
    )

    return labels_df


def build_repeat_offender_labels(
    field_visits: pd.DataFrame,
    cutoff: dt.date,
    min_visits: int = 2,
    window_months: int = 3,
) -> set[str]:
    """Identify gateways with repeated visits (likely chronic issues).

    Args:
        field_visits: Field visits DataFrame.
        cutoff: Only consider visits before this date.
        min_visits: Minimum visits to be considered a repeat offender.
        window_months: Look-back window in months.

    Returns:
        Set of gateway IDs that are repeat offenders.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    window_start = cutoff_ts - pd.DateOffset(months=window_months)

    recent_visits = field_visits[
        (field_visits["requested_on"] >= window_start)
        & (field_visits["requested_on"] < cutoff_ts)
    ]

    visit_counts = recent_visits.groupby("gateway_id").size()
    repeat_offenders = set(visit_counts[visit_counts >= min_visits].index)

    logger.info("Found %d repeat offenders (>=%d visits in %d months before %s)",
                len(repeat_offenders), min_visits, window_months, cutoff)

    return repeat_offenders


def get_training_weeks(
    field_visits: pd.DataFrame,
    scored_first_monday: dt.date = dt.date(2026, 2, 2),
) -> list[dt.date]:
    """Generate training Mondays from field visit data range.

    Uses the field visit date range to determine which Mondays
    have enough data for label construction. Stops before the scored window
    to prevent temporal leakage.

    Returns:
        List of Monday dates suitable for training.
    """
    # Field visits range: Feb 2025 - Jan 2026
    # We need at least 28 days of telemetry before each Monday
    # and lookahead days of field visit data after each Monday
    earliest_visit = field_visits["requested_on"].min()
    latest_visit = field_visits["requested_on"].max()

    # Start training from the first Monday at least 28 days after telemetry starts
    # (Aug 2025 + 28 days = ~Sep 2025)
    first_monday = dt.date(2025, 9, 1)
    # Adjust to actual Monday
    while first_monday.weekday() != 0:
        first_monday += dt.timedelta(days=1)

    # Last training Monday: at least 14 days before scored window starts
    # So labels have enough lookahead
    last_monday = scored_first_monday - dt.timedelta(days=21)

    mondays = []
    current = first_monday
    while current <= last_monday:
        mondays.append(current)
        current += dt.timedelta(days=7)

    logger.info("Training weeks: %d Mondays from %s to %s",
                len(mondays), mondays[0] if mondays else "none",
                mondays[-1] if mondays else "none")
    return mondays
