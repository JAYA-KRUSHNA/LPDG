"""Gateway ID normalization utilities.

Gateway IDs appear in two formats across datasets:
  - Bare hex:        0639EA5602C1        (telemetry, meter_read_success)
  - Colon-separated: 06:39:EA:56:02:C1  (gateway_master, field_visits, engineer_review)

All internal processing uses bare uppercase hex (12 characters).
"""

from __future__ import annotations

import re

_BARE_HEX = re.compile(r"^[0-9A-Fa-f]{12}$")
_COLON_SEP = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def normalize_gateway_id(gateway_id: str) -> str:
    """Normalize a gateway ID to bare uppercase hex (12 chars).

    Accepts both bare hex and colon-separated formats.
    Raises ValueError for unrecognized formats.
    """
    text = str(gateway_id).strip()
    if _BARE_HEX.match(text):
        return text.upper()
    if _COLON_SEP.match(text):
        return text.replace(":", "").upper()
    raise ValueError(
        f"Unrecognized gateway ID format: {text!r}. "
        "Expected 12 hex chars (e.g. 0639EA5602C1) or "
        "colon-separated (e.g. 06:39:EA:56:02:C1)."
    )


def normalize_id_column(series):
    """Normalize a pandas Series of gateway IDs in place.

    Returns the normalized series. Raises ValueError on the first bad ID.
    """
    return series.apply(normalize_gateway_id)
