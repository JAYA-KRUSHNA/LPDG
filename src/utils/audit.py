"""Pipeline audit trail — logs every pipeline execution for traceability.

Provides a complete record of when each pipeline step ran, what model
was used, and what the outcome was.

This enables:
  - Tracing any prediction back to its training run
  - Detecting if the pipeline was run with stale data
  - Compliance and audit requirements

Usage:
    python -m src.utils.audit          # Show recent pipeline events
    python -m src.utils.audit --all    # Show all events
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _get_log_dir() -> pathlib.Path:
    """Get the logs directory, creating it if needed."""
    # Walk up from this file to find project root
    current = pathlib.Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "default.yaml").exists():
            log_dir = current / "logs"
            log_dir.mkdir(exist_ok=True)
            return log_dir
        current = current.parent
    # Fallback
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def log_pipeline_event(
    action: str,
    details: dict,
    log_dir: pathlib.Path | None = None,
) -> None:
    """Append a pipeline event to the audit trail.

    Args:
        action: Pipeline step name (e.g., "train", "predict", "drift").
        details: Dict of event-specific details.
        log_dir: Override log directory (default: project_root/logs/).
    """
    if log_dir is None:
        log_dir = _get_log_dir()

    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_audit.jsonl"

    event = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "action": action,
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        **details,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

    logger.debug("Audit event logged: %s", action)


def get_recent_events(
    log_dir: pathlib.Path | None = None,
    n: int = 20,
) -> list[dict]:
    """Read the last N pipeline events from the audit trail.

    Returns:
        List of event dicts, most recent last.
    """
    if log_dir is None:
        log_dir = _get_log_dir()

    log_path = pathlib.Path(log_dir) / "pipeline_audit.jsonl"

    if not log_path.exists():
        return []

    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return events[-n:]


def format_audit_log(events: list[dict]) -> str:
    """Format audit events as a readable log."""
    if not events:
        return "No pipeline events recorded yet. Run 'make all' to create one."

    lines = []
    lines.append("=" * 85)
    lines.append("PIPELINE AUDIT TRAIL")
    lines.append("=" * 85)
    lines.append(
        f"{'Timestamp':<22} {'Action':<10} {'Model':>8} {'Details':<40}"
    )
    lines.append("-" * 85)

    for event in events:
        ts = event.get("timestamp", "")[:19].replace("T", " ")
        action = event.get("action", "?")
        version = event.get("model_version", event.get("version", "—"))
        status = event.get("status", "")

        # Build a short detail string
        detail_parts = []
        if status:
            detail_parts.append(status)
        if "n_rows" in event:
            detail_parts.append(f"{event['n_rows']} rows")
        if "n_samples" in event:
            detail_parts.append(f"{event['n_samples']} samples")
        if "elapsed_seconds" in event:
            detail_parts.append(f"{event['elapsed_seconds']:.1f}s")
        if "drift_severity" in event:
            detail_parts.append(f"drift={event['drift_severity']}")
        if "data_hash" in event:
            detail_parts.append(f"hash={event['data_hash'][:8]}")

        detail_str = ", ".join(detail_parts) if detail_parts else "—"

        lines.append(
            f"{ts:<22} {action:<10} {str(version):>8} {detail_str:<40}"
        )

    lines.append("=" * 85)
    lines.append(f"Showing {len(events)} events")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for viewing the audit trail."""
    parser = argparse.ArgumentParser(description="View pipeline audit trail")
    parser.add_argument("--all", action="store_true", help="Show all events")
    parser.add_argument("--n", type=int, default=20, help="Number of events")
    parser.add_argument("--log-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    n = 9999 if args.all else args.n
    events = get_recent_events(args.log_dir, n=n)
    print(format_audit_log(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
