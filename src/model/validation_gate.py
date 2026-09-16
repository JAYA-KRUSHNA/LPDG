"""Pre-prediction validation gate — safety checks before serving predictions.

Runs automated checks to ensure the pipeline is safe to execute:
  1. Model integrity: model.lgb, metadata.json, features.json all present
  2. Feature schema: saved feature list matches what the model expects
  3. Data drift: runs drift detection, blocks on CRITICAL severity

Requirement F3+: "Not just detecting drift, but acting on it."
"""

from __future__ import annotations

import json
import pathlib

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def check_model_integrity(model_dir: pathlib.Path) -> list[dict]:
    """Validate that the current model has all required files.

    Returns:
        List of check results with name, status, detail, severity.
    """
    model_dir = pathlib.Path(model_dir)
    checks = []

    # Check 'current' symlink/pointer exists
    current_path = model_dir / "current"
    if not current_path.exists():
        checks.append({
            "name": "model_current",
            "status": "FAIL",
            "detail": "No 'current' model pointer found. Run 'make train' first.",
            "severity": "CRITICAL",
        })
        return checks  # Can't check further without a model

    checks.append({
        "name": "model_current",
        "status": "PASS",
        "detail": f"Current model pointer exists",
        "severity": "OK",
    })

    # Resolve the actual version directory
    if current_path.is_symlink():
        version_dir = current_path.resolve()
    else:
        version_dir = current_path

    # Check required files
    required_files = {
        "model.lgb": "Serialized model",
        "metadata.json": "Training metadata",
        "features.json": "Feature schema",
    }

    for filename, description in required_files.items():
        filepath = version_dir / filename
        if filepath.exists():
            checks.append({
                "name": f"model_file_{filename}",
                "status": "PASS",
                "detail": f"{description} present",
                "severity": "OK",
            })
        else:
            checks.append({
                "name": f"model_file_{filename}",
                "status": "FAIL",
                "detail": f"Missing {description}: {filepath}",
                "severity": "CRITICAL",
            })

    # Validate feature count matches metadata
    meta_path = version_dir / "metadata.json"
    feat_path = version_dir / "features.json"
    if meta_path.exists() and feat_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        with open(feat_path) as f:
            feat = json.load(f)

        meta_n = meta.get("n_features", 0)
        feat_n = feat.get("n_features", 0)
        feat_names = feat.get("feature_names", [])

        if meta_n == feat_n == len(feat_names):
            checks.append({
                "name": "feature_consistency",
                "status": "PASS",
                "detail": f"Feature count consistent: {feat_n} features",
                "severity": "OK",
            })
        else:
            checks.append({
                "name": "feature_consistency",
                "status": "FAIL",
                "detail": (f"Feature count mismatch: metadata={meta_n}, "
                           f"features.json={feat_n}, names={len(feat_names)}"),
                "severity": "WARNING",
            })

    return checks


def run_pre_prediction_checks(
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    config: dict,
) -> tuple[bool, str]:
    """Run all pre-prediction validation checks.

    Args:
        data_dir: Path to data directory.
        model_dir: Path to model directory.
        config: Application config dict.

    Returns:
        Tuple of (passed: bool, report: str).
        passed=False means predictions should NOT be generated.
    """
    all_checks = []
    severity_order = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
    max_severity = "OK"

    # 1. Model integrity checks
    integrity_checks = check_model_integrity(model_dir)
    all_checks.extend(integrity_checks)

    # 2. Data drift checks (only if model exists and data is available)
    has_critical_model = any(
        c["severity"] == "CRITICAL" for c in integrity_checks
    )

    if not has_critical_model:
        try:
            from src.data.drift import run_drift_detection
            drift_report = run_drift_detection(data_dir, model_dir, config)

            for check in drift_report.checks:
                all_checks.append(check)

        except Exception as e:
            # Drift detection failure should warn but not block
            all_checks.append({
                "name": "drift_detection",
                "status": "SKIP",
                "detail": f"Drift detection skipped: {e}",
                "severity": "WARNING",
            })

    # Determine overall severity
    for check in all_checks:
        sev = check.get("severity", "OK")
        if severity_order.get(sev, 0) > severity_order.get(max_severity, 0):
            max_severity = sev

    # Build report
    report_lines = [f"Pre-Prediction Validation — Overall: {max_severity}", ""]

    for check in all_checks:
        marker = "✓" if check["status"] in ("PASS", "OK") else "✗"
        if check["status"] == "SKIP":
            marker = "○"
        report_lines.append(
            f"  {marker} [{check['severity']}] {check['name']}: {check['detail']}"
        )

    report = "\n".join(report_lines)

    # CRITICAL = block, everything else = proceed
    passed = max_severity != "CRITICAL"

    if passed:
        logger.info("Pre-prediction validation PASSED (severity: %s)",
                     max_severity)
    else:
        logger.error("Pre-prediction validation FAILED (severity: CRITICAL)")

    return passed, report
