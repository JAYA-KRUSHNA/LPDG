"""Tests for MLOps enhancements: experiment tracking, comparison, validation gate, audit."""

import json
import pathlib
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Experiment Tracker Tests
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    """Tests for src.model.experiment_tracker."""

    def _make_metadata(self):
        """Create mock training metadata matching train.py's output format."""
        return {
            "version": "v1.0.0",
            "trained_at": "2026-09-15T10:00:00+00:00",
            "training_samples": 6400,
            "n_features": 68,
            "positive_rate": 0.02625,
            "config": {
                "objective": "binary",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "scale_pos_weight": 1.58,
                "min_child_samples": 10,
                "subsample": 0.8,
            },
            "cv_metrics": {
                "cv_auc_roc_mean": 0.8799,
                "cv_auc_pr_mean": 0.2926,
                "cv_logloss_mean": 0.0884,
                "cv_caught_mean": 6.33,
                "cv_missed_cost_mean": 29800.0,
                "cv_recall_at_15_mean": 0.1133,
                "final_num_trees": 34,
                "cv_folds": 3,
            },
            "data_hash": "55d4e2df9c6a18cf",
            "random_seed": 42,
        }

    def test_log_and_load(self, tmp_path):
        """Log an experiment and verify it can be loaded back."""
        from src.model.experiment_tracker import log_experiment, load_experiment_log

        metadata = self._make_metadata()
        log_experiment(tmp_path, "v1.0.0", metadata, elapsed_seconds=9.5)

        experiments = load_experiment_log(tmp_path)
        assert len(experiments) == 1

        exp = experiments[0]
        assert exp["version"] == "v1.0.0"
        assert exp["n_features"] == 68
        assert exp["metrics"]["auc_roc"] == 0.8799
        assert exp["training_seconds"] == 9.5

    def test_multiple_experiments(self, tmp_path):
        """Logging multiple experiments appends correctly (JSONL format)."""
        from src.model.experiment_tracker import log_experiment, load_experiment_log

        metadata = self._make_metadata()

        log_experiment(tmp_path, "v1.0.0", metadata, 9.5)
        log_experiment(tmp_path, "v1.1.0", metadata, 10.2)
        log_experiment(tmp_path, "v1.2.0", metadata, 8.7)

        experiments = load_experiment_log(tmp_path)
        assert len(experiments) == 3
        assert experiments[0]["version"] == "v1.0.0"
        assert experiments[2]["version"] == "v1.2.0"

    def test_load_empty(self, tmp_path):
        """Loading from a directory with no log returns empty list."""
        from src.model.experiment_tracker import load_experiment_log

        experiments = load_experiment_log(tmp_path)
        assert experiments == []

    def test_format_history(self, tmp_path):
        """Format experiment history produces readable output."""
        from src.model.experiment_tracker import (
            log_experiment, load_experiment_log, format_experiment_history,
        )

        metadata = self._make_metadata()
        log_experiment(tmp_path, "v1.0.0", metadata, 9.5)

        experiments = load_experiment_log(tmp_path)
        output = format_experiment_history(experiments)

        assert "EXPERIMENT HISTORY" in output
        assert "v1.0.0" in output
        assert "0.8799" in output

    def test_format_empty_history(self):
        """Formatting empty history returns helpful message."""
        from src.model.experiment_tracker import format_experiment_history

        output = format_experiment_history([])
        assert "No experiments recorded" in output

    def test_jsonl_format(self, tmp_path):
        """Each experiment is a single valid JSON line."""
        from src.model.experiment_tracker import log_experiment

        metadata = self._make_metadata()
        log_experiment(tmp_path, "v1.0.0", metadata, 9.5)
        log_experiment(tmp_path, "v1.1.0", metadata, 10.0)

        log_path = tmp_path / "experiment_log.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)  # Must not raise
            assert "version" in parsed
            assert "metrics" in parsed


# ---------------------------------------------------------------------------
# Model Comparison Tests
# ---------------------------------------------------------------------------

class TestModelComparison:
    """Tests for src.model.compare."""

    def _create_model_version(self, model_dir, version, auc_roc=0.85, num_trees=30):
        """Create a mock model version directory with metadata and features."""
        version_dir = model_dir / version
        version_dir.mkdir(parents=True)

        metadata = {
            "version": version,
            "trained_at": "2026-09-15T10:00:00+00:00",
            "training_samples": 6400,
            "n_features": 68,
            "positive_rate": 0.02625,
            "config": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "scale_pos_weight": 1.58,
            },
            "cv_metrics": {
                "cv_auc_roc_mean": auc_roc,
                "cv_auc_pr_mean": 0.29,
                "cv_logloss_mean": 0.088,
                "cv_caught_mean": 6.3,
                "cv_missed_cost_mean": 29800.0,
                "cv_recall_at_15_mean": 0.113,
                "final_num_trees": num_trees,
            },
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        features = {
            "feature_names": [f"feat_{i}" for i in range(68)],
            "feature_importance": {f"feat_{i}": 100 - i for i in range(68)},
            "n_features": 68,
        }
        with open(version_dir / "features.json", "w") as f:
            json.dump(features, f)

    def test_compare_two_versions(self, tmp_path):
        """Compare two model versions and verify deltas are computed."""
        from src.model.compare import compare_versions

        self._create_model_version(tmp_path, "v1.0.0", auc_roc=0.85, num_trees=30)
        self._create_model_version(tmp_path, "v1.1.0", auc_roc=0.88, num_trees=34)

        result = compare_versions(tmp_path, "v1.0.0", "v1.1.0")

        assert result["version_a"] == "v1.0.0"
        assert result["version_b"] == "v1.1.0"
        assert result["deltas"]["cv_auc_roc_mean"] == pytest.approx(0.03, abs=0.001)
        assert result["deltas"]["final_num_trees"] == 4

    def test_compare_missing_version(self, tmp_path):
        """Comparing with a non-existent version raises FileNotFoundError."""
        from src.model.compare import compare_versions

        self._create_model_version(tmp_path, "v1.0.0")

        with pytest.raises(FileNotFoundError):
            compare_versions(tmp_path, "v1.0.0", "v9.9.9")

    def test_format_comparison(self, tmp_path):
        """Formatting a comparison produces readable output."""
        from src.model.compare import compare_versions, format_comparison

        self._create_model_version(tmp_path, "v1.0.0", auc_roc=0.85)
        self._create_model_version(tmp_path, "v1.1.0", auc_roc=0.88)

        result = compare_versions(tmp_path, "v1.0.0", "v1.1.0")
        output = format_comparison(result)

        assert "MODEL COMPARISON" in output
        assert "v1.0.0" in output
        assert "v1.1.0" in output
        assert "AUC-ROC" in output


# ---------------------------------------------------------------------------
# Validation Gate Tests
# ---------------------------------------------------------------------------

class TestValidationGate:
    """Tests for src.model.validation_gate."""

    def _create_valid_model(self, model_dir):
        """Create a minimal valid model directory structure."""
        version_dir = model_dir / "v1.0.0"
        version_dir.mkdir(parents=True)

        # Create model.lgb (just a placeholder file)
        (version_dir / "model.lgb").write_text("placeholder")

        # Create metadata.json
        metadata = {"version": "v1.0.0", "n_features": 3}
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create features.json
        features = {
            "feature_names": ["feat_0", "feat_1", "feat_2"],
            "feature_importance": {"feat_0": 100, "feat_1": 50, "feat_2": 25},
            "n_features": 3,
        }
        with open(version_dir / "features.json", "w") as f:
            json.dump(features, f)

        # Create 'current' symlink
        current = model_dir / "current"
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to("v1.0.0")

    def test_integrity_pass(self, tmp_path):
        """Model integrity checks pass with a valid model directory."""
        from src.model.validation_gate import check_model_integrity

        self._create_valid_model(tmp_path)
        checks = check_model_integrity(tmp_path)

        statuses = [c["status"] for c in checks]
        assert all(s == "PASS" for s in statuses)

    def test_integrity_no_model(self, tmp_path):
        """Model integrity fails when no 'current' pointer exists."""
        from src.model.validation_gate import check_model_integrity

        checks = check_model_integrity(tmp_path)

        assert any(c["severity"] == "CRITICAL" for c in checks)
        assert any("current" in c["name"] for c in checks)

    def test_integrity_missing_file(self, tmp_path):
        """Model integrity catches a missing model.lgb file."""
        from src.model.validation_gate import check_model_integrity

        self._create_valid_model(tmp_path)

        # Remove model.lgb
        (tmp_path / "v1.0.0" / "model.lgb").unlink()

        checks = check_model_integrity(tmp_path)
        critical = [c for c in checks if c["severity"] == "CRITICAL"]
        assert len(critical) >= 1
        assert any("model.lgb" in c["detail"] for c in critical)

    def test_feature_consistency_check(self, tmp_path):
        """Feature count mismatch is detected."""
        from src.model.validation_gate import check_model_integrity

        self._create_valid_model(tmp_path)

        # Corrupt metadata to have wrong feature count
        meta_path = tmp_path / "v1.0.0" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["n_features"] = 999
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        checks = check_model_integrity(tmp_path)
        warnings = [c for c in checks if c["severity"] == "WARNING"]
        assert len(warnings) >= 1
        assert any("mismatch" in c["detail"].lower() for c in warnings)


# ---------------------------------------------------------------------------
# Audit Trail Tests
# ---------------------------------------------------------------------------

class TestAuditTrail:
    """Tests for src.utils.audit."""

    def test_log_and_read(self, tmp_path):
        """Log an event and verify it can be read back."""
        from src.utils.audit import log_pipeline_event, get_recent_events

        log_pipeline_event("train", {
            "version": "v1.0.0",
            "n_samples": 6400,
            "status": "success",
        }, log_dir=tmp_path)

        events = get_recent_events(tmp_path)
        assert len(events) == 1
        assert events[0]["action"] == "train"
        assert events[0]["version"] == "v1.0.0"

    def test_multiple_events(self, tmp_path):
        """Multiple events are logged in order."""
        from src.utils.audit import log_pipeline_event, get_recent_events

        log_pipeline_event("train", {"version": "v1.0.0"}, log_dir=tmp_path)
        log_pipeline_event("predict", {"model_version": "v1.0.0", "n_rows": 120}, log_dir=tmp_path)
        log_pipeline_event("drift", {"drift_severity": "OK"}, log_dir=tmp_path)

        events = get_recent_events(tmp_path)
        assert len(events) == 3
        assert events[0]["action"] == "train"
        assert events[1]["action"] == "predict"
        assert events[2]["action"] == "drift"

    def test_recent_events_limit(self, tmp_path):
        """get_recent_events respects the n limit."""
        from src.utils.audit import log_pipeline_event, get_recent_events

        for i in range(10):
            log_pipeline_event("train", {"version": f"v{i}.0.0"}, log_dir=tmp_path)

        events = get_recent_events(tmp_path, n=3)
        assert len(events) == 3
        # Should be the last 3
        assert events[0]["version"] == "v7.0.0"

    def test_format_audit_log(self, tmp_path):
        """Formatting audit log produces readable output."""
        from src.utils.audit import log_pipeline_event, get_recent_events, format_audit_log

        log_pipeline_event("train", {
            "version": "v1.0.0",
            "elapsed_seconds": 9.5,
            "status": "success",
        }, log_dir=tmp_path)

        events = get_recent_events(tmp_path)
        output = format_audit_log(events)

        assert "PIPELINE AUDIT TRAIL" in output
        assert "train" in output

    def test_empty_audit_log(self, tmp_path):
        """Formatting empty log returns helpful message."""
        from src.utils.audit import get_recent_events, format_audit_log

        events = get_recent_events(tmp_path)
        output = format_audit_log(events)
        assert "No pipeline events" in output

    def test_jsonl_format(self, tmp_path):
        """Audit entries are valid JSON lines."""
        from src.utils.audit import log_pipeline_event

        log_pipeline_event("train", {"version": "v1.0.0"}, log_dir=tmp_path)
        log_pipeline_event("predict", {"n_rows": 120}, log_dir=tmp_path)

        log_path = tmp_path / "pipeline_audit.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)  # Must not raise
            assert "timestamp" in parsed
            assert "action" in parsed
