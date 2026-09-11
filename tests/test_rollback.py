"""Tests for model reproducibility and rollback.

MLOps requirements:
  F2: Same inputs + same version = same answer
  F5: A way to go back to the previous version, one you have actually tried
"""

import pathlib
import shutil

import pandas as pd
import pytest

from src.model.registry import (
    load_model, save_model, rollback,
    get_next_version, list_versions, get_current_version,
)

MODEL_DIR = pathlib.Path("models")


class TestReproducibility:
    """F2: Same inputs + same version give the same answer."""

    def test_same_model_same_predictions(self):
        """Loading the same model twice produces identical predictions."""
        import numpy as np

        booster1, meta1, feat1 = load_model(MODEL_DIR)
        booster2, meta2, feat2 = load_model(MODEL_DIR)

        # Create a dummy input
        np.random.seed(42)
        X = np.random.rand(10, len(feat1))

        pred1 = booster1.predict(X)
        pred2 = booster2.predict(X)

        np.testing.assert_array_equal(pred1, pred2,
                                       err_msg="Same model produces different predictions")

    def test_model_has_metadata(self):
        """Saved model includes training metadata for audit trail."""
        _, meta, _ = load_model(MODEL_DIR)
        assert "version" in meta
        assert "trained_at" in meta
        assert "random_seed" in meta
        assert "data_hash" in meta
        assert "training_samples" in meta


class TestRollback:
    """F5: Rollback to previous version — actually tested."""

    def test_rollback_works(self, tmp_path):
        """Create two model versions and rollback to the first."""
        import lightgbm as lgb
        import numpy as np

        # Create a temporary model directory
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Train two simple models with different seeds
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = (X[:, 0] > 0.5).astype(int)

        # Model v1
        ds = lgb.Dataset(X, label=y)
        booster1 = lgb.train(
            {"objective": "binary", "num_leaves": 4, "verbose": -1, "seed": 1},
            ds, num_boost_round=10,
        )
        save_model(booster1, "v1.0.0", model_dir,
                   {"version": "v1.0.0", "random_seed": 1},
                   [f"f{i}" for i in range(5)], {}, {})

        # Model v2
        booster2 = lgb.train(
            {"objective": "binary", "num_leaves": 8, "verbose": -1, "seed": 2},
            ds, num_boost_round=20,
        )
        save_model(booster2, "v1.1.0", model_dir,
                   {"version": "v1.1.0", "random_seed": 2},
                   [f"f{i}" for i in range(5)], {}, {})

        # Current should be v1.1.0
        assert get_current_version(model_dir) == "v1.1.0"

        # Rollback to v1.0.0
        rollback(model_dir, "v1.0.0")
        assert get_current_version(model_dir) == "v1.0.0"

        # Load and verify it's the old model
        loaded, meta, _ = load_model(model_dir)
        assert meta["version"] == "v1.0.0"
        assert loaded.num_trees() == 10  # v1 had 10 trees

    def test_rollback_nonexistent_raises(self, tmp_path):
        """Rollback to a version that doesn't exist should fail."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            rollback(model_dir, "v99.0.0")

    def test_list_versions(self, tmp_path):
        """List available model versions."""
        model_dir = tmp_path / "models"
        (model_dir / "v1.0.0").mkdir(parents=True)
        (model_dir / "v1.1.0").mkdir(parents=True)
        (model_dir / "v2.0.0").mkdir(parents=True)

        versions = list_versions(model_dir)
        assert versions == ["v1.0.0", "v1.1.0", "v2.0.0"]


class TestModelRegistry:
    """Model versioning."""

    def test_get_next_version_empty(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        assert get_next_version(model_dir) == "v1.0.0"

    def test_get_next_version_increments(self, tmp_path):
        model_dir = tmp_path / "models"
        (model_dir / "v1.0.0").mkdir(parents=True)
        (model_dir / "v1.1.0").mkdir(parents=True)
        assert get_next_version(model_dir) == "v1.2.0"
