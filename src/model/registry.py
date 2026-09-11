"""Model registry — versioning, saving, loading, and rollback.

Models are stored as:
    models/v1.0.0/
        model.lgb           — serialized LightGBM booster
        metadata.json       — training info, metrics, hashes
        features.json       — feature names and importance
        config.yaml         — exact config used for training
    models/current -> v1.0.0/  — symlink to active version

MLOps requirements addressed:
    F1: Training and predicting use separate model files
    F2: Same inputs + same version = same answer (reproducibility)
    F5: Rollback by moving the symlink
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil

import lightgbm as lgb
import yaml

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def get_next_version(model_dir: pathlib.Path) -> str:
    """Determine the next semantic version (v{major}.{minor}.{patch}).

    Increments the minor version from the highest existing version.
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    pattern = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
    for item in model_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing.append((int(match.group(1)),
                                 int(match.group(2)),
                                 int(match.group(3))))

    if not existing:
        return "v1.0.0"

    existing.sort()
    major, minor, patch = existing[-1]
    return f"v{major}.{minor + 1}.0"


def save_model(
    booster: lgb.Booster,
    version: str,
    model_dir: pathlib.Path,
    metadata: dict,
    feature_names: list[str],
    feature_importance: dict[str, float],
    config: dict,
) -> pathlib.Path:
    """Save a trained model with full metadata.

    Args:
        booster: Trained LightGBM booster.
        version: Version string (e.g., "v1.0.0").
        model_dir: Root model directory.
        metadata: Training metadata dict.
        feature_names: Ordered list of feature names.
        feature_importance: Feature name -> importance score.
        config: Full config dict.

    Returns:
        Path to the saved model version directory.
    """
    model_dir = pathlib.Path(model_dir)
    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = version_dir / "model.lgb"
    booster.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    # Save metadata
    meta_path = version_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save feature info
    features_path = version_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "feature_importance": feature_importance,
            "n_features": len(feature_names),
        }, f, indent=2)

    # Save config snapshot
    config_path = version_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Update current symlink
    _set_current(model_dir, version)

    logger.info("Model %s saved with %d features", version, len(feature_names))
    return version_dir


def _set_current(model_dir: pathlib.Path, version: str) -> None:
    """Set the 'current' symlink to point to the given version."""
    current_link = model_dir / "current"

    # Remove existing symlink or directory
    if current_link.is_symlink() or current_link.exists():
        if current_link.is_symlink():
            current_link.unlink()
        else:
            shutil.rmtree(current_link)

    # Create relative symlink
    current_link.symlink_to(version)
    logger.info("Set current model to %s", version)


def load_model(model_dir: pathlib.Path | None = None, version: str | None = None):
    """Load a trained model from the registry.

    Args:
        model_dir: Root model directory. Defaults to config value.
        version: Specific version to load. Defaults to 'current'.

    Returns:
        Tuple of (booster, metadata, feature_names).
    """
    if model_dir is None:
        from src.utils.config import get_model_dir
        model_dir = get_model_dir()

    model_dir = pathlib.Path(model_dir)

    if version is None:
        version_dir = model_dir / "current"
        if not version_dir.exists():
            raise FileNotFoundError(
                f"No current model found at {version_dir}. Run training first."
            )
        # Resolve symlink to get actual version
        resolved = version_dir.resolve()
        version = resolved.name
    else:
        version_dir = model_dir / version

    if not version_dir.exists():
        raise FileNotFoundError(f"Model version not found: {version_dir}")

    # Load model
    model_path = version_dir / "model.lgb"
    booster = lgb.Booster(model_file=str(model_path))

    # Load metadata
    meta_path = version_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Load feature names
    features_path = version_dir / "features.json"
    with open(features_path) as f:
        features_info = json.load(f)
    feature_names = features_info["feature_names"]

    logger.info("Loaded model %s (%d features, %d trees)",
                version, len(feature_names), booster.num_trees())

    return booster, metadata, feature_names


def rollback(model_dir: pathlib.Path, target_version: str) -> None:
    """Rollback to a previous model version.

    Simply moves the 'current' symlink. The old model files are untouched.

    Args:
        model_dir: Root model directory.
        target_version: Version to rollback to (e.g., "v1.0.0").
    """
    model_dir = pathlib.Path(model_dir)
    target_dir = model_dir / target_version

    if not target_dir.exists():
        available = list_versions(model_dir)
        raise FileNotFoundError(
            f"Version {target_version} not found. Available: {available}"
        )

    # Record what we're rolling back from
    current_link = model_dir / "current"
    if current_link.is_symlink():
        old_version = os.readlink(current_link)
        logger.info("Rolling back from %s to %s", old_version, target_version)
    else:
        logger.info("Setting model to %s (no previous version)", target_version)

    _set_current(model_dir, target_version)
    logger.info("Rollback complete. Current model is now %s", target_version)


def list_versions(model_dir: pathlib.Path) -> list[str]:
    """List all available model versions, sorted."""
    model_dir = pathlib.Path(model_dir)
    versions = []
    pattern = re.compile(r"^v\d+\.\d+\.\d+$")
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                versions.append(item.name)
    return sorted(versions)


def get_current_version(model_dir: pathlib.Path | str) -> str | None:
    """Get the current active model version, or None if no model is deployed."""
    model_dir = pathlib.Path(model_dir)
    current_link = model_dir / "current"
    if current_link.is_symlink():
        return os.readlink(current_link)
    return None
