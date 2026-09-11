"""ML model training, prediction, evaluation, and versioning.

Modules:
    train    — LightGBM training pipeline with gateway-level CV
    predict  — Inference pipeline with human-readable reason generation
    evaluate — Cost-based comparison against baseline
    registry — Model versioning, saving, loading, and rollback
"""

# Lazy imports to avoid loading LightGBM at package import time
# (LightGBM requires libgomp which may not be available in all contexts)


def load_model(*args, **kwargs):
    """Load a trained model from the registry."""
    from src.model.registry import load_model as _load_model
    return _load_model(*args, **kwargs)


def rollback(*args, **kwargs):
    """Rollback to a previous model version."""
    from src.model.registry import rollback as _rollback
    return _rollback(*args, **kwargs)


def list_versions(*args, **kwargs):
    """List all available model versions."""
    from src.model.registry import list_versions as _list_versions
    return _list_versions(*args, **kwargs)


__all__ = ["load_model", "rollback", "list_versions"]
