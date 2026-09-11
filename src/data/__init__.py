"""Data loading, feature engineering, label construction, and drift detection.

Modules:
    loader   — Load and validate all datasets with ID normalization
    features — 68-feature engineering pipeline from 5 data sources
    labels   — Training label construction from field visit outcomes
    drift    — Data drift detection (schema, statistical, volume)
"""

from src.data.loader import load_all, DataBundle
from src.data.features import build_features_for_week
from src.data.labels import build_training_labels

__all__ = ["load_all", "DataBundle", "build_features_for_week", "build_training_labels"]
