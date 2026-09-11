"""Test suite for LPDG Gateway Health Prediction System.

Test modules:
    test_loader    — Data loading, ID normalization, schema validation
    test_features  — Feature engineering pipeline correctness
    test_e2e       — Full pipeline → valid predictions.csv
    test_drift     — Data drift detection
    test_rollback  — Model reproducibility and rollback
"""
