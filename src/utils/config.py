"""Configuration loader.

Reads config/default.yaml and allows environment variable overrides.
Environment variables take precedence over YAML values.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml


_CONFIG_CACHE: dict[str, Any] | None = None


def _find_project_root() -> pathlib.Path:
    """Walk up from this file to find the project root (contains config/)."""
    current = pathlib.Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "default.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Cannot find project root (no config/default.yaml found)")


def load_config(config_path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML with environment variable overrides.

    Environment variable mapping:
        DATA_DIR    -> paths.data_dir
        MODEL_DIR   -> paths.model_dir
        COST_FP     -> cost.visit_cost
        COST_FN     -> cost.miss_cost
        RANDOM_SEED -> model.random_seed
        LOG_LEVEL   -> (returned as top-level key)
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = _find_project_root() / "config" / "default.yaml"

    config_path = pathlib.Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    env_overrides = {
        "DATA_DIR": ("paths", "data_dir"),
        "MODEL_DIR": ("paths", "model_dir"),
        "COST_FP": ("cost", "visit_cost", int),
        "COST_FN": ("cost", "miss_cost", int),
        "RANDOM_SEED": ("model", "random_seed", int),
    }

    for env_var, path_spec in env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            section = path_spec[0]
            key = path_spec[1]
            cast_fn = path_spec[2] if len(path_spec) > 2 else str
            if section not in config:
                config[section] = {}
            config[section][key] = cast_fn(value)

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    config["log_level"] = log_level

    if config_path is None:
        _CONFIG_CACHE = config

    return config


def get_data_dir(config: dict | None = None) -> pathlib.Path:
    """Get the data directory path from config."""
    if config is None:
        config = load_config()
    return pathlib.Path(config["paths"]["data_dir"]).resolve()


def get_model_dir(config: dict | None = None) -> pathlib.Path:
    """Get the model directory path from config."""
    if config is None:
        config = load_config()
    return pathlib.Path(config["paths"]["model_dir"]).resolve()
