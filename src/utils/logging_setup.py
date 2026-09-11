"""Structured logging setup.

Provides consistent, readable log output across all modules.
Log level is controlled via the LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure root logger with structured output.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL environment variable or INFO.

    Returns:
        The root logger, configured.
    """
    global _CONFIGURED

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(numeric_level)
        _CONFIGURED = True

    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Calls setup_logging() if not yet configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
