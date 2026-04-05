"""Centralised logging configuration for the M5 pipeline."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Configure root logger with console (and optional file) handler."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ["cmdstanpy", "prophet", "lightgbm", "numexpr"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
