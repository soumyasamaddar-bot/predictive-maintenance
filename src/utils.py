"""Shared helpers: logging setup, directory creation and small utilities."""

import logging
import sys
from pathlib import Path
from typing import Iterable

import config


def setup_logging() -> None:
    """Configure root logging to write to both the console and a log file."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # silence font/category chatter


def ensure_directories(directories: Iterable[Path] = None) -> None:
    """Create every output directory used by the pipeline if it is missing.

    Args:
        directories: Optional iterable of paths. Defaults to
            ``config.ALL_OUTPUT_DIRS``.
    """
    for directory in directories or config.ALL_OUTPUT_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)


def format_percent(value: float, decimals: int = 2) -> str:
    """Return ``value`` (0-1 fraction) formatted as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
