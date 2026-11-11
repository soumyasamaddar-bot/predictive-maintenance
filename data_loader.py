"""Data loading and saving. Contains no business logic."""

import logging
from pathlib import Path

import pandas as pd

import config
from src import data_generator

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or fails schema validation."""


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all required columns are present.

    Raises:
        DataLoadError: If one or more required columns are missing.
    """
    missing = [c for c in config.REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Dataset is missing required columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with clear error messages.

    Raises:
        DataLoadError: If the file is missing, empty or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"File is empty: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Could not parse {path}: {exc}") from exc
    logger.info("Loaded %s (%d rows, %d columns)", path.name, *df.shape)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s (%d rows)", path.name, len(df))


def load_raw_data(generate_if_missing: bool = True) -> pd.DataFrame:
    """Load the raw sensor dataset, generating synthetic data if absent.

    Args:
        generate_if_missing: When True and no raw file exists, a synthetic
            dataset is generated and saved to ``config.RAW_DATA_FILE``.

    Returns:
        The raw, schema-validated DataFrame.
    """
    if not config.RAW_DATA_FILE.exists():
        if not generate_if_missing:
            raise DataLoadError(f"Raw data not found at {config.RAW_DATA_FILE}")
        logger.warning("No raw data found - generating synthetic dataset.")
        save_csv(data_generator.generate_sensor_data(), config.RAW_DATA_FILE)

    df = load_csv(config.RAW_DATA_FILE)
    validate_schema(df)
    return df
