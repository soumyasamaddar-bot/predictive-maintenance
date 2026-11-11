"""Data cleaning and preparation.

Every function takes a DataFrame and returns a new one (inputs are never
mutated). Imputation that needs training statistics is deliberately *not*
done here - it lives inside the model pipeline so it is fitted on the
training set only, which prevents data leakage.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def standardize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and upper-case machine identifiers."""
    df = df.copy()
    df[config.MACHINE_ID_COL] = (
        df[config.MACHINE_ID_COL].astype(str).str.strip().str.upper()
    )
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and coerce numeric columns; drop unusable rows.

    Rows whose timestamp or target cannot be parsed are removed because they
    cannot be used for time-based analysis or supervised learning.
    """
    df = df.copy()
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce")
    for col in config.NUMERIC_COLS + [config.TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[config.TIMESTAMP_COL, config.TARGET_COL])
    if len(df) < before:
        logger.warning("Dropped %d rows with invalid timestamp/target", before - len(df))
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, then duplicate (machine, timestamp) keys."""
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[config.MACHINE_ID_COL, config.TIMESTAMP_COL], keep="first")
    return df


def fix_temperature_units(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Convert temperatures that were logged in deg F to deg C.

    Returns:
        The corrected DataFrame and the number of values converted.
    """
    df = df.copy()
    mask = df[config.TEMPERATURE_COL] > config.FAHRENHEIT_DETECTION_THRESHOLD
    df.loc[mask, config.TEMPERATURE_COL] = (df.loc[mask, config.TEMPERATURE_COL] - 32) * 5 / 9
    return df, int(mask.sum())


def mask_invalid_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Replace physically impossible values with NaN.

    Returns:
        The DataFrame and a ``{column: n_invalid_values}`` mapping.
    """
    df = df.copy()
    invalid_counts: Dict[str, int] = {}
    for col, (low, high) in config.VALID_RANGES.items():
        mask = (df[col] < low) | (df[col] > high)
        invalid_counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    return df, invalid_counts


def fill_sensor_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short sensor gaps within each machine.

    Only past values are used (no back-fill), so no future information leaks
    into earlier rows. Remaining gaps are imputed later in the model pipeline.
    """
    df = df.sort_values([config.MACHINE_ID_COL, config.TIMESTAMP_COL]).copy()
    df[config.SENSOR_COLS] = (
        df.groupby(config.MACHINE_ID_COL)[config.SENSOR_COLS]
        .ffill(limit=config.FORWARD_FILL_LIMIT)
    )
    return df


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run the full cleaning pipeline.

    Args:
        raw: Raw dataset as loaded from disk.

    Returns:
        The cleaned DataFrame (sorted by machine and time) and a report
        dictionary summarising every cleaning action, for documentation.
    """
    report: Dict[str, object] = {
        "rows_before": len(raw),
        "missing_before": int(raw[config.NUMERIC_COLS].isna().sum().sum()),
        "duplicates_before": int(raw.duplicated().sum()),
    }

    df = standardize_identifiers(raw)
    df = convert_data_types(df)
    df = remove_duplicates(df)
    df, report["temperature_unit_fixes"] = fix_temperature_units(df)
    df, report["invalid_values_by_column"] = mask_invalid_values(df)
    df = fill_sensor_gaps(df)
    df = df.reset_index(drop=True)

    report["rows_after"] = len(df)
    report["missing_after"] = int(df[config.NUMERIC_COLS].isna().sum().sum())
    report["duplicates_after"] = int(df.duplicated().sum())
    logger.info("Cleaning complete: %d -> %d rows, missing %d -> %d",
                report["rows_before"], report["rows_after"],
                report["missing_before"], report["missing_after"])
    return df, report
