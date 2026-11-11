"""Feature engineering from raw sensor and machine-usage data.

All rolling / lag / expanding features are computed per machine and use only
*past* readings, so they are safe to compute before the train/test split
without leaking future information.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _by_machine(df: pd.DataFrame, col: str):
    """Shortcut: group a column by machine."""
    return df.groupby(config.MACHINE_ID_COL)[col]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean / max / min / std over ``config.ROLLING_WINDOW`` readings."""
    df = df.copy()
    window = config.ROLLING_WINDOW
    for col in config.ROLLING_FEATURE_COLS:
        grouped = _by_machine(df, col)
        df[f"{col}_roll_mean"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_roll_max"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).max())
        df[f"{col}_roll_min"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).min())
        df[f"{col}_roll_std"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        ).fillna(0.0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged values and change since the previous reading."""
    df = df.copy()
    for col in config.LAG_FEATURE_COLS:
        grouped = _by_machine(df, col)
        for lag in config.LAG_PERIODS:
            df[f"{col}_lag{lag}"] = grouped.shift(lag)
        df[f"{col}_diff1"] = grouped.diff(1)
    return df


def add_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance outside the normal operating range, scaled by range width.

    0 means the reading is inside the normal band; 0.5 means it is half a
    band-width beyond the nearest limit.
    """
    df = df.copy()
    for col, (low, high) in config.NORMAL_RANGES.items():
        width = high - low
        below = (low - df[col]).clip(lower=0)
        above = (df[col] - high).clip(lower=0)
        df[f"{col}_deviation"] = (below + above) / width
    return df


def add_band_features(df: pd.DataFrame) -> pd.DataFrame:
    """Operating-hours bands and machine-age groups (as text categories)."""
    df = df.copy()
    df["Operating_Hours_Band"] = pd.cut(
        df[config.OPERATING_HOURS_COL], bins=config.OPERATING_HOURS_BINS,
        labels=config.OPERATING_HOURS_LABELS, include_lowest=True,
    ).astype(str)
    df["Machine_Age_Group"] = pd.cut(
        df[config.MACHINE_AGE_COL], bins=config.MACHINE_AGE_BINS,
        labels=config.MACHINE_AGE_LABELS, include_lowest=True,
    ).astype(str)
    return df


def add_maintenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Maintenance frequency: maintenance events per year of machine age."""
    df = df.copy()
    age_years = df[config.MACHINE_AGE_COL].clip(lower=0.5)
    df["Maintenance_Frequency"] = df[config.MAINTENANCE_COUNT_COL] / age_years
    return df


def add_machine_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal machine-level aggregates (expanding mean) and relative readings."""
    df = df.copy()
    for col in (config.TEMPERATURE_COL, config.VIBRATION_COL):
        expanding_mean = _by_machine(df, col).transform(lambda s: s.expanding(min_periods=1).mean())
        df[f"{col}_machine_mean"] = expanding_mean
        df[f"{col}_vs_machine_mean"] = df[col] - expanding_mean
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature-engineering step.

    Args:
        df: Cleaned data sorted by machine and timestamp.

    Returns:
        DataFrame with the original columns plus engineered features.
    """
    df = df.sort_values([config.MACHINE_ID_COL, config.TIMESTAMP_COL]).reset_index(drop=True)
    for step in (add_rolling_features, add_lag_features, add_deviation_features,
                 add_band_features, add_maintenance_features, add_machine_level_features):
        df = step(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    logger.info("Feature engineering complete: %d columns", df.shape[1])
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the model input columns (everything except IDs, time, target)."""
    return [c for c in df.columns if c not in config.EXCLUDE_FROM_FEATURES]
