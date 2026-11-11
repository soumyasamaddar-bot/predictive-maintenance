"""Exploratory data analysis (business logic only - no plotting, no I/O).

Each function returns a DataFrame or Series that the visualization and
reporting modules can consume.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    """Headline facts about the dataset (size, span, target balance)."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_machines": df[config.MACHINE_ID_COL].nunique(),
        "start": df[config.TIMESTAMP_COL].min(),
        "end": df[config.TIMESTAMP_COL].max(),
        "failure_rate": float(df[config.TARGET_COL].mean()),
        "data_type": "Time-series (regular readings per machine)",
    }


def failure_rate_by_machine(df: pd.DataFrame) -> pd.DataFrame:
    """Failure-labelled reading rate per machine, highest first."""
    return (
        df.groupby(config.MACHINE_ID_COL)[config.TARGET_COL]
        .agg(failure_readings="sum", total_readings="count", failure_rate="mean")
        .sort_values("failure_rate", ascending=False)
        .reset_index()
    )


def sensor_summary_by_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of every sensor split by failure status, plus the % difference."""
    summary = df.groupby(config.TARGET_COL)[config.SENSOR_COLS].mean().T
    summary.columns = ["Normal_Mean", "Failure_Mean"]
    summary["Pct_Difference"] = (
        (summary["Failure_Mean"] - summary["Normal_Mean"]) / summary["Normal_Mean"] * 100
    )
    return summary.reset_index().rename(columns={"index": "Sensor"})


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between sensors, usage fields and the failure target."""
    cols = config.SENSOR_COLS + [config.OPERATING_HOURS_COL, config.TARGET_COL]
    return df[cols].corr()


def sensor_failure_correlation(df: pd.DataFrame) -> pd.Series:
    """Correlation of each sensor with failure, strongest first."""
    corr = correlation_matrix(df)[config.TARGET_COL].drop(config.TARGET_COL)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def failure_trend(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Failure-labelled readings aggregated over time (default weekly)."""
    trend = (
        df.set_index(config.TIMESTAMP_COL)[config.TARGET_COL]
        .resample(freq).agg(["sum", "mean"])
        .rename(columns={"sum": "failure_readings", "mean": "failure_rate"})
        .reset_index()
    )
    return trend


def failure_rate_by_operating_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rate across operating-hour bands (requires band features)."""
    return (
        df.groupby("Operating_Hours_Band")[config.TARGET_COL]
        .agg(failure_rate="mean", readings="count")
        .reindex(config.OPERATING_HOURS_LABELS)
        .reset_index()
    )


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count IQR-based outliers for each sensor."""
    rows = []
    for col in config.SENSOR_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - config.OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + config.OUTLIER_IQR_MULTIPLIER * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"Sensor": col, "Lower_Bound": low, "Upper_Bound": high,
                     "Outliers": n_out, "Outlier_Pct": n_out / df[col].notna().sum() * 100})
    return pd.DataFrame(rows)


def detect_anomalous_machines(df: pd.DataFrame) -> pd.DataFrame:
    """Rank machines by abnormal operating behaviour using Isolation Forest.

    The model is fitted on per-machine sensor statistics (mean and 95th
    percentile), so it flags machines that behave differently from the fleet.

    Returns:
        One row per machine with an ``Anomaly_Score`` (higher = more
        abnormal) and an ``Is_Anomalous`` flag.
    """
    grouped = df.groupby(config.MACHINE_ID_COL)[config.SENSOR_COLS]
    stats = pd.concat(
        [grouped.mean().add_suffix("_mean"), grouped.quantile(0.95).add_suffix("_p95")],
        axis=1,
    ).fillna(0.0)

    scaled = StandardScaler().fit_transform(stats)
    forest = IsolationForest(random_state=config.RANDOM_STATE, **config.ISOLATION_FOREST_PARAMS)
    forest.fit(scaled)

    result = pd.DataFrame({
        config.MACHINE_ID_COL: stats.index,
        "Anomaly_Score": -forest.score_samples(scaled),
        "Is_Anomalous": forest.predict(scaled) == -1,
    })
    return result.sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)


def data_quality_table(report: Dict[str, object]) -> pd.DataFrame:
    """Before/after cleaning table used in the documentation."""
    return pd.DataFrame({
        "Stage": ["Before Cleaning", "After Cleaning"],
        "Records": [report["rows_before"], report["rows_after"]],
        "Missing Values": [report["missing_before"], report["missing_after"]],
        "Duplicates": [report["duplicates_before"], report["duplicates_after"]],
    })
