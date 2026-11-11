"""Analysis service used by the Streamlit front end (app.py).

Runs the same stages as ``main.py`` (clean -> features -> analysis -> model
-> risk scoring) on an uploaded DataFrame, entirely in memory. Nothing is
written to disk, so uploads never overwrite the project's own outputs.
"""

import io
import logging
from typing import Dict, Optional

import pandas as pd

import config
from src import (analysis, data_loader, evaluation, feature_engineering,
                 modeling, preprocessing, reporting)

logger = logging.getLogger(__name__)

MIN_ROWS = 500          # below this the chronological split is unreliable


class AnalysisError(Exception):
    """A problem with the uploaded data that the user can fix."""


def read_uploaded_csv(file_like) -> pd.DataFrame:
    """Read an uploaded CSV and check it has the required columns.

    Raises:
        AnalysisError: With a plain-language message when the file is unusable.
    """
    try:
        df = pd.read_csv(file_like)
    except pd.errors.EmptyDataError as exc:
        raise AnalysisError("The uploaded file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise AnalysisError(f"The file could not be read as a CSV: {exc}") from exc
    try:
        data_loader.validate_schema(df)
    except data_loader.DataLoadError as exc:
        raise AnalysisError(str(exc)) from exc
    if len(df) < MIN_ROWS:
        raise AnalysisError(f"Only {len(df)} rows found; at least {MIN_ROWS} are needed to train a model.")
    return df


def run_analysis(raw: pd.DataFrame, compare_all_models: bool = False) -> Dict[str, object]:
    """Run the full analysis on a raw sensor DataFrame.

    Args:
        raw: Data with all ``config.REQUIRED_COLS``.
        compare_all_models: Train all four models (slower). When False only
            the primary model (Random Forest) is trained.

    Returns:
        Dict of result tables and metrics for display and download.

    Raises:
        AnalysisError: If the data cannot be modelled (e.g. no failures).
    """
    try:
        clean, cleaning = preprocessing.clean_data(raw)
        features = feature_engineering.build_features(clean)
        if features[config.TARGET_COL].nunique() < 2:
            raise AnalysisError("The Failure column contains only one value (all 0 or all 1); "
                                "the model needs both normal and failure readings.")
        feature_cols = feature_engineering.get_feature_columns(features)
        train, test = modeling.chronological_split(features)
    except AnalysisError:
        raise
    except (ValueError, KeyError) as exc:
        raise AnalysisError(str(exc)) from exc

    estimators = modeling.get_estimators()
    names = list(estimators) if compare_all_models else [config.PRIMARY_MODEL_NAME]
    models = {n: modeling.train_model(n, estimators[n], feature_cols, train) for n in names}

    comparison = evaluation.compare_models(models, test, feature_cols)
    primary = models[config.PRIMARY_MODEL_NAME]
    y_test = test[config.TARGET_COL]
    prob = primary.predict_proba(test[feature_cols])[:, 1]

    scored = reporting.score_readings(features, primary, feature_cols, test[config.TIMESTAMP_COL].min())
    machine_summary = reporting.build_machine_risk_summary(scored)
    return {
        "overview": analysis.dataset_overview(clean),
        "cleaning": cleaning,
        "quality_table": analysis.data_quality_table(cleaning),
        "clean": clean,
        "features": features,
        "failure_by_machine": analysis.failure_rate_by_machine(features),
        "sensor_summary": analysis.sensor_summary_by_failure(features),
        "sensor_correlation": analysis.sensor_failure_correlation(features).rename("Correlation").reset_index()
                                .rename(columns={"index": "Sensor"}),
        "failure_trend": analysis.failure_trend(features),
        "anomalies": analysis.detect_anomalous_machines(features),
        "comparison": comparison,
        "primary_metrics": evaluation.compute_metrics(y_test, prob),
        "importance": evaluation.get_feature_importance(primary),
        "thresholds": evaluation.threshold_analysis(y_test, prob),
        "y_test": y_test.to_numpy(),
        "test_prob": prob,
        "scored": scored,
        "machine_summary": machine_summary,
        "kpis": reporting.calculate_kpis(scored, machine_summary),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame for a download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
