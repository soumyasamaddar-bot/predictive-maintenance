"""Entry point for the Predictive Maintenance pipeline.

Usage:
    python main.py                    # run the full pipeline
    python main.py --regenerate-data  # rebuild the synthetic dataset first
    python main.py --skip-plots       # skip figure generation (faster)

Pipeline stages:
    1. Load raw data           (src/data_loader.py)
    2. Clean & prepare         (src/preprocessing.py)
    3. Exploratory analysis    (src/analysis.py, src/visualization.py)
    4. Feature engineering     (src/feature_engineering.py)
    5. Train & evaluate models (src/modeling.py, src/evaluation.py)
    6. Risk scoring & export   (src/reporting.py)
"""

import argparse
import logging
import sys
from typing import Dict, Tuple

import pandas as pd

import config
from src import (analysis, data_loader, evaluation, feature_engineering,
                 modeling, preprocessing, reporting, utils, visualization)

logger = logging.getLogger("pipeline")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Predictive maintenance pipeline")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Delete and regenerate the synthetic raw dataset.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip figure generation.")
    return parser.parse_args()


def run_data_preparation(regenerate: bool) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Stages 1-2: load, validate and clean the raw data."""
    if regenerate and config.RAW_DATA_FILE.exists():
        config.RAW_DATA_FILE.unlink()
    raw = data_loader.load_raw_data()
    clean, report = preprocessing.clean_data(raw)
    data_loader.save_csv(clean, config.PROCESSED_DATA_FILE)
    return clean, report


def run_exploratory_analysis(features: pd.DataFrame, plot: bool) -> Dict[str, pd.DataFrame]:
    """Stage 3: compute EDA tables and (optionally) draw the charts."""
    tables = {
        "failure_rate_by_machine": analysis.failure_rate_by_machine(features),
        "sensor_summary_by_failure": analysis.sensor_summary_by_failure(features),
        "sensor_failure_correlation": analysis.sensor_failure_correlation(features).rename("Correlation").reset_index(),
        "failure_trend_weekly": analysis.failure_trend(features),
        "failure_rate_by_operating_hours": analysis.failure_rate_by_operating_hours(features),
        "sensor_outliers": analysis.outlier_summary(features),
        "machine_anomaly_scores": analysis.detect_anomalous_machines(features),
    }
    if plot:
        visualization.plot_failure_rate_by_machine(tables["failure_rate_by_machine"])
        visualization.plot_sensor_distribution(features, config.TEMPERATURE_COL, "02_temperature_distribution")
        visualization.plot_sensor_boxplot(features, config.VIBRATION_COL, "03_vibration_boxplot")
        visualization.plot_sensor_distribution(features, config.VIBRATION_COL, "04_vibration_distribution")
        visualization.plot_pressure_rpm_scatter(features)
        visualization.plot_correlation_heatmap(analysis.correlation_matrix(features))
        visualization.plot_failure_trend(tables["failure_trend_weekly"])
        visualization.plot_operating_hours_failure(tables["failure_rate_by_operating_hours"])
        top_machines = tables["failure_rate_by_machine"][config.MACHINE_ID_COL].head(config.SENSOR_TREND_MACHINES).tolist()
        visualization.plot_sensor_trends(features, top_machines)
        visualization.plot_anomaly_scores(tables["machine_anomaly_scores"])
    return tables


def run_modeling(features: pd.DataFrame, plot: bool):
    """Stage 5: split chronologically, train, compare and explain models."""
    feature_cols = feature_engineering.get_feature_columns(features)
    train, test = modeling.chronological_split(features)
    models = modeling.train_all_models(feature_cols, train)

    comparison = evaluation.compare_models(models, test, feature_cols)
    primary = models[config.PRIMARY_MODEL_NAME]
    y_test = test[config.TARGET_COL]
    probabilities = {name: m.predict_proba(test[feature_cols])[:, 1] for name, m in models.items()}
    primary_metrics = evaluation.compute_metrics(y_test, probabilities[config.PRIMARY_MODEL_NAME])
    importance = evaluation.get_feature_importance(primary)
    thresholds = evaluation.threshold_analysis(y_test, probabilities[config.PRIMARY_MODEL_NAME])

    modeling.save_model(primary, feature_cols)
    if plot:
        visualization.plot_model_comparison(comparison)
        visualization.plot_confusion_matrix(y_test, probabilities[config.PRIMARY_MODEL_NAME], config.PRIMARY_MODEL_NAME)
        visualization.plot_roc_pr_curves(y_test, probabilities)
        visualization.plot_feature_importance(importance, config.PRIMARY_MODEL_NAME)

    return {
        "primary": primary, "feature_cols": feature_cols, "test_start": test[config.TIMESTAMP_COL].min(),
        "comparison": comparison, "primary_metrics": primary_metrics,
        "importance": importance, "thresholds": thresholds,
    }


def run_scoring_and_export(features: pd.DataFrame, results: dict, overview: dict,
                           cleaning: dict, eda_tables: Dict[str, pd.DataFrame], plot: bool) -> None:
    """Stage 6: score machines, build KPIs, and export reports + dashboard data."""
    scored = reporting.score_readings(features, results["primary"], results["feature_cols"], results["test_start"])
    machine_summary = reporting.build_machine_risk_summary(scored)
    kpis = reporting.calculate_kpis(scored, machine_summary)
    reporting.export_dashboard_tables(scored, machine_summary, kpis)

    tables = {
        **eda_tables,
        "data_quality_before_after": analysis.data_quality_table(cleaning),
        "model_comparison": results["comparison"],
        "feature_importance": results["importance"],
        "threshold_analysis": results["thresholds"],
    }
    reporting.save_report_tables(tables)
    reporting.write_summary_report(overview, cleaning, results["comparison"],
                                   results["primary_metrics"], results["importance"], machine_summary)
    if plot:
        visualization.plot_risk_distribution(machine_summary)
        visualization.plot_top_risk_machines(machine_summary)


def main() -> int:
    """Run the full pipeline. Returns a process exit code."""
    args = parse_arguments()
    utils.ensure_directories()
    utils.setup_logging()
    plot = not args.skip_plots

    try:
        clean, cleaning_report = run_data_preparation(args.regenerate_data)
        features = feature_engineering.build_features(clean)
        data_loader.save_csv(features, config.FEATURE_DATA_FILE)

        overview = analysis.dataset_overview(clean)
        eda_tables = run_exploratory_analysis(features, plot)
        results = run_modeling(features, plot)
        run_scoring_and_export(features, results, overview, cleaning_report, eda_tables, plot)
    except (data_loader.DataLoadError, ValueError, FileNotFoundError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    except Exception:  # last-resort handler so the failure is logged with a traceback
        logger.exception("Unexpected error in pipeline")
        return 2

    logger.info("Pipeline finished successfully. See the outputs/ folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
