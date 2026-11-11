"""Risk scoring, KPI calculation, recommendations and report/dashboard export.

Business rules (risk bands, recommendations) are driven by ``config``.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from src import data_loader, modeling

logger = logging.getLogger(__name__)


def assign_risk_status(probability: pd.Series) -> pd.Series:
    """Map failure probability to Healthy / Warning / High Risk."""
    warning = config.RISK_THRESHOLDS["Warning"]
    high = config.RISK_THRESHOLDS["High Risk"]
    return pd.Series(
        np.select([probability >= high, probability >= warning],
                  ["High Risk", "Warning"], default="Healthy"),
        index=probability.index,
    )


def score_readings(df: pd.DataFrame, pipeline: Pipeline, feature_cols, test_start) -> pd.DataFrame:
    """Score every reading with the model.

    Adds failure probability, risk status and a ``Dataset_Split`` column so
    readings seen during training are distinguishable from unseen ones.
    """
    scored = df.copy()
    scored["Failure_Probability"] = modeling.predict_probability(pipeline, scored, feature_cols)
    scored["Risk_Status"] = assign_risk_status(scored["Failure_Probability"])
    scored["Dataset_Split"] = np.where(scored[config.TIMESTAMP_COL] >= test_start, "Test", "Train")
    return scored


def build_machine_risk_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per machine based on its latest reading, highest risk first."""
    latest = (scored.sort_values(config.TIMESTAMP_COL)
              .groupby(config.MACHINE_ID_COL).tail(1))
    columns = [config.MACHINE_ID_COL, config.TIMESTAMP_COL, "Failure_Probability", "Risk_Status",
               config.TEMPERATURE_COL, config.VIBRATION_COL, config.OPERATING_HOURS_COL,
               config.MAINTENANCE_COUNT_COL, config.MACHINE_AGE_COL]
    summary = latest[columns].sort_values("Failure_Probability", ascending=False).reset_index(drop=True)
    summary["Priority_Rank"] = np.arange(1, len(summary) + 1)
    summary["Recommendation"] = summary["Risk_Status"].map(config.RECOMMENDATIONS)
    return summary


def calculate_kpis(scored: pd.DataFrame, machine_summary: pd.DataFrame) -> pd.DataFrame:
    """Dashboard KPI cards (fleet-level)."""
    status = machine_summary["Risk_Status"].value_counts()
    kpis = {
        "Total Machines": len(machine_summary),
        "Healthy Machines": int(status.get("Healthy", 0)),
        "Warning Machines": int(status.get("Warning", 0)),
        "High Risk Machines": int(status.get("High Risk", 0)),
        "Failure Rate (%)": round(scored[config.TARGET_COL].mean() * 100, 2),
        "Average Temperature": round(scored[config.TEMPERATURE_COL].mean(), 2),
        "Average Vibration": round(scored[config.VIBRATION_COL].mean(), 3),
        "Total Readings": len(scored),
    }
    return pd.DataFrame({"KPI": list(kpis), "Value": list(kpis.values())})


def export_dashboard_tables(scored: pd.DataFrame, machine_summary: pd.DataFrame,
                            kpis: pd.DataFrame) -> None:
    """Write the CSV tables used as Power BI / Tableau data sources."""
    keep = [config.MACHINE_ID_COL, config.TIMESTAMP_COL] + config.SENSOR_COLS + [
        config.OPERATING_HOURS_COL, config.MAINTENANCE_COUNT_COL, config.MACHINE_AGE_COL,
        config.TARGET_COL, "Failure_Probability", "Risk_Status", "Dataset_Split"]
    data_loader.save_csv(scored[keep], config.DASHBOARD_DIR / "scored_readings.csv")
    data_loader.save_csv(machine_summary, config.DASHBOARD_DIR / "machine_risk_summary.csv")
    data_loader.save_csv(kpis, config.DASHBOARD_DIR / "kpi_summary.csv")


def _data_dictionary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [(field, kind, desc) for field, (kind, desc) in config.DATA_DICTIONARY.items()],
        columns=["Field", "Type", "Description"])


def save_report_tables(tables: Dict[str, pd.DataFrame]) -> None:
    """Save every named table to ``outputs/reports`` as CSV."""
    for name, table in tables.items():
        data_loader.save_csv(table, config.REPORTS_DIR / f"{name}.csv")
    data_loader.save_csv(_data_dictionary_table(), config.REPORTS_DIR / "data_dictionary.csv")


def write_summary_report(overview: Dict[str, object], cleaning: Dict[str, object],
                         comparison: pd.DataFrame, primary_metrics: Dict[str, float],
                         top_features: pd.DataFrame, machine_summary: pd.DataFrame) -> None:
    """Write a plain-text run summary with the key results for documentation."""
    lines = [
        "PREDICTIVE MAINTENANCE - RUN SUMMARY", "=" * 40,
        f"Machines: {overview['n_machines']}  |  Readings: {overview['n_rows']}",
        f"Period: {overview['start']} to {overview['end']}",
        f"Failure-labelled readings: {overview['failure_rate'] * 100:.2f}%", "",
        "DATA CLEANING",
        f"  Rows {cleaning['rows_before']} -> {cleaning['rows_after']}",
        f"  Missing values {cleaning['missing_before']} -> {cleaning['missing_after']}",
        f"  Duplicates removed: {cleaning['duplicates_before']}",
        f"  Temperature unit fixes (F -> C): {cleaning['temperature_unit_fixes']}",
        f"  Invalid values masked: {sum(cleaning['invalid_values_by_column'].values())}", "",
        f"MODEL COMPARISON (test set, threshold {config.DECISION_THRESHOLD})",
        comparison.round(3).to_string(index=False), "",
        f"PRIMARY MODEL: {config.PRIMARY_MODEL_NAME}",
        f"  Recall {primary_metrics['Recall']:.3f} | Precision {primary_metrics['Precision']:.3f} "
        f"| F1 {primary_metrics['F1']:.3f} | ROC-AUC {primary_metrics['ROC_AUC']:.3f}",
        f"  Missed failures (FN): {primary_metrics['FN']}  |  False alarms (FP): {primary_metrics['FP']}",
        "  A false negative is a missed failure (unplanned downtime); a false positive is an",
        "  unnecessary inspection. In maintenance, missed failures usually cost more, so",
        "  recall is prioritised.", "",
        f"TOP {config.TOP_N_FEATURES} FEATURES", top_features.head(config.TOP_N_FEATURES).round(4).to_string(index=False), "",
        f"TOP {config.TOP_N_RISK_MACHINES} PRIORITY MACHINES",
        machine_summary.head(config.TOP_N_RISK_MACHINES)[
            [config.MACHINE_ID_COL, "Failure_Probability", "Risk_Status", "Recommendation"]
        ].round(3).to_string(index=False),
    ]
    path = config.REPORTS_DIR / "run_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", path.name)
