"""All chart creation. Functions receive prepared data and save figures.

No analysis or modelling logic lives here - inputs are the DataFrames
produced by ``analysis``, ``evaluation`` and ``reporting``.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

import config

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to the figures folder and release its memory."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", path.name)
    return path


def _new_figure(figsize=None):
    sns.set_theme(style=config.SEABORN_STYLE)
    return plt.subplots(figsize=figsize or config.DEFAULT_FIGSIZE)


def plot_failure_rate_by_machine(rates: pd.DataFrame) -> Path:
    fig, ax = _new_figure()
    sns.barplot(data=rates, x=config.MACHINE_ID_COL, y="failure_rate", color="#c0392b", ax=ax)
    ax.set(title="Failure Rate by Machine", xlabel="Machine", ylabel="Share of readings labelled failure")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, "01_failure_rate_by_machine")


def plot_sensor_distribution(df: pd.DataFrame, sensor: str, file_name: str) -> Path:
    """Histogram of a sensor split by failure status."""
    fig, ax = _new_figure()
    sns.histplot(data=df, x=sensor, hue=config.TARGET_COL, bins=50, stat="density",
                 common_norm=False, element="step", palette={0: "#2980b9", 1: "#c0392b"}, ax=ax)
    ax.set(title=f"{sensor} Distribution: Normal (0) vs Failure (1)")
    return _save(fig, file_name)


def plot_sensor_boxplot(df: pd.DataFrame, sensor: str, file_name: str) -> Path:
    fig, ax = _new_figure((7, 6))
    sns.boxplot(data=df, x=config.TARGET_COL, y=sensor, hue=config.TARGET_COL,
                palette={0: "#2980b9", 1: "#c0392b"}, legend=False, ax=ax)
    ax.set(title=f"{sensor} by Failure Status", xlabel="Failure (0 = normal, 1 = failure)")
    return _save(fig, file_name)


def plot_pressure_rpm_scatter(df: pd.DataFrame) -> Path:
    fig, ax = _new_figure()
    sample = df.sample(n=min(6000, len(df)), random_state=config.RANDOM_STATE)
    sns.scatterplot(data=sample, x=config.RPM_COL, y=config.PRESSURE_COL,
                    hue=config.TARGET_COL, palette={0: "#2980b9", 1: "#c0392b"},
                    alpha=0.5, s=18, ax=ax)
    ax.set(title="Pressure vs RPM")
    return _save(fig, "05_pressure_rpm_scatter")


def plot_correlation_heatmap(corr: pd.DataFrame) -> Path:
    fig, ax = _new_figure((9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: Sensors, Operating Hours and Failure")
    return _save(fig, "06_correlation_heatmap")


def plot_failure_trend(trend: pd.DataFrame) -> Path:
    fig, ax = _new_figure()
    ax.plot(trend[config.TIMESTAMP_COL], trend["failure_readings"], marker="o", color="#c0392b")
    ax.set(title="Weekly Failure-Labelled Readings (Fleet)", xlabel="Week", ylabel="Readings")
    return _save(fig, "07_failure_trend")


def plot_operating_hours_failure(rates: pd.DataFrame) -> Path:
    fig, ax = _new_figure((8, 6))
    sns.barplot(data=rates, x="Operating_Hours_Band", y="failure_rate", color="#e67e22", ax=ax)
    ax.set(title="Failure Rate by Operating Hours Band", xlabel="Operating hours", ylabel="Failure rate")
    return _save(fig, "08_operating_hours_vs_failure")


def plot_sensor_trends(df: pd.DataFrame, machine_ids: List[str]) -> Path:
    """Temperature and vibration over time for selected machines."""
    sns.set_theme(style=config.SEABORN_STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for sensor, ax in zip((config.TEMPERATURE_COL, config.VIBRATION_COL), axes):
        for mid in machine_ids:
            part = df[df[config.MACHINE_ID_COL] == mid]
            ax.plot(part[config.TIMESTAMP_COL], part[sensor], linewidth=0.8, label=mid)
            failures = part[part[config.TARGET_COL] == 1]
            ax.scatter(failures[config.TIMESTAMP_COL], failures[sensor], s=6, color="red", alpha=0.4)
        ax.set_ylabel(sensor)
    axes[0].set_title("Sensor Trends (red dots = failure-labelled readings)")
    axes[0].legend(loc="upper right", ncol=len(machine_ids))
    return _save(fig, "09_sensor_trends")


def plot_anomaly_scores(anomalies: pd.DataFrame) -> Path:
    fig, ax = _new_figure()
    colors = anomalies["Is_Anomalous"].map({True: "#c0392b", False: "#7f8c8d"})
    ax.bar(anomalies[config.MACHINE_ID_COL], anomalies["Anomaly_Score"], color=colors)
    ax.set(title="Machine Anomaly Score (Isolation Forest; red = flagged)", ylabel="Anomaly score")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, "10_machine_anomaly_scores")


def plot_model_comparison(comparison: pd.DataFrame) -> Path:
    metrics = ["Precision", "Recall", "F1", "ROC_AUC"]
    melted = comparison.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    fig, ax = _new_figure((11, 6))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set(title="Model Comparison on Test Set", ylim=(0, 1.05))
    # Legend below the plot so it never covers the bars.
    ax.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(metrics), frameon=False)
    return _save(fig, "11_model_comparison")


def plot_confusion_matrix(y_true, y_prob, model_name: str) -> Path:
    fig, ax = _new_figure((6, 5))
    y_pred = (np.asarray(y_prob) >= config.DECISION_THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Healthy", "Failure"], cmap="Blues", ax=ax, colorbar=False,
        values_format="d", text_kw={"fontsize": 16, "fontweight": "bold"})
    ax.grid(False)  # grid lines made the counts hard to read
    ax.set_title(f"Confusion Matrix - {model_name}")
    return _save(fig, "12_confusion_matrix")


def plot_roc_pr_curves(y_true, probabilities: Dict[str, np.ndarray]) -> Path:
    """ROC and precision-recall curves for every model."""
    sns.set_theme(style=config.SEABORN_STYLE)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ax_roc.plot(fpr, tpr, label=name)
        ax_pr.plot(recall, precision, label=name)
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set(title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax_roc.legend(loc="lower right")
    return _save(fig, "13_roc_pr_curves")


def plot_feature_importance(importance: pd.DataFrame, model_name: str) -> Path:
    top = importance.head(config.TOP_N_FEATURES).iloc[::-1]
    fig, ax = _new_figure()
    ax.barh(top["Feature"], top["Importance"], color="#16a085")
    ax.set(title=f"Top {config.TOP_N_FEATURES} Features - {model_name}", xlabel="Importance")
    return _save(fig, "14_feature_importance")


def plot_risk_distribution(summary: pd.DataFrame) -> Path:
    counts = summary["Risk_Status"].value_counts().reindex(config.RISK_LABELS).fillna(0)
    fig, ax = _new_figure((7, 5))
    ax.bar(counts.index, counts.values, color=["#27ae60", "#f39c12", "#c0392b"])
    ax.set(title="Machine Risk Distribution (latest reading)", ylabel="Number of machines")
    for i, value in enumerate(counts.values):
        ax.text(i, value, int(value), ha="center", va="bottom")
    return _save(fig, "15_risk_distribution")


def plot_top_risk_machines(summary: pd.DataFrame) -> Path:
    top = summary.head(config.TOP_N_RISK_MACHINES).iloc[::-1]
    fig, ax = _new_figure()
    ax.barh(top[config.MACHINE_ID_COL], top["Failure_Probability"], color="#c0392b")
    ax.set(title="Top High-Risk Machines", xlabel="Predicted failure probability", xlim=(0, 1))
    return _save(fig, "16_top_risk_machines")
