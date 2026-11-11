"""Model evaluation metrics and explainability (no plotting)."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

import config

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_prob, threshold: float = None) -> Dict[str, float]:
    """Compute classification metrics at a probability threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted failure probabilities.
        threshold: Decision threshold (defaults to ``config.DECISION_THRESHOLD``).

    Returns:
        Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion
        matrix counts (TN, FP, FN, TP).
    """
    threshold = config.DECISION_THRESHOLD if threshold is None else threshold
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def compare_models(models: Dict[str, Pipeline], test: pd.DataFrame,
                   feature_cols) -> pd.DataFrame:
    """Evaluate every model on the test set and return a comparison table."""
    rows = []
    for name, pipeline in models.items():
        prob = pipeline.predict_proba(test[feature_cols])[:, 1]
        rows.append({"Model": name, **compute_metrics(test[config.TARGET_COL], prob)})
    return pd.DataFrame(rows).sort_values("Recall", ascending=False).reset_index(drop=True)


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Feature importance (tree models) or absolute coefficients (linear)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model does not expose feature importance.")
    return (pd.DataFrame({"Feature": names, "Importance": values})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def threshold_analysis(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision/recall trade-off across thresholds (supports recall-focused tuning)."""
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.95, 0.1)
    rows = [{"Threshold": round(float(t), 2), **{
        k: v for k, v in compute_metrics(y_true, y_prob, t).items()
        if k in ("Precision", "Recall", "F1", "FP", "FN")}} for t in thresholds]
    return pd.DataFrame(rows)
