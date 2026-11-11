"""Model construction, chronological splitting, training and persistence."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config

logger = logging.getLogger(__name__)


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by time (not randomly) to avoid leakage.

    The last ``config.TEST_SIZE`` fraction of the timeline becomes the test
    set. Training rows within ``config.SPLIT_EMBARGO_HOURS`` of the split
    point are dropped because their labels look into the test period.

    Raises:
        ValueError: If either split would contain only one class.
    """
    timestamps = df[config.TIMESTAMP_COL]
    cutoff = timestamps.quantile(1 - config.TEST_SIZE)
    embargo_start = cutoff - pd.Timedelta(hours=config.SPLIT_EMBARGO_HOURS)

    train = df[timestamps < embargo_start]
    test = df[timestamps >= cutoff]
    for name, part in (("train", train), ("test", test)):
        if part[config.TARGET_COL].nunique() < 2:
            raise ValueError(f"The {name} split contains a single class; "
                             "provide more data or adjust config.TEST_SIZE.")
    logger.info("Split at %s: train=%d rows (%.2f%% failures), test=%d rows (%.2f%% failures)",
                cutoff, len(train), train[config.TARGET_COL].mean() * 100,
                len(test), test[config.TARGET_COL].mean() * 100)
    return train, test


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """Imputation + scaling for numeric features, one-hot for categoricals.

    Statistics are learned inside the pipeline, i.e. from training data only.
    """
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def get_estimators() -> Dict[str, object]:
    """Return the candidate classifiers, configured from ``config.MODEL_PARAMS``."""
    params = config.MODEL_PARAMS
    state = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(random_state=state, **params["Logistic Regression"]),
        "Decision Tree": DecisionTreeClassifier(random_state=state, **params["Decision Tree"]),
        "Random Forest": RandomForestClassifier(random_state=state, **params["Random Forest"]),
        "Gradient Boosting": GradientBoostingClassifier(random_state=state, **params["Gradient Boosting"]),
    }


def train_model(name: str, estimator, feature_cols: List[str],
                train: pd.DataFrame) -> Pipeline:
    """Fit one model inside a full preprocessing pipeline."""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(feature_cols)),
        ("model", estimator),
    ])
    fit_params = {}
    if name in config.MODELS_NEEDING_SAMPLE_WEIGHT:
        fit_params["model__sample_weight"] = compute_sample_weight(
            "balanced", train[config.TARGET_COL])
    pipeline.fit(train[feature_cols], train[config.TARGET_COL], **fit_params)
    logger.info("Trained %s", name)
    return pipeline


def train_all_models(feature_cols: List[str], train: pd.DataFrame) -> Dict[str, Pipeline]:
    """Train every candidate model and return them keyed by name."""
    return {name: train_model(name, est, feature_cols, train)
            for name, est in get_estimators().items()}


def predict_probability(pipeline: Pipeline, df: pd.DataFrame, feature_cols: List[str]):
    """Probability of failure for each row."""
    return pipeline.predict_proba(df[feature_cols])[:, 1]


def save_model(pipeline: Pipeline, feature_cols: List[str], path: Path = None) -> Path:
    """Persist the fitted pipeline together with its feature list."""
    path = Path(path or config.MODEL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_cols": feature_cols}, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Path = None) -> Tuple[Pipeline, List[str]]:
    """Load a previously saved pipeline and its feature list.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(path or config.MODEL_FILE)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run main.py first.")
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["feature_cols"]
