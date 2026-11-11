"""Unit tests for cleaning, feature engineering, splitting and evaluation.

Run with:  python -m unittest discover -s tests   (or: pytest)
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import app_service, evaluation, feature_engineering, modeling, preprocessing, reporting  # noqa: E402


def make_frame(n: int = 40, machines=("M1", "M2")) -> pd.DataFrame:
    """Small, clean synthetic frame for testing."""
    rng = np.random.default_rng(0)
    rows = []
    for mid in machines:
        ts = pd.date_range("2026-01-01", periods=n, freq="4h")
        frame = pd.DataFrame({
            config.MACHINE_ID_COL: mid, config.TIMESTAMP_COL: ts,
            config.TEMPERATURE_COL: rng.normal(75, 3, n),
            config.PRESSURE_COL: rng.normal(145, 5, n),
            config.VIBRATION_COL: rng.normal(0.8, 0.08, n),
            config.RPM_COL: rng.normal(2850, 40, n),
            config.CURRENT_COL: rng.normal(12.4, 0.6, n),
            config.VOLTAGE_COL: rng.normal(415, 4, n),
            config.OPERATING_HOURS_COL: np.arange(n) * 4.0 + 1000,
            config.MAINTENANCE_COUNT_COL: 2.0,
            config.MACHINE_AGE_COL: 5.0,
            config.TARGET_COL: (np.arange(n) % 10 == 0).astype(int),
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


class TestPreprocessing(unittest.TestCase):
    def test_duplicates_removed(self):
        df = make_frame()
        cleaned, report = preprocessing.clean_data(pd.concat([df, df.head(5)]))
        self.assertEqual(len(cleaned), len(df))
        self.assertEqual(report["duplicates_after"], 0)

    def test_fahrenheit_converted(self):
        df = make_frame()
        df.loc[0, config.TEMPERATURE_COL] = 167.0  # ~75 C in Fahrenheit
        fixed, count = preprocessing.fix_temperature_units(df)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(fixed.loc[0, config.TEMPERATURE_COL], 75.0, places=1)

    def test_invalid_values_masked(self):
        df = make_frame()
        df.loc[3, config.VIBRATION_COL] = 99.0
        masked, counts = preprocessing.mask_invalid_values(df)
        self.assertTrue(np.isnan(masked.loc[3, config.VIBRATION_COL]))
        self.assertEqual(counts[config.VIBRATION_COL], 1)

    def test_gap_fill_uses_only_past_values(self):
        df = make_frame().sort_values([config.MACHINE_ID_COL, config.TIMESTAMP_COL])
        df.loc[df.index[0], config.TEMPERATURE_COL] = np.nan  # first reading of M1
        filled = preprocessing.fill_sensor_gaps(df)
        self.assertTrue(np.isnan(filled.iloc[0][config.TEMPERATURE_COL]))  # no back-fill


class TestFeatureEngineering(unittest.TestCase):
    def test_features_are_per_machine_and_causal(self):
        df = make_frame()
        full = feature_engineering.build_features(df)
        # Features of the first 20 readings must not change if later data is removed.
        truncated_input = df[df[config.TIMESTAMP_COL] < df[config.TIMESTAMP_COL].min() + pd.Timedelta(hours=80)]
        truncated = feature_engineering.build_features(truncated_input)
        key = [config.MACHINE_ID_COL, config.TIMESTAMP_COL]
        merged = truncated.merge(full, on=key, suffixes=("_t", "_f"))
        col = f"{config.TEMPERATURE_COL}_roll_mean"
        np.testing.assert_allclose(merged[f"{col}_t"], merged[f"{col}_f"])

    def test_deviation_zero_inside_normal_range(self):
        df = make_frame()
        df[config.TEMPERATURE_COL] = 75.0
        result = feature_engineering.add_deviation_features(df)
        self.assertTrue((result[f"{config.TEMPERATURE_COL}_deviation"] == 0).all())

    def test_feature_columns_exclude_target(self):
        feats = feature_engineering.build_features(make_frame())
        cols = feature_engineering.get_feature_columns(feats)
        for excluded in config.EXCLUDE_FROM_FEATURES:
            self.assertNotIn(excluded, cols)


class TestModelingAndEvaluation(unittest.TestCase):
    def test_chronological_split_has_no_overlap(self):
        feats = feature_engineering.build_features(make_frame(n=200))
        train, test = modeling.chronological_split(feats)
        self.assertLess(train[config.TIMESTAMP_COL].max(), test[config.TIMESTAMP_COL].min())

    def test_metrics_perfect_prediction(self):
        metrics = evaluation.compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
        self.assertEqual(metrics["Recall"], 1.0)
        self.assertEqual(metrics["FN"], 0)

    def test_risk_status_bands(self):
        probs = pd.Series([0.05, 0.45, 0.95])
        self.assertListEqual(reporting.assign_risk_status(probs).tolist(),
                             ["Healthy", "Warning", "High Risk"])


class TestAppService(unittest.TestCase):
    """Checks the upload validation used by the Streamlit front end."""

    def test_missing_column_is_reported(self):
        import io
        frame = make_frame(n=300).drop(columns=[config.VIBRATION_COL])
        with self.assertRaises(app_service.AnalysisError):
            app_service.read_uploaded_csv(io.StringIO(frame.to_csv(index=False)))

    def test_too_few_rows_rejected(self):
        import io
        frame = make_frame(n=20)
        with self.assertRaises(app_service.AnalysisError):
            app_service.read_uploaded_csv(io.StringIO(frame.to_csv(index=False)))

    def test_single_class_target_rejected(self):
        frame = make_frame(n=300)
        frame[config.TARGET_COL] = 0
        with self.assertRaises(app_service.AnalysisError):
            app_service.run_analysis(frame)


if __name__ == "__main__":
    unittest.main()
