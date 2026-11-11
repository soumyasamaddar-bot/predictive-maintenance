"""Central configuration for the Predictive Maintenance project.

Every path, column name, threshold and model hyper-parameter lives here so that
no value is hard-coded elsewhere in the code base. To change behaviour, edit
this file (or override via command-line arguments in ``main.py``).
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
LOGS_DIR = OUTPUT_DIR / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "sensor_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "sensor_data_clean.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "sensor_data_features.csv"
MODEL_FILE = MODELS_DIR / "random_forest_model.joblib"
LOG_FILE = LOGS_DIR / "pipeline.log"

ALL_OUTPUT_DIRS = [
    PROCESSED_DATA_DIR, RAW_DATA_DIR, FIGURES_DIR, MODELS_DIR,
    REPORTS_DIR, DASHBOARD_DIR, LOGS_DIR,
]

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
MACHINE_ID_COL = "Machine_ID"
TIMESTAMP_COL = "Timestamp"
TARGET_COL = "Failure"

TEMPERATURE_COL = "Temperature"
PRESSURE_COL = "Pressure"
VIBRATION_COL = "Vibration"
RPM_COL = "RPM"
CURRENT_COL = "Current"
VOLTAGE_COL = "Voltage"
OPERATING_HOURS_COL = "Operating_Hours"
MAINTENANCE_COUNT_COL = "Maintenance_Count"
MACHINE_AGE_COL = "Machine_Age"

SENSOR_COLS = [
    TEMPERATURE_COL, PRESSURE_COL, VIBRATION_COL,
    RPM_COL, CURRENT_COL, VOLTAGE_COL,
]
NUMERIC_COLS = SENSOR_COLS + [
    OPERATING_HOURS_COL, MAINTENANCE_COUNT_COL, MACHINE_AGE_COL,
]
REQUIRED_COLS = [MACHINE_ID_COL, TIMESTAMP_COL] + NUMERIC_COLS + [TARGET_COL]

# Business meaning of each field (used for the data dictionary report).
DATA_DICTIONARY = {
    MACHINE_ID_COL: ("Categorical", "Unique equipment identifier"),
    TIMESTAMP_COL: ("Date/Time", "Time of the sensor reading"),
    TEMPERATURE_COL: ("Numeric (deg C)", "Thermal condition of the machine"),
    PRESSURE_COL: ("Numeric (psi)", "Operating pressure"),
    VIBRATION_COL: ("Numeric (mm/s)", "Mechanical condition (bearing/imbalance)"),
    RPM_COL: ("Numeric (rpm)", "Operating speed"),
    CURRENT_COL: ("Numeric (A)", "Electrical load"),
    VOLTAGE_COL: ("Numeric (V)", "Electrical supply condition"),
    OPERATING_HOURS_COL: ("Numeric (hours)", "Cumulative equipment usage"),
    MAINTENANCE_COUNT_COL: ("Numeric (count)", "Maintenance events so far"),
    MACHINE_AGE_COL: ("Numeric (years)", "Asset age"),
    TARGET_COL: ("Binary 0/1", "1 = failure occurs within the prediction horizon"),
}

# --------------------------------------------------------------------------- #
# Synthetic data generation (used only when no raw data file is supplied)
# --------------------------------------------------------------------------- #
SYNTH_N_MACHINES = 20
SYNTH_MACHINE_ID_START = 101
SYNTH_MACHINE_ID_PREFIX = "M"
SYNTH_START_DATE = "2026-01-01"
SYNTH_N_DAYS = 240
SYNTH_READING_INTERVAL_HOURS = 4
SYNTH_FAILURE_HORIZON_READINGS = 6          # label = failure within next 24 h
SYNTH_MEAN_TIME_BETWEEN_FAILURES_READINGS = 260
SYNTH_DEGRADATION_WINDOW_READINGS = 60      # sensors drift this long before failure
SYNTH_MISSING_RATE = 0.012
SYNTH_DUPLICATE_RATE = 0.004
SYNTH_OUTLIER_RATE = 0.002
SYNTH_FAHRENHEIT_RATE = 0.004               # rows logged in deg F by mistake
SYNTH_RANDOM_SEED = 42

# Healthy baseline for each sensor: (mean, std)
SYNTH_BASELINES = {
    TEMPERATURE_COL: (75.0, 3.0),
    PRESSURE_COL: (145.0, 5.0),
    VIBRATION_COL: (0.80, 0.08),
    RPM_COL: (2850.0, 40.0),
    CURRENT_COL: (12.4, 0.6),
    VOLTAGE_COL: (415.0, 4.0),
}
# Extra drift added at the moment of failure (grows during degradation window)
SYNTH_DEGRADATION_DRIFT = {
    TEMPERATURE_COL: 22.0,
    PRESSURE_COL: 14.0,
    VIBRATION_COL: 1.10,
    RPM_COL: -120.0,
    CURRENT_COL: 3.0,
    VOLTAGE_COL: -6.0,
}
SYNTH_MACHINE_AGE_RANGE = (1, 12)
SYNTH_INITIAL_HOURS_RANGE = (200, 20000)

# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
# Plausible physical range per sensor. Values outside are treated as invalid.
VALID_RANGES = {
    TEMPERATURE_COL: (-20.0, 200.0),
    PRESSURE_COL: (0.0, 400.0),
    VIBRATION_COL: (0.0, 15.0),
    RPM_COL: (0.0, 6000.0),
    CURRENT_COL: (0.0, 60.0),
    VOLTAGE_COL: (300.0, 500.0),
    OPERATING_HOURS_COL: (0.0, 200000.0),
    MAINTENANCE_COUNT_COL: (0.0, 1000.0),
    MACHINE_AGE_COL: (0.0, 60.0),
}
# Temperatures above this are assumed to be recorded in deg F and converted.
FAHRENHEIT_DETECTION_THRESHOLD = 150.0
FORWARD_FILL_LIMIT = 3          # max consecutive readings to forward-fill
OUTLIER_IQR_MULTIPLIER = 3.0    # used for reporting sensor outliers in EDA

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
ROLLING_WINDOW = 6              # readings (24 h at 4-hourly sampling)
ROLLING_FEATURE_COLS = [TEMPERATURE_COL, VIBRATION_COL, PRESSURE_COL]
LAG_FEATURE_COLS = [TEMPERATURE_COL, VIBRATION_COL]
LAG_PERIODS = [1, 3]

# "Normal operating range" used for the deviation features.
NORMAL_RANGES = {
    TEMPERATURE_COL: (65.0, 85.0),
    PRESSURE_COL: (130.0, 160.0),
    VIBRATION_COL: (0.5, 1.1),
    RPM_COL: (2750.0, 2950.0),
    CURRENT_COL: (10.5, 14.5),
    VOLTAGE_COL: (400.0, 430.0),
}

OPERATING_HOURS_BINS = [0, 5000, 10000, 15000, float("inf")]
OPERATING_HOURS_LABELS = ["0-5k", "5k-10k", "10k-15k", "15k+"]
MACHINE_AGE_BINS = [0, 3, 7, float("inf")]
MACHINE_AGE_LABELS = ["New", "Mid-life", "Old"]

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.25                # chronological split: last 25% of time = test
CATEGORICAL_FEATURES = ["Operating_Hours_Band", "Machine_Age_Group"]
EXCLUDE_FROM_FEATURES = [MACHINE_ID_COL, TIMESTAMP_COL, TARGET_COL]
PRIMARY_MODEL_NAME = "Random Forest"
DECISION_THRESHOLD = 0.5
# Training rows this close to the split point are dropped (embargo) because
# their labels look into the test period (the label horizon).
SPLIT_EMBARGO_HOURS = 24
# Models whose estimator needs explicit sample weights to handle imbalance.
MODELS_NEEDING_SAMPLE_WEIGHT = ["Gradient Boosting"]

MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 2000, "class_weight": "balanced", "C": 1.0,
    },
    "Decision Tree": {
        "max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced",
    },
    "Random Forest": {
        "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
        "class_weight": "balanced_subsample", "n_jobs": -1,
    },
    "Gradient Boosting": {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
        "subsample": 0.8,
    },
}

ISOLATION_FOREST_PARAMS = {"n_estimators": 200, "contamination": 0.02}
TOP_N_FEATURES = 15

# --------------------------------------------------------------------------- #
# Risk scoring & maintenance recommendations
# --------------------------------------------------------------------------- #
RISK_THRESHOLDS = {"Warning": 0.30, "High Risk": 0.60}
RISK_LABELS = ["Healthy", "Warning", "High Risk"]
TOP_N_RISK_MACHINES = 10
RECOMMENDATIONS = {
    "High Risk": "Schedule preventive maintenance immediately (within 24 h).",
    "Warning": "Plan an inspection within the next maintenance window.",
    "Healthy": "Continue routine monitoring.",
}

# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
FIGURE_DPI = 130
FIGURE_FORMAT = "png"
SEABORN_STYLE = "whitegrid"
DEFAULT_FIGSIZE = (10, 6)
SENSOR_TREND_MACHINES = 3       # machines shown in sensor trend chart

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
