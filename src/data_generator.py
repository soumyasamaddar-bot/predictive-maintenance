"""Synthetic IoT sensor data generator.

The project scope allows a "realistic synthetic IoT dataset". This module
produces one when no real dataset is placed in ``data/raw/``. It simulates a
fleet of machines whose sensors drift away from healthy baselines in the
period before each failure, and then injects realistic data-quality problems
(missing values, duplicates, outliers, unit errors) so the cleaning stage has
real work to do.

To use a real dataset instead, simply place a CSV with the columns listed in
``config.REQUIRED_COLS`` at ``config.RAW_DATA_FILE``.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _machine_ids() -> List[str]:
    """Build the list of machine identifiers, e.g. ``M101 ... M120``."""
    start = config.SYNTH_MACHINE_ID_START
    return [
        f"{config.SYNTH_MACHINE_ID_PREFIX}{start + i}"
        for i in range(config.SYNTH_N_MACHINES)
    ]


def _simulate_failure_steps(n_readings: int, rng: np.random.Generator,
                            proneness: float) -> List[int]:
    """Draw the reading indices at which a machine fails.

    Args:
        n_readings: Number of readings simulated for the machine.
        rng: Random generator.
        proneness: Multiplier (>1 = fails more often) for this machine.

    Returns:
        Sorted list of failure indices, separated by at least one
        degradation window.
    """
    mean_gap = config.SYNTH_MEAN_TIME_BETWEEN_FAILURES_READINGS / proneness
    min_gap = config.SYNTH_DEGRADATION_WINDOW_READINGS + 10
    failures, position = [], 0
    while True:
        position += max(min_gap, int(rng.exponential(mean_gap)))
        if position >= n_readings:
            return failures
        failures.append(position)


def _simulate_machine(machine_id: str, timestamps: pd.DatetimeIndex,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the full reading history of a single machine."""
    n = len(timestamps)
    proneness = rng.uniform(0.6, 2.2)
    age = int(rng.integers(*config.SYNTH_MACHINE_AGE_RANGE))
    hours = float(rng.uniform(*config.SYNTH_INITIAL_HOURS_RANGE))
    failure_steps = _simulate_failure_steps(n, rng, proneness * (1 + age / 20))

    # health_stress in [0, 1]: 0 = healthy, 1 = failing right now.
    stress = np.zeros(n)
    window = config.SYNTH_DEGRADATION_WINDOW_READINGS
    for step in failure_steps:
        start = max(0, step - window)
        stress[start:step + 1] = np.linspace(0, 1, step + 1 - start) ** 1.5

    data: Dict[str, np.ndarray] = {}
    for col, (mean, std) in config.SYNTH_BASELINES.items():
        drift = config.SYNTH_DEGRADATION_DRIFT[col] * stress
        data[col] = rng.normal(mean, std, n) + drift

    hours_per_reading = config.SYNTH_READING_INTERVAL_HOURS * rng.uniform(0.7, 1.0)
    operating_hours = hours + np.arange(n) * hours_per_reading

    # Maintenance count increases after each failure (repair event).
    maintenance = np.full(n, float(rng.integers(0, 4)))
    for step in failure_steps:
        maintenance[step + 1:] += 1

    # Label: failure occurs within the next H readings.
    label = np.zeros(n, dtype=int)
    horizon = config.SYNTH_FAILURE_HORIZON_READINGS
    for step in failure_steps:
        label[max(0, step - horizon + 1):step + 1] = 1

    frame = pd.DataFrame(data)
    frame.insert(0, config.TIMESTAMP_COL, timestamps)
    frame.insert(0, config.MACHINE_ID_COL, machine_id)
    frame[config.OPERATING_HOURS_COL] = operating_hours.round(1)
    frame[config.MAINTENANCE_COUNT_COL] = maintenance
    frame[config.MACHINE_AGE_COL] = age + operating_hours / 8760 - hours / 8760
    frame[config.TARGET_COL] = label
    return frame


def _inject_data_quality_issues(df: pd.DataFrame,
                                rng: np.random.Generator) -> pd.DataFrame:
    """Inject missing values, outliers, unit errors and duplicates."""
    df = df.copy()
    n = len(df)

    for col in config.SENSOR_COLS:
        mask = rng.random(n) < config.SYNTH_MISSING_RATE
        df.loc[mask, col] = np.nan

    # Sensor spikes (outliers / invalid values)
    spike_idx = rng.choice(n, int(n * config.SYNTH_OUTLIER_RATE), replace=False)
    df.loc[spike_idx, config.VIBRATION_COL] = rng.uniform(20, 40, len(spike_idx))

    # Temperatures logged in Fahrenheit by mistake (inconsistent units)
    f_idx = rng.choice(n, int(n * config.SYNTH_FAHRENHEIT_RATE), replace=False)
    df.loc[f_idx, config.TEMPERATURE_COL] = (
        df.loc[f_idx, config.TEMPERATURE_COL] * 9 / 5 + 32
    )

    # Exact duplicate rows
    dup = df.sample(frac=config.SYNTH_DUPLICATE_RATE, random_state=config.SYNTH_RANDOM_SEED)
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=config.SYNTH_RANDOM_SEED).reset_index(drop=True)


def generate_sensor_data() -> pd.DataFrame:
    """Generate the full synthetic IoT dataset.

    Returns:
        Raw (dirty) sensor readings for all machines.
    """
    rng = np.random.default_rng(config.SYNTH_RANDOM_SEED)
    n_readings = int(config.SYNTH_N_DAYS * 24 / config.SYNTH_READING_INTERVAL_HOURS)
    timestamps = pd.date_range(
        config.SYNTH_START_DATE, periods=n_readings,
        freq=f"{config.SYNTH_READING_INTERVAL_HOURS}h",
    )
    frames = [_simulate_machine(mid, timestamps, rng) for mid in _machine_ids()]
    raw = _inject_data_quality_issues(pd.concat(frames, ignore_index=True), rng)
    logger.info("Generated synthetic dataset: %d rows, %d machines",
                len(raw), config.SYNTH_N_MACHINES)
    return raw
