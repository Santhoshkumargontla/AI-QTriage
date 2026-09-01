"""Supported sensor simulation scenarios.

Canonical names only. There are no undocumented aliases.
"""

from typing import Optional

import numpy as np
import pandas as pd

SUPPORTED_SCENARIOS = (
    "football_fall",
    "sudden_fall",
    "sudden_impact",
    "normal_movement",
)

# Documented aliases only. Empty: callers must use canonical names above.
SCENARIO_ALIASES: dict[str, str] = {}


def resolve_scenario(name: str) -> Optional[str]:
    key = (name or "").strip().lower()
    if key in SUPPORTED_SCENARIOS:
        return key
    return SCENARIO_ALIASES.get(key)


def generate_scenario_dataframe(scenario: str, num_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Build a 50 Hz synthetic log. Kinetics come from the scenario, not from imputed defaults."""
    resolved = resolve_scenario(scenario)
    if resolved is None:
        raise ValueError(
            "Invalid scenario type. Use: " + ", ".join(SUPPORTED_SCENARIOS)
        )
    rng = np.random.RandomState(seed)
    timestamps = [round(i * 0.02, 3) for i in range(num_samples)]
    accel_x = rng.normal(0, 0.1, num_samples)
    accel_y = rng.normal(9.81, 0.1, num_samples)
    accel_z = rng.normal(0, 0.1, num_samples)
    gyro_x = rng.normal(0, 0.02, num_samples)
    gyro_y = rng.normal(0, 0.02, num_samples)
    gyro_z = rng.normal(0, 0.02, num_samples)
    speed = rng.normal(1.2, 0.1, num_samples)

    if resolved == "football_fall":
        accel_y[50] = 42.5
        accel_x[50] = 12.0
        accel_z[50] = 10.0
        gyro_y[50] = 3.5
        speed = np.array([5.4 if t < 1.0 else 0.0 for t in timestamps], dtype=float)
    elif resolved == "sudden_fall":
        for idx in range(40, 50):
            accel_x[idx] = rng.normal(0, 0.02)
            accel_y[idx] = rng.normal(0, 0.02)
            accel_z[idx] = rng.normal(0, 0.02)
        accel_y[50] = 38.0
        accel_x[50] = 8.0
        accel_z[50] = 8.0
        gyro_z[50] = 4.2
        speed = np.array([1.5 if t < 1.0 else 0.0 for t in timestamps], dtype=float)
    elif resolved == "sudden_impact":
        accel_x[50] = 49.0
        accel_y[50] = 20.0
        accel_z[50] = 15.0
        gyro_x[50] = 2.8
        speed = np.full(num_samples, 8.5, dtype=float)
    elif resolved == "normal_movement":
        pass

    return pd.DataFrame({
        "timestamp": timestamps,
        "accelerometer_x": accel_x,
        "accelerometer_y": accel_y,
        "accelerometer_z": accel_z,
        "gyroscope_x": gyro_x,
        "gyroscope_y": gyro_y,
        "gyroscope_z": gyro_z,
        "latitude": [37.7749 for _ in range(num_samples)],
        "longitude": [-122.4194 for _ in range(num_samples)],
        "speed": speed,
    })
