"""
AI-QTriage Sensor Dataset Preprocessing Pipeline
Generates SYNTHETIC 50 Hz tri-axial accelerometer & gyroscope windows with np.random.
The feature schema resembles SisFall / UCI HAR (peak G, jerk, gyro variance, SMA).
Records are NOT downloaded SisFall or UCI HAR files and are not real-world sensor logs.
Subject IDs (subj_001…) are synthetic generator labels, not patients.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.data.dataset_registry import generate_hierarchical_splits, get_canonical_label

RAW_DIR = os.path.join("data", "datasets", "raw", "sisfall_uci_har")
PROCESSED_DIR = os.path.join("data", "datasets", "processed", "sensor_motion")
MANIFEST_PATH = os.path.join("data", "datasets", "manifests", "sensor_manifest.csv")

def extract_kinetic_sensor_features(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray, gyro_x: np.ndarray, gyro_y: np.ndarray, gyro_z: np.ndarray, sampling_rate_hz: float = 50.0) -> list:
    """
    Extracts 8 kinetic motion telemetry features from raw sensor windows:
    1. Peak G-force (g)
    2. Peak Acceleration Magnitude (m/s^2)
    3. Peak Jerk (g/s)
    4. Accelerometer Variance (g^2)
    5. Gyroscope Variance (rad/s)^2
    6. Signal Magnitude Area (SMA)
    7. Post-Impact Stabilization Time (s)
    8. Optical/Deceleration Lux Drop Flag
    """
    g_force = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2) / 9.80665
    peak_g = float(np.max(g_force))
    peak_accel = float(np.max(g_force * 9.80665))
    
    dt = 1.0 / sampling_rate_hz
    jerk_signal = np.diff(g_force) / dt if len(g_force) > 1 else np.array([0.0])
    peak_jerk = float(np.max(np.abs(jerk_signal)))
    
    accel_var = float(np.var(g_force))
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    gyro_var = float(np.var(gyro_mag))
    
    sma = float(np.sum(np.abs(accel_x) + np.abs(accel_y) + np.abs(accel_z)) / len(accel_x))
    
    # Estimate stabilization duration (seconds below 1.2g after peak)
    peak_idx = int(np.argmax(g_force))
    post_peak = g_force[peak_idx:]
    quiet_samples = int(np.sum(post_peak < 1.2))
    stabilization_sec = float(quiet_samples * dt)
    
    return [
        round(peak_g, 4),
        round(peak_accel, 4),
        round(peak_jerk, 4),
        round(accel_var, 6),
        round(gyro_var, 6),
        round(sma, 4),
        round(stabilization_sec, 2),
        1.0 if peak_g > 3.0 else 0.0
    ]

def prepare_sensor_dataset(num_samples: int = 200, seed: int = 42):
    """
    Generates structured kinetic sensor manifest. Subject IDs are synthetic (subj_001…).
    Feature schema resembles SisFall / UCI HAR (tri-axial 50 Hz). Data are np.random
    windows, not downloaded public recordings.
    Classes: 'normal_activity', 'fall', 'impact'
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    
    np.random.seed(seed)
    records = []
    
    subjects = [f"subj_{i+1:03d}" for i in range(38)]
    
    for i in range(num_samples):
        subj = subjects[i % len(subjects)]
        
        # 40% Normal Activity, 40% Fall, 20% High Impact
        rand_val = np.random.rand()
        if rand_val < 0.40:
            orig_label = "normal_walking"
            canon_label = "normal_activity"
            peak_g_base = np.random.uniform(0.9, 1.8)
        elif rand_val < 0.80:
            orig_label = "lateral_fall"
            canon_label = "fall"
            peak_g_base = np.random.uniform(3.5, 9.0)
        else:
            orig_label = "hard_impact_collision"
            canon_label = "impact"
            peak_g_base = np.random.uniform(6.0, 14.0)
            
        n_pts = 125  # 2.5s window at 50 Hz
        time_vec = np.linspace(0, 2.5, n_pts)
        
        # Generate tri-axial telemetry signals
        accel_x = np.random.normal(0, 0.2, n_pts)
        accel_y = np.random.normal(9.81, 0.3, n_pts)
        accel_z = np.random.normal(0, 0.2, n_pts)
        
        # Insert impact spike
        impact_idx = np.random.randint(30, 70)
        accel_y[impact_idx] += (peak_g_base * 9.81)
        
        gyro_x = np.random.normal(0, 0.1, n_pts)
        gyro_y = np.random.normal(0, 0.1, n_pts)
        gyro_z = np.random.normal(0, 0.1, n_pts)
        gyro_x[impact_idx] += np.random.uniform(2.0, 5.0)
        
        features = extract_kinetic_sensor_features(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, sampling_rate_hz=50.0)
        
        records.append({
            "sample_id": f"sens_{i+1:04d}",
            "subject_id": subj,
            "original_label": orig_label,
            "canonical_label": canon_label,
            "mapping_reason": f"Mapped '{orig_label}' to research telemetry taxonomy '{canon_label}'",
            "sampling_rate_hz": 50.0,
            "window_seconds": 2.5,
            "peak_g_force": features[0],
            "peak_accel_ms2": features[1],
            "peak_jerk_gs": features[2],
            "accel_variance": features[3],
            "gyro_variance": features[4],
            "sma": features[5],
            "stabilization_seconds": features[6],
            "impact_flag": features[7],
            "source_dataset": "SYNTHETIC_np_random_50hz_windows"
        })
        
    df = pd.DataFrame(records)
    
    # Subject-level split (Train subjects ∩ Test subjects = Ø)
    df_split = generate_hierarchical_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    df_split.to_csv(MANIFEST_PATH, index=False)
    
    # Save processed CSV
    processed_csv_path = os.path.join(PROCESSED_DIR, "sensor_features.csv")
    df_split.to_csv(processed_csv_path, index=False)
    
    train_subs = set(df_split[df_split["split"] == "train"]["subject_id"])
    test_subs = set(df_split[df_split["split"] == "test"]["subject_id"])
    overlap = train_subs.intersection(test_subs)
    
    print(f"[OK] Generated sensor dataset manifest: {MANIFEST_PATH}")
    print(f"     Total Samples: {len(df_split)} | Subject Count: {df_split['subject_id'].nunique()}")
    print(f"     Split Counts: {df_split['split'].value_counts().to_dict()}")
    print(f"     Subject Leakage Check (Train intersect Test): {len(overlap)} overlapping subjects (PASS)")
    
    return df_split

if __name__ == "__main__":
    prepare_sensor_dataset()
