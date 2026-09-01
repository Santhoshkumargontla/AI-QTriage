import os
import json
import numpy as np
import pandas as pd

class SensorValidationError(Exception):
    """Raised when sensor CSV validation fails."""
    pass

def validate_sensor_csv(csv_path: str):
    """
    Validates that the sensor CSV/JSON meets schema and sampling frequency (50Hz) requirements.
    Raises SensorValidationError if invalid.
    """
    if not os.path.exists(csv_path):
        raise SensorValidationError("Sensor file does not exist.")
        
    try:
        if csv_path.lower().endswith(".json"):
            df = pd.read_json(csv_path)
        else:
            df = pd.read_csv(csv_path)
    except (ValueError, OSError, pd.errors.ParserError, json.JSONDecodeError) as e:
        raise SensorValidationError(f"Failed to parse file: {str(e)}")

    if df.empty:
        raise SensorValidationError("Sensor log is empty.")

    # 1. Column Validation (support both old short schema and new long schema)
    # Kinetic columns only. latitude/longitude/speed are optional and never imputed.
    short_cols = {"timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}
    long_accel_gyro = {
        "timestamp",
        "accelerometer_x", "accelerometer_y", "accelerometer_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z",
    }

    has_short = sum(1 for col in short_cols if col in df.columns)
    has_long = sum(1 for col in long_accel_gyro if col in df.columns)

    if has_long >= has_short:
        missing_cols = [col for col in long_accel_gyro if col not in df.columns]
        if missing_cols:
            raise SensorValidationError(f"Missing required sensor columns:\n" + "\n".join(missing_cols))

        df = df.rename(columns={
            "accelerometer_x": "accel_x",
            "accelerometer_y": "accel_y",
            "accelerometer_z": "accel_z",
            "gyroscope_x": "gyro_x",
            "gyroscope_y": "gyro_y",
            "gyroscope_z": "gyro_z"
        })
    else:
        missing_cols = short_cols - set(df.columns)
        if missing_cols:
            raise SensorValidationError(f"Missing required sensor columns: {', '.join(missing_cols)}")

    # 2. Check minimum length
    if len(df) < 10:
        raise SensorValidationError("Sensor log contains too few samples (minimum 10 required).")

    lux_available = "optical_lux" in df.columns
    if not lux_available:
        df["optical_lux"] = np.nan

    # 4. Frequency Validation (50Hz target -> ~20ms interval)
    timestamps = df["timestamp"].values
    time_diffs = np.diff(timestamps)
    
    if len(time_diffs) > 0:
        mean_diff = np.mean(time_diffs)
        # Calculate effective frequency: mean diff unit is assumed to be seconds.
        # If timestamps are in milliseconds, convert to seconds
        if mean_diff > 1.0:
            mean_diff = mean_diff / 1000.0
            
        freq = 1.0 / mean_diff if mean_diff > 0 else 0
        if freq < 40.0:  # Require at least 40Hz to allow for minor jitter
            raise SensorValidationError(
                f"Sampling rate is insufficient. Measured: {freq:.1f}Hz. Minimum required: 40Hz (50Hz target)."
            )
            
    return df

def process_sensor_data(csv_path: str) -> dict:
    """
    Processes sensor data to reconstruct the physical accident timeline and impact parameters.
    """
    df = validate_sensor_csv(csv_path)
    
    # Check timestamp unit (sec vs ms)
    timestamps = df["timestamp"].values
    is_ms = np.mean(np.diff(timestamps)) > 1.0
    if is_ms:
        time_sec = timestamps / 1000.0
    else:
        time_sec = timestamps
        
    # Relative time from start
    time_rel = time_sec - time_sec[0]

    accel_x = df["accel_x"].values
    accel_y = df["accel_y"].values
    accel_z = df["accel_z"].values
    lux = df["optical_lux"].values

    # Compute acceleration magnitude (assuming input is in m/s^2)
    a_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    # Peak index
    peak_idx = int(np.argmax(a_mag))
    peak_val = float(a_mag[peak_idx])
    peak_time_rel = float(time_rel[peak_idx])
    peak_g = peak_val / 9.80665

    # 1. Pre-impact velocity change (integrate acceleration up to peak)
    # Delta V = sum(a * dt) for the 1.0 second before peak
    dt = np.diff(time_rel)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.02)
    
    pre_peak_mask = (time_rel <= peak_time_rel) & (time_rel >= max(0.0, peak_time_rel - 1.0))
    delta_v = float(np.sum(a_mag[pre_peak_mask] * dt[pre_peak_mask]))

    # 2. Post-impact stabilization time
    # Time after peak it takes for acceleration to remain below 1.5g (14.7 m/s^2) for at least 0.5s
    threshold_stable = 1.5 * 9.80665
    stable_time_rel = None
    
    for i in range(peak_idx, len(time_rel)):
        post_window = a_mag[i:]
        if np.all(post_window < threshold_stable):
            stable_time_rel = float(time_rel[i])
            break
            
    stabilization_duration = (stable_time_rel - peak_time_rel) if stable_time_rel is not None else None

    # Gyro / jerk / SMA — computed from measured samples, never invented.
    gyro_var = None
    gyro_available = all(c in df.columns for c in ("gyro_x", "gyro_y", "gyro_z"))
    if gyro_available:
        gyro_mag = np.sqrt(df["gyro_x"].values ** 2 + df["gyro_y"].values ** 2 + df["gyro_z"].values ** 2)
        if np.isfinite(gyro_mag).any():
            gyro_var = float(np.nanvar(gyro_mag))
        else:
            gyro_available = False
    accel_variance = float(np.var(a_mag))
    sma = float(np.mean(np.abs(accel_x) + np.abs(accel_y) + np.abs(accel_z)))
    if len(a_mag) > 1:
        jerk = np.diff(a_mag) / np.maximum(dt[:-1] if len(dt) == len(a_mag) else dt[: len(a_mag) - 1], 1e-6)
        peak_jerk_gs = float(np.max(np.abs(jerk)) / 9.80665) if len(jerk) else 0.0
    else:
        peak_jerk_gs = 0.0

    lux_available = bool(np.isfinite(lux).any())
    lux_drop_detected = None
    if lux_available:
        impact_window_mask = (time_rel >= peak_time_rel - 0.2) & (time_rel <= peak_time_rel + 0.2)
        lux_window = lux[impact_window_mask]
        lux_drop_detected = False
        if len(lux_window) > 0 and np.isfinite(lux_window).any():
            min_lux_window = float(np.nanmin(lux_window))
            if min_lux_window < 10.0:
                lux_drop_detected = True

    # 4. Construct Chronological Timeline Events
    events = [
        {
            "time_offset_seconds": 0.0,
            "event_name": "Sensor Log Started",
            "description": "Baseline monitoring initialized."
        }
    ]
    
    if lux_drop_detected is True:
        # Locate exact index of the drop
        drop_idx = int(np.argmin(lux[impact_window_mask]))
        drop_time = float(time_rel[impact_window_mask][drop_idx])
        events.append({
            "time_offset_seconds": round(drop_time, 3),
            "event_name": "Ambient Light Drop",
            "description": "Sudden drop in light levels detected, indicating coverage or impact occlusion."
        })
        
    events.append({
        "time_offset_seconds": round(peak_time_rel, 3),
        "event_name": "Peak Impact Acceleration",
        "description": f"Maximum physical impact vector measured at {peak_g:.2f}g force."
    })
    
    if stable_time_rel is not None:
        events.append({
            "time_offset_seconds": round(stable_time_rel, 3),
            "event_name": "Physical Stabilization",
            "description": f"Kinetic motion stabilized to baseline level (<1.5g) after {stabilization_duration:.2f}s."
        })
    else:
        events.append({
            "time_offset_seconds": round(float(time_rel[-1]), 3),
            "event_name": "Monitoring Concluded",
            "description": "Sensor log completed without satisfying stabilization criteria."
        })

    # Sort events by timestamp
    events = sorted(events, key=lambda x: x["time_offset_seconds"])

    stab_out = round(stabilization_duration, 2) if stabilization_duration is not None else None
    recording_duration = float(time_rel[-1]) if len(time_rel) else None
    res_summary = {
        "timeline_label": "Timeline reconstruction. This shows the physical impact window and stabilization profile measured by sensors.",
        "peak_g_force": round(peak_g, 2),
        "peak_acceleration": round(peak_val, 2),
        "peak_jerk_gs": round(peak_jerk_gs, 4),
        "accel_variance": round(accel_variance, 4),
        "gyro_variance": round(gyro_var, 4) if gyro_var is not None else None,
        "sma": round(sma, 4),
        "peak_time_offset": round(peak_time_rel, 3),
        "pre_impact_delta_v": round(delta_v, 2),
        "post_impact_stabilization_seconds": stab_out,
        "optical_lux_drop": lux_drop_detected,
        "lux_feature_available": lux_available,
        "gyro_feature_available": gyro_available,
        "sample_count": int(len(df)),
        "recording_duration_seconds": round(recording_duration, 3) if recording_duration is not None else None,
        "events": events
    }

    from ml.classifiers.sensor_classifier import SensorClassifier
    clf = SensorClassifier()
    motion = clf.predict_from_summary(res_summary)
    res_summary["motion_classification"] = motion
    res_summary["predicted_motion_class"] = motion.get("predicted_motion_class")
    res_summary["motion_confidence"] = motion.get("confidence")
    res_summary["motion_probabilities"] = motion.get("probabilities")
    res_summary["classifier_status"] = motion.get("classifier_status") or motion.get("status")

    return res_summary
