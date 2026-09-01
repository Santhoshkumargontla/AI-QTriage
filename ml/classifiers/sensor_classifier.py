import os
import math
import joblib
import numpy as np
from typing import Dict, Any, List
from ml.models.canonical_paths import SENSOR_MODEL, SENSOR_SCALER, resolve_existing

FEATURE_MISSING = "FEATURE_MISSING"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

REQUIRED_FEATURES = [
    "peak_g_force",
    "peak_acceleration",
    "peak_jerk_gs",
    "accel_variance",
    "gyro_variance",
    "sma",
    "post_impact_stabilization_seconds",
]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except (TypeError, ValueError):
        return True


class SensorClassifier:
    """Motion classifier. Missing features are reported, never invented."""

    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model_path = os.path.normpath(resolve_existing(model_path or SENSOR_MODEL))
        self.scaler_path = os.path.normpath(resolve_existing(scaler_path or SENSOR_SCALER))
        self.classes = ["normal_activity", "fall", "impact"]
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.load_error = None

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            import xgboost as xgb
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
            except (OSError, ValueError, RuntimeError) as e:
                self.load_error = str(e)
                print(f"Warning: Failed to load sensor classifier: {e}")

    def predict_from_summary(self, sensor_summary: dict) -> Dict[str, Any]:
        if not self.is_trained or self.model is None or self.scaler is None:
            return {
                "predicted_motion_class": None,
                "confidence": None,
                "probabilities": None,
                "status": MODEL_UNAVAILABLE,
                "classifier_status": MODEL_UNAVAILABLE,
                "load_error": self.load_error,
                "missing_features": list(REQUIRED_FEATURES),
            }

        missing: List[str] = [key for key in REQUIRED_FEATURES if _is_missing(sensor_summary.get(key))]
        if missing:
            return {
                "predicted_motion_class": None,
                "confidence": None,
                "probabilities": None,
                "status": FEATURE_MISSING,
                "classifier_status": FEATURE_MISSING,
                "missing_features": missing,
            }

        peak_g = float(sensor_summary["peak_g_force"])
        peak_accel_ms2 = float(sensor_summary["peak_acceleration"])
        peak_jerk = float(sensor_summary["peak_jerk_gs"])
        accel_var = float(sensor_summary["accel_variance"])
        gyro_var = float(sensor_summary["gyro_variance"])
        sma = float(sensor_summary["sma"])
        stab_sec = float(sensor_summary["post_impact_stabilization_seconds"])
        impact_flag = 1.0 if peak_g > 3.0 else 0.0

        raw_vec = np.array([[
            peak_g, peak_accel_ms2, peak_jerk,
            accel_var, gyro_var, sma,
            stab_sec, impact_flag
        ]], dtype=np.float32)

        scaled_vec = self.scaler.transform(raw_vec)
        pred_idx = int(self.model.predict(scaled_vec)[0])
        probs = self.model.predict_proba(scaled_vec)[0].tolist()
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return {
            "predicted_motion_class": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict,
            "status": "classified",
            "classifier_status": "classified",
            "missing_features": [],
        }
