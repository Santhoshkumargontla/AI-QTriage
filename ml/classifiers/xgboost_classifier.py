import hashlib
import json
import os

import numpy as np
import xgboost as xgb
import shap

from ml.fusion.feature_fusion import FEATURE_NAMES, N_FEATURES
from ml.models.canonical_paths import XGB_CANONICAL, XGB_METADATA, exists, resolve_existing

MODEL_ARTIFACT_MISSING = "MODEL_ARTIFACT_MISSING"
XGB_METADATA_PATH = XGB_METADATA
PROVENANCE_CLASSES = ("REAL", "SYNTHETIC", "MIXED")


def provenance_class_from_metadata(meta: dict | None) -> str:
    """Map training metadata to REAL / SYNTHETIC / MIXED. Never invent REAL."""
    meta = meta or {}
    explicit = meta.get("data_provenance_class")
    if explicit in PROVENANCE_CLASSES:
        return explicit
    metrics = meta.get("metrics") or {}
    try:
        clinical = int(metrics.get("genuinely_paired_clinical_samples") or 0)
    except (TypeError, ValueError):
        clinical = 0
    try:
        synthetic = int(metrics.get("synthetic_multimodal_fusion_samples") or 0)
    except (TypeError, ValueError):
        synthetic = 0
    raw = str(meta.get("data_provenance") or "").lower()
    if clinical > 0 and synthetic > 0:
        return "MIXED"
    if clinical > 0 and synthetic == 0:
        return "REAL"
    if "synthetic" in raw or synthetic > 0:
        return "SYNTHETIC"
    if "mixed" in raw:
        return "MIXED"
    if raw in ("real", "clinical"):
        return "REAL"
    return "SYNTHETIC"


def load_xgboost_metadata(path: str = XGB_METADATA_PATH) -> dict:
    located = resolve_existing(path)
    if not exists(located) and not os.path.exists(located):
        return {}
    try:
        with open(located, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class XGBoostClassifier:
    """Wrapper for training, predicting, and generating SHAP attributions with XGBoost."""

    def __init__(self, model_path: str = None):
        self.model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )
        self.feature_names = list(FEATURE_NAMES)
        self.is_trained = False
        self.status = "UNINITIALIZED"
        self.model_path = model_path
        self.metadata = load_xgboost_metadata()
        self.data_provenance = provenance_class_from_metadata(self.metadata)
        self.data_provenance_detail = self.metadata.get("data_provenance")
        self.artifact_sha256 = None

        if model_path:
            self._require_and_load(resolve_existing(model_path))

    def _artifact_error(self, path: str, extra: str = "") -> RuntimeError:
        suffix = f": {extra}" if extra else ""
        return RuntimeError(f"{MODEL_ARTIFACT_MISSING}: {path}{suffix}")

    def _require_and_load(self, model_path: str):
        if not os.path.exists(model_path):
            self.status = MODEL_ARTIFACT_MISSING
            self.is_trained = False
            raise self._artifact_error(model_path)
        try:
            self.load_model(model_path)
        except RuntimeError:
            raise
        except (OSError, ValueError, xgb.core.XGBoostError) as exc:
            self.status = MODEL_ARTIFACT_MISSING
            self.is_trained = False
            raise self._artifact_error(model_path, str(exc)) from exc

    def load_model(self, model_path: str):
        """Loads trained XGBoost booster from file. Never trains a replacement."""
        if not os.path.exists(model_path):
            self.status = MODEL_ARTIFACT_MISSING
            self.is_trained = False
            raise self._artifact_error(model_path)
        self.model.load_model(model_path)
        n_in = int(getattr(self.model, "n_features_in_", 0) or 0)
        if n_in == 0:
            try:
                n_in = int(self.model.get_booster().num_features())
            except (AttributeError, TypeError, ValueError):
                n_in = 0
        if n_in != N_FEATURES:
            self.status = MODEL_ARTIFACT_MISSING
            self.is_trained = False
            raise ValueError(
                f"XGBoost schema mismatch: expected {N_FEATURES} features, booster has {n_in}"
            )
        self.model_path = model_path
        self.is_trained = True
        self.status = "LOADED"
        self.artifact_sha256 = _sha256_file(model_path)
        print(f"XGBoost model successfully loaded from {model_path}")

    def save_model(self, model_path: str):
        """Saves current booster weights to file."""
        self.model.save_model(model_path)
        print(f"XGBoost model successfully saved to {model_path}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fits the classifier on fused feature matrices. Training scripts only — never called at analyze time."""
        self.model.fit(X, y)
        self.is_trained = True
        self.status = "TRAINED_IN_MEMORY"
        print("XGBoost model training completed successfully.")

    def _ensure_ready(self):
        if self.status == MODEL_ARTIFACT_MISSING:
            raise self._artifact_error(self.model_path or XGB_CANONICAL)
        if not self.is_trained:
            raise RuntimeError("XGBoost classifier is not trained yet.")

    def predict(self, fused_vector: np.ndarray) -> tuple:
        """
        Predicts category and returns label index and class probabilities.
        Returns:
            - pred_idx (int): 0 for LOW, 1 for MODERATE, 2 for HIGH
            - probabilities (list of float): list of length 3
        """
        self._ensure_ready()

        x_in = fused_vector.reshape(1, -1)
        expected_features = int(getattr(self.model, "n_features_in_", N_FEATURES) or N_FEATURES)
        if x_in.shape[1] != expected_features:
            raise ValueError(
                f"Feature vector shape mismatch: expected {expected_features} features, got {x_in.shape[1]}"
            )

        pred_idx = int(self.model.predict(x_in)[0])
        probs = self.model.predict_proba(x_in)[0].tolist()
        return pred_idx, probs

    def explain_prediction(self, fused_vector: np.ndarray, predicted_class: int) -> list:
        """
        Calculates local SHAP contribution values for the given vector.
        Returns a list of dictionaries with feature name and its SHAP contribution value.
        """
        self._ensure_ready()

        x_in = fused_vector.reshape(1, -1)
        expected_features = int(getattr(self.model, "n_features_in_", N_FEATURES) or N_FEATURES)
        if x_in.shape[1] != expected_features:
            raise ValueError(
                f"Feature vector shape mismatch: expected {expected_features} features, got {x_in.shape[1]}"
            )

        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(x_in)

        if isinstance(shap_vals, list):
            class_shap = shap_vals[predicted_class][0]
        elif len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, predicted_class]
        else:
            class_shap = shap_vals[0]

        local_explanations = []
        for i, name in enumerate(self.feature_names):
            local_explanations.append({
                "feature": name,
                "shap_value": float(class_shap[i]),
                "description": (
                    f"Feature contribution analysis (SHAP) for {name}. "
                    "This shows how much this variable contributed to the prediction."
                ),
            })

        local_explanations = sorted(local_explanations, key=lambda x: abs(x["shap_value"]), reverse=True)
        return local_explanations

    def get_global_importance(self) -> list:
        """
        Extracts global feature importances directly from the booster.
        Returns a sorted list of importances paired with feature names.
        """
        self._ensure_ready()

        importances = self.model.feature_importances_
        global_imp = []
        for i, name in enumerate(self.feature_names):
            global_imp.append({
                "feature": name,
                "importance": float(importances[i]),
            })

        global_imp = sorted(global_imp, key=lambda x: x["importance"], reverse=True)
        return global_imp

    def get_info(self) -> dict:
        return {
            "status": self.status,
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "canonical_path": XGB_CANONICAL.replace("\\", "/"),
            "n_features": N_FEATURES,
            "feature_names": list(self.feature_names),
            "data_provenance": self.data_provenance,
            "data_provenance_detail": self.data_provenance_detail,
            "artifact_sha256": self.artifact_sha256,
            "model_version": self.metadata.get("version"),
            "metadata_status": self.metadata.get("status"),
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
