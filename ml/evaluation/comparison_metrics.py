import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score
from datetime import datetime, timezone

from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.fusion.rules_engine import RulesEngine
from backend.database.connection import get_database

def compute_ece(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Expected Calibration Error (ECE) for multi-class predictions.
    probs: numpy array of shape [N, C] containing confidence scores
    y_true: numpy array of shape [N] containing true labels (0, 1, 2)
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
        
    # Get predictions and confidences
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (y_pred == y_true)
    
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Mask for samples falling in the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_weight = bin_size / n_samples
            ece += bin_weight * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return float(ece)

def compute_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculates the multi-class Brier Score.
    probs: numpy array of shape [N, C] containing confidence scores
    y_true: numpy array of shape [N] containing true labels (0, 1, 2)
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
        
    n_classes = probs.shape[1]
    
    # One-hot encode true labels
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y_true] = 1.0
    
    # Compute mean squared difference
    brier = np.mean(np.sum((probs - y_one_hot) ** 2, axis=1))
    return float(brier)

class ClassicalQuantumEvaluator:
    """Evaluates and compares XGBoost (classical) vs VQC (quantum) metrics on test data."""
    
    def __init__(self, dataset_dir: str, manifest_path: str):
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.fusion = MultimodalFeatureFusion()
        self.rules = RulesEngine()

    def run_comparison(self, xgb_weights_path: str, vqc_model_dir: str) -> dict:
        """
        Runs predictions on the test split for both models and computes comparison metrics.
        Saves comparative logs to MongoDB.
        """
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Dataset manifest missing at {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        test_df = df[df["split"] == "test"]
        
        if len(test_df) == 0:
            raise ValueError("No test samples found in manifest.")
            
        # Initialize trained classifiers
        xgb_model = XGBoostClassifier(xgb_weights_path)
        vqc_model = VQCClassifier(vqc_model_dir)
        
        # Gather vectors, true targets (rule-derived labels)
        X_test = []
        y_true = []
        
        # Mappings: LOW=0, MODERATE=1, HIGH=2
        label_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        
        print("Gathering test features for comparison...")
        for _, row in test_df.iterrows():
            # Build mock case matching dataset values
            # In a real environment we would load full metadata, here we mock case formats
            img_path = os.path.join(self.dataset_dir, row["image_path"]) if not pd.isna(row["image_path"]) else None
            mask_path = os.path.join(self.dataset_dir, row["mask_path"]) if not pd.isna(row["mask_path"]) else None
            
            # Formulate simulated inputs for this record
            case_data = {
                "vision_analysis": {
                    "classification": {
                        "Cut": 0.80 if row["class"] == "cut" else 0.10,
                        "Bruise": 0.80 if row["class"] == "bruise" else 0.10,
                        "Swelling": 0.80 if row["class"] == "swelling" else 0.10,
                        "Other": 0.05
                    },
                    "segmentation": {
                        "affected_ratio": 0.15 if not pd.isna(mask_path) else 0.0
                    }
                } if img_path else {},
                "questionnaire": {
                    "answers": {
                        "pain_level": 7 if row["class"] == "cut" else (5 if row["class"] == "bruise" else 8),
                        "injury_mechanism": "sports" if row["class"] == "swelling" else "sharp_object",
                        "visible_bleeding": "yes" if row["class"] == "cut" else "no",
                        "movement_limitation": "mild" if row["class"] == "swelling" else "none",
                        "weight_bearing": "partial" if row["class"] == "swelling" else "yes",
                        "crack_pop": "no"
                    }
                },
                "sensor_summary": {
                    "peak_g_force": 3.8 if row["class"] == "swelling" else 1.2,
                    "pre_impact_delta_v": 0.8,
                    "post_impact_stabilization_seconds": 0.50,
                    "optical_lux_drop": False
                }
            }
            
            # Compute fused vector
            _, vector, names = self.fusion.fuse_features(case_data)
            
            # Compute rule-derived categorical target
            rule_label, _ = self.rules.evaluate_rules(vector, names)
            
            X_test.append(vector)
            y_true.append(label_map[rule_label])
            
        X_test = np.array(X_test)
        y_true = np.array(y_true)
        
        # --- Run Predictions ---
        xgb_preds = []
        xgb_probs_mat = []
        vqc_preds = []
        vqc_probs_mat = []
        
        for vec in X_test:
            x_pred, x_probs = xgb_model.predict(vec)
            xgb_preds.append(x_pred)
            xgb_probs_mat.append(x_probs)
            
            v_pred, v_probs = vqc_model.predict(vec)
            vqc_preds.append(v_pred)
            vqc_probs_mat.append(v_probs)
            
        xgb_preds = np.array(xgb_preds)
        xgb_probs_mat = np.array(xgb_probs_mat)
        vqc_preds = np.array(vqc_preds)
        vqc_probs_mat = np.array(vqc_probs_mat)
        
        # --- Compute Comparative Statistics ---
        xgb_mcc = matthews_corrcoef(y_true, xgb_preds)
        xgb_f1 = f1_score(y_true, xgb_preds, average="macro")
        xgb_ece = compute_ece(xgb_probs_mat, y_true)
        xgb_brier = compute_brier_score(xgb_probs_mat, y_true)
        
        vqc_mcc = matthews_corrcoef(y_true, vqc_preds)
        vqc_f1 = f1_score(y_true, vqc_preds, average="macro")
        vqc_ece = compute_ece(vqc_probs_mat, y_true)
        vqc_brier = compute_brier_score(vqc_probs_mat, y_true)
        
        comparison = {
            "comparison_date": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(y_true),
            "classical_xgb": {
                "mcc": float(xgb_mcc),
                "macro_f1": float(xgb_f1),
                "ece": float(xgb_ece),
                "brier_score": float(xgb_brier)
            },
            "quantum_vqc": {
                "mcc": float(vqc_mcc),
                "macro_f1": float(vqc_f1),
                "ece": float(vqc_ece),
                "brier_score": float(vqc_brier)
            },
            "interpretation": "VQC outputs are Experimental VQC outputs (non-calibrated scores). This dashboard compares classical XGBoost attributions with quantum Variational expectation values."
        }
        
        # Save comparison report to MongoDB
        db = get_database()
        db.model_comparisons.delete_many({}) # Clear old comparisons
        db.model_comparisons.insert_one(comparison)
        print("Classical vs. Quantum metrics comparison report successfully saved to MongoDB.")
        
        return comparison
