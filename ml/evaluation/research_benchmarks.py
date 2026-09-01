import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.fusion.rules_engine import RulesEngine
from backend.database.connection import get_database

class ResearchBenchmarkRunner:
    """Calculates ablation metrics, robustness perturbations, and uncertainty coverage for models."""
    
    def __init__(self, dataset_dir: str, manifest_path: str):
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.fusion = MultimodalFeatureFusion()
        self.rules = RulesEngine()

    def _load_test_vectors(self) -> tuple:
        """Loads and fuses test vectors from the manifest dataset."""
        df = pd.read_csv(self.manifest_path)
        test_df = df[df["split"] == "test"]
        
        if len(test_df) == 0:
            raise ValueError("No test samples found in manifest.")
            
        vectors = []
        labels = []
        label_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        
        for _, row in test_df.iterrows():
            img_path = os.path.join(self.dataset_dir, row["image_path"]) if not pd.isna(row["image_path"]) else None
            mask_path = os.path.join(self.dataset_dir, row["mask_path"]) if not pd.isna(row["mask_path"]) else None
            
            # Simulated data dictionary
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
            
            _, vector, names = self.fusion.fuse_features(case_data)
            rule_label, _ = self.rules.evaluate_rules(vector, names)
            
            vectors.append(vector)
            labels.append(label_map[rule_label])
            
        return np.array(vectors), np.array(labels), names

    def run_all_benchmarks(self, xgb_weights_path: str, vqc_model_dir: str) -> dict:
        """
        Runs ablation, perturbations, and coverage analyses, committing results to MongoDB.
        """
        X_test, y_true, names = self._load_test_vectors()
        
        xgb = XGBoostClassifier(xgb_weights_path)
        vqc = VQCClassifier(vqc_model_dir)
        
        # --- 1. ABLATION STUDIES ---
        ablation_results = []
        configurations = ["full", "no_vision", "no_questionnaire", "no_sensor"]
        
        for config in configurations:
            # Clone test vectors
            X_ablabted = X_test.copy()
            
            for vec in X_ablabted:
                if config == "no_vision":
                    vec[names.index("vision_present")] = 0.0
                    # Load fallbacks
                    vec[names.index("prob_cut")] = 0.25
                    vec[names.index("prob_bruise")] = 0.25
                    vec[names.index("prob_swelling")] = 0.25
                    vec[names.index("prob_other")] = 0.25
                    vec[names.index("affected_ratio")] = 0.0
                elif config == "no_questionnaire":
                    vec[names.index("questionnaire_present")] = 0.0
                    vec[names.index("pain_level")] = 0.50
                    vec[names.index("mech_fall")] = 0.0
                    vec[names.index("mech_impact")] = 0.0
                    vec[names.index("mech_sports")] = 0.0
                    vec[names.index("mech_sharp")] = 0.0
                    vec[names.index("mech_other")] = 1.0
                    vec[names.index("direct_impact")] = 0.0
                    vec[names.index("visible_bleeding")] = 0.0
                    vec[names.index("movement_limitation")] = 0.0
                    vec[names.index("weight_bearing")] = 0.0
                    vec[names.index("crack_pop")] = 0.0
                elif config == "no_sensor":
                    vec[names.index("sensor_present")] = 0.0
                    vec[names.index("peak_g_force")] = 1.0
                    vec[names.index("delta_v")] = 0.0
                    vec[names.index("stabilization_time")] = 0.0
                    vec[names.index("lux_drop")] = 0.0
            
            # Predict
            xgb_preds = [xgb.predict(v)[0] for v in X_ablabted]
            vqc_preds = [vqc.predict(v)[0] for v in X_ablabted]
            
            ablation_results.append({
                "configuration": config,
                "xgb_accuracy": float(accuracy_score(y_true, xgb_preds)),
                "xgb_f1": float(f1_score(y_true, xgb_preds, average="macro", zero_division=0)),
                "xgb_mcc": float(matthews_corrcoef(y_true, xgb_preds)),
                "vqc_accuracy": float(accuracy_score(y_true, vqc_preds)),
                "vqc_f1": float(f1_score(y_true, vqc_preds, average="macro", zero_division=0)),
                "vqc_mcc": float(matthews_corrcoef(y_true, vqc_preds))
            })

        # --- 2. ROBUSTNESS PERTURBATIONS ---
        perturbation_results = []
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
        
        for sigma in noise_levels:
            # Clone test vectors and inject noise into continuous features
            X_noise = X_test.copy()
            if sigma > 0.0:
                # Target pain level, peak g-force, and delta_v
                noise_pain = np.random.normal(0, sigma, len(X_noise))
                noise_g = np.random.normal(0, sigma * 5.0, len(X_noise)) # scale g-force noise
                
                for idx, vec in enumerate(X_noise):
                    vec[names.index("pain_level")] = np.clip(vec[names.index("pain_level")] + noise_pain[idx], 0.0, 1.0)
                    vec[names.index("peak_g_force")] = max(1.0, vec[names.index("peak_g_force")] + noise_g[idx])
            
            xgb_preds = [xgb.predict(v)[0] for v in X_noise]
            vqc_preds = [vqc.predict(v)[0] for v in X_noise]
            
            perturbation_results.append({
                "noise_level": sigma,
                "xgb_accuracy": float(accuracy_score(y_true, xgb_preds)),
                "xgb_f1": float(f1_score(y_true, xgb_preds, average="macro", zero_division=0)),
                "vqc_accuracy": float(accuracy_score(y_true, vqc_preds)),
                "vqc_f1": float(f1_score(y_true, vqc_preds, average="macro", zero_division=0))
            })

        # --- 3. UNCERTAINTY & COVERAGE ANALYSIS ---
        coverage_threshold = 0.70
        xgb_cov_probs = []
        vqc_cov_probs = []
        
        # Gather predicted score vectors
        for vec in X_test:
            _, x_p = xgb.predict(vec)
            xgb_cov_probs.append(x_p)
            
            _, v_p = vqc.predict(vec)
            vqc_cov_probs.append(v_p)
            
        xgb_cov_probs = np.array(xgb_cov_probs)
        vqc_cov_probs = np.array(vqc_cov_probs)
        
        # Helper stats computation
        def get_coverage_stats(probs, targets):
            confidences = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1)
            
            covered_mask = confidences >= coverage_threshold
            n_covered = np.sum(covered_mask)
            total = len(targets)
            
            coverage_ratio = n_covered / total if total > 0 else 0.0
            
            # Shannon Entropy calculation: H = -sum(p * log2(p))
            # Handle log of zero
            entropy = -np.sum(probs * np.log2(probs + 1e-15), axis=1)
            mean_entropy = float(np.mean(entropy))
            
            acc_in = float(accuracy_score(targets[covered_mask], preds[covered_mask])) if n_covered > 0 else 1.0
            acc_out = float(accuracy_score(targets[~covered_mask], preds[~covered_mask])) if (total - n_covered) > 0 else 0.0
            
            return {
                "coverage_ratio": float(coverage_ratio),
                "accuracy_inside_coverage": acc_in,
                "accuracy_outside_coverage": acc_out,
                "mean_entropy": mean_entropy
            }

        xgb_cov_stats = get_coverage_stats(xgb_cov_probs, y_true)
        vqc_cov_stats = get_coverage_stats(vqc_cov_probs, y_true)

        # Assemble full benchmark report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(y_true),
            "ablation_study": ablation_results,
            "robustness_perturbations": perturbation_results,
            "uncertainty_coverage": {
                "threshold": coverage_threshold,
                "classical_xgb": xgb_cov_stats,
                "quantum_vqc": vqc_cov_stats
            }
        }

        # Save to MongoDB
        db = get_database()
        db.ablation_studies.delete_many({}) # Clear old records
        db.ablation_studies.insert_one(report)
        print("Research benchmark report successfully written to MongoDB.")
        
        return report
