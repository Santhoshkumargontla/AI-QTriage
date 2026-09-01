import numpy as np

class EvidenceConsistencyAnalyzer:
    """Calculates feature alignment across modalities and evaluates model sensitivity boundaries."""
    
    def __init__(self):
        pass

    def calculate_consistency(
        self, 
        fused_vector: np.ndarray, 
        feature_names: list,
        xgb_prediction_class: str = None,
        vqc_prediction_class: str = None,
        segmentation_reliable: bool = True,
        is_demo: bool = False
    ) -> dict:
        """
        Computes an alignment score and deduction justification between independent modalities,
        taking model prediction agreement, segmentation reliability, and input modalities into account.
        """
        feat = {name: float(fused_vector[i]) for i, name in enumerate(feature_names)}
        
        score = 100.0
        conflicts = []
        agreements = []

        # 1. Model Agreement Evaluation
        models_disagree = False
        if xgb_prediction_class is not None and vqc_prediction_class is not None:
            xgb_clean = str(xgb_prediction_class).upper().strip()
            vqc_clean = str(vqc_prediction_class).upper().strip()
            if xgb_clean != vqc_clean:
                models_disagree = True
                score -= 40.0
                conflicts.append(f"Model Disagreement: Classical XGBoost ({xgb_prediction_class}) and Quantum VQC ({vqc_prediction_class}) predicted different severity classes.")
            else:
                agreements.append(f"Model Agreement: Both Classical XGBoost and Quantum VQC predicted {xgb_prediction_class}.")

        # 2. Segmentation Reliability Evaluation
        if not segmentation_reliable:
            score -= 15.0
            conflicts.append("Segmentation Warning: No reliable segmentation mask was available for area calculation.")

        # 3. Data Provenance Notice
        if is_demo:
            score -= 10.0
            conflicts.append("Data Notice: Assessment uses synthetic or demonstration data.")

        # Extract features
        pain = round(feat.get("pain_level", 0.0) * 10.0)
        bleeding = feat.get("visible_bleeding", 0.0) > 0.5
        g_force = feat.get("peak_g_force", 0.0)
        movement = feat.get("movement_limitation", 0.0)
        weight_bearing = feat.get("weight_bearing", 0.0)
        crack_pop = feat.get("crack_pop", 0.0) > 0.5
        direct_impact = feat.get("direct_impact", 0.0) > 0.5
        
        prob_cut = feat.get("prob_cut", 0.0)
        prob_bruise = feat.get("prob_bruise", 0.0)
        prob_swelling = feat.get("prob_swelling", 0.0)

        vision_present = feat.get("vision_present", 0.0) > 0.5
        sensor_present = feat.get("sensor_present", 0.0) > 0.5

        # 4. Feature Heuristic Conflicts
        if prob_swelling > 0.5 and sensor_present and g_force < 1.5:
            score -= 25.0
            conflicts.append("Conflict: High visual swelling prediction is not supported by low sensor kinetic impact peak (<1.5g).")
            
        if prob_cut > 0.5 and not bleeding:
            score -= 20.0
            conflicts.append("Conflict: Visual cut prediction is present but no active bleeding was reported.")
            
        if crack_pop and sensor_present and g_force < 2.0:
            score -= 30.0
            conflicts.append("Conflict: Reported crack/pop sensation is not supported by sufficient sensor impact peak (<2.0g).")
            
        if pain >= 8 and sensor_present and g_force < 1.2 and not vision_present:
            score -= 25.0
            conflicts.append("Conflict: Severe subjective pain (>=8/10) reported without visible findings or measured physical impact.")

        # 5. Feature Heuristic Agreements
        if prob_swelling > 0.6 and g_force >= 3.5:
            agreements.append("Agreement: Visual swelling corresponds with strong kinetic peak impact.")
        if bleeding and prob_cut > 0.6:
            agreements.append("Agreement: Reported bleeding matches visual laceration detection.")
        if direct_impact and g_force >= 3.0:
            agreements.append("Agreement: Subjective direct impact is corroborated by sensor kinetic metrics.")

        # Ensure bounds
        score = max(0.0, min(100.0, score))
        
        # Mapping labels — Model disagreement MUST prevent "Highly Consistent"
        if models_disagree:
            if score >= 50.0:
                status = "Partially Consistent"
            else:
                status = "Conflicting Evidence Detected"
        else:
            if score >= 80.0:
                status = "Highly Consistent"
            elif score >= 50.0:
                status = "Partially Consistent"
            else:
                status = "Conflicting Evidence Detected"

        return {
            "score": float(round(score, 1)),
            "status": status,
            "conflicts": conflicts,
            "agreements": agreements,
            "explanation": "Evidence consistency scoring measures alignment between independent data channels and model predictions. It does not validate clinical diagnostic accuracy."
        }

    def analyze_counterfactuals(self, fused_vector: np.ndarray, feature_names: list, xgb_model, vqc_model) -> dict:
        """
        Performs a sweep over key inputs to locate prediction transition thresholds.
        """
        # Get baseline prediction index
        base_xgb_pred, _ = xgb_model.predict(fused_vector)
        vqc_available = vqc_model is not None and getattr(vqc_model, "is_trained", False)

        def _vqc_pred(vec):
            nonlocal vqc_available
            if not vqc_available:
                return None
            try:
                idx, _ = vqc_model.predict(vec)
                return idx
            except (RuntimeError, ValueError, FileNotFoundError, OSError):
                vqc_available = False
                return None

        base_vqc_pred = _vqc_pred(fused_vector)

        label_map = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        
        # --- A. Pain Level Sweep ---
        # Sweep pain from 0.0 (0/10) to 1.0 (10/10) in increments of 0.1
        pain_idx = feature_names.index("pain_level")
        pain_xgb_transitions = []
        pain_vqc_transitions = []
        
        for p_val in np.linspace(0.0, 1.0, 11):
            test_vec = fused_vector.copy()
            test_vec[pain_idx] = p_val
            
            x_pred, _ = xgb_model.predict(test_vec)
            if x_pred != base_xgb_pred:
                pain_xgb_transitions.append({
                    "pain_level": int(round(p_val * 10.0)),
                    "new_prediction": label_map[x_pred]
                })
            v_pred = _vqc_pred(test_vec)
            if v_pred is not None and v_pred != base_vqc_pred:
                    pain_vqc_transitions.append({
                        "pain_level": int(round(p_val * 10.0)),
                        "new_prediction": label_map[v_pred]
                    })

        # --- B. Peak G-Force Sweep ---
        # Sweep peak g-force from 1.0g to 8.0g in increments of 0.5
        g_idx = feature_names.index("peak_g_force")
        g_xgb_transitions = []
        g_vqc_transitions = []
        
        for g_val in np.linspace(1.0, 8.0, 15):
            test_vec = fused_vector.copy()
            test_vec[g_idx] = g_val
            
            x_pred, _ = xgb_model.predict(test_vec)
            if x_pred != base_xgb_pred:
                g_xgb_transitions.append({
                    "peak_g_force": float(round(g_val, 2)),
                    "new_prediction": label_map[x_pred]
                })
            v_pred = _vqc_pred(test_vec)
            if v_pred is not None and v_pred != base_vqc_pred:
                    g_vqc_transitions.append({
                        "peak_g_force": float(round(g_val, 2)),
                        "new_prediction": label_map[v_pred]
                    })

        return {
            "explanation": "Counterfactual sensitivity analysis. This shows how altering specific inputs (like pain level or sensor peak g-force) shifts the model's output category.",
            "baseline_predictions": {
                "classical_xgb": label_map[base_xgb_pred],
                "quantum_vqc": label_map[base_vqc_pred] if base_vqc_pred is not None else None
            },
            "pain_sensitivity": {
                "classical_xgb_transitions": pain_xgb_transitions[:2], # keep first transitions for brevity
                "quantum_vqc_transitions": pain_vqc_transitions[:2]
            },
            "g_force_sensitivity": {
                "classical_xgb_transitions": g_xgb_transitions[:2],
                "quantum_vqc_transitions": g_vqc_transitions[:2]
            }
        }
