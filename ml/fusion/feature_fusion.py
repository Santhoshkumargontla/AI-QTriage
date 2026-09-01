import numpy as np

# Canonical 23-feature schema. Order is part of the contract with XGBoost.
FEATURE_NAMES = [
    "vision_present",
    "prob_cut", "prob_bruise", "prob_swelling", "prob_other",
    "affected_ratio",
    "questionnaire_present",
    "pain_level",
    "mech_fall", "mech_impact", "mech_sports", "mech_sharp", "mech_other",
    "direct_impact", "visible_bleeding", "movement_limitation", "weight_bearing", "crack_pop",
    "sensor_present",
    "peak_g_force", "delta_v", "stabilization_time", "lux_drop",
]
N_FEATURES = 23
assert len(FEATURE_NAMES) == N_FEATURES


def _as_float(value, default=0.0):
    """Convert to float. Preserve actual zeros. None/unparseable → default (0.0), never 1.0 or 5.0."""
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _norm_answer(value):
    """Canonicalize a questionnaire answer. None / blank / not_provided stay missing (not 'yes'/'other')."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "not_provided", "none", "null", "nan"):
        return None
    return text


class MultimodalFeatureFusion:
    """Combines vision, questionnaire, and sensor inputs with missing modality fallbacks."""
    
    def __init__(self):
        self.feature_names = list(FEATURE_NAMES)

    def fuse_features(self, case_data: dict) -> tuple:
        """
        Fuses case data into a unified dictionary and a flat numerical numpy array.
        Returns:
            - fused_dict (dict)
            - fused_vector (np.ndarray of shape [23])
            - feature_names (list of str)
        """
        # --- 1. Vision Modality ---
        vision_analysis = case_data.get("vision_analysis", {})
        classes = vision_analysis.get("classification") if vision_analysis else None
        if vision_analysis and isinstance(classes, dict):
            # Extract probabilities. None (withheld class scores) → 0.0. Actual 0.0 is kept.
            prob_cut = _as_float(classes.get("Cut"))
            prob_bruise = _as_float(classes.get("Bruise"))
            # Swelling is legacy XGBoost schema only — not an active EfficientNet class.
            prob_swelling = _as_float(classes.get("Swelling"))
            # `Normal` / `OOD_Reject` are abstention classes, not 23-d injury features.
            prob_other = _as_float(classes.get("Other"))
            for reject_key in ("Normal", "OOD_Reject", "Ood_Reject", "ood_reject"):
                if classes.get(reject_key) is not None:
                    prob_other = max(prob_other, _as_float(classes.get(reject_key)))
            # kaggle-v1 injury classes without dedicated XGB dims → conservative prob_other bucket.
            for extra_injury in ("Burn", "Wound", "Laceration", "Abrasion"):
                if classes.get(extra_injury) is not None:
                    prob_other = max(prob_other, _as_float(classes.get(extra_injury)))
            
            segmentation = vision_analysis.get("segmentation") or {}
            affected_ratio = _as_float(segmentation.get("affected_ratio"))
            vision_present = 1.0
        else:
            # Option A Missingness: presence = 0.0, feature dimensions = 0.0
            prob_cut = 0.0
            prob_bruise = 0.0
            prob_swelling = 0.0
            prob_other = 0.0
            affected_ratio = 0.0
            vision_present = 0.0

        # --- 2. Questionnaire Modality ---
        quest = case_data.get("questionnaire", {})
        answers = quest.get("answers", {}) if quest else {}
        
        # Check presence
        if quest and answers:
            questionnaire_present = 1.0
            
            # Pain level: check pain_level, pain, pain_score keys in canonical answers
            raw_pain = None
            for key in ["pain_level", "pain", "pain_score"]:
                if key in answers and answers[key] is not None:
                    raw_pain = answers[key]
                    break
            
            if raw_pain is not None:
                try:
                    pain_val = float(raw_pain)
                except (ValueError, TypeError):
                    pain_val = 0.0
            else:
                pain_val = 0.0

            pain_level = pain_val / 10.0

            # Mechanism: missing stays all-zero. Do not invent "other" for unanswered cause.
            cause_val = _norm_answer(answers.get("cause")) or _norm_answer(answers.get("injury_mechanism"))
            if cause_val is None:
                mech_fall = mech_impact = mech_sports = mech_sharp = mech_other = 0.0
            else:
                mech_fall = 1.0 if cause_val in ("fall", "accident") else 0.0
                mech_impact = 1.0 if cause_val in ("direct_impact", "impact", "direct_blow") else 0.0
                mech_sports = 1.0 if cause_val in ("sports", "twist") else 0.0
                mech_sharp = 1.0 if cause_val in ("sharp_object", "cut") else 0.0
                mech_other = 1.0 if not (mech_fall or mech_impact or mech_sports or mech_sharp) else 0.0

            di = _norm_answer(answers.get("direct_impact"))
            direct_impact = 1.0 if (
                di in ("yes", "true", "1") or cause_val in ("direct_impact", "impact", "direct_blow")
            ) else 0.0

            bleeding_val = _norm_answer(answers.get("bleeding")) or _norm_answer(answers.get("visible_bleeding"))
            visible_bleeding = 1.0 if bleeding_val in ("yes", "mild", "heavy", "true") else 0.0

            crack_pop = 1.0 if _norm_answer(answers.get("crack_pop")) in ("yes", "true", "1") else 0.0

            # Categorical movement (none/missing=0.0, mild/moderate=0.5, severe=1.0)
            mov = _norm_answer(answers.get("movement")) or _norm_answer(answers.get("movement_limitation"))
            if mov in ("cannot_move", "cannot move", "severe"):
                movement_limitation = 1.0
            elif mov in ("limited", "with_pain", "mild", "stiff", "moderate"):
                movement_limitation = 0.5
            else:
                movement_limitation = 0.0

            # Categorical weight bearing (full/yes/missing=0.0, partial=0.5, unable/no=1.0)
            wb = _norm_answer(answers.get("limb_use")) or _norm_answer(answers.get("weight_bearing"))
            if wb in ("no", "cannot use", "cannot bear weight", "unable"):
                weight_bearing = 1.0
            elif wb in ("with_pain", "partial"):
                weight_bearing = 0.5
            else:
                weight_bearing = 0.0

        else:
            # Option A Missingness: presence = 0.0, feature dimensions = 0.0
            questionnaire_present = 0.0
            pain_level = 0.0
            mech_fall = 0.0
            mech_impact = 0.0
            mech_sports = 0.0
            mech_sharp = 0.0
            mech_other = 0.0
            direct_impact = 0.0
            visible_bleeding = 0.0
            movement_limitation = 0.0
            weight_bearing = 0.0
            crack_pop = 0.0

        # --- 3. Sensor Modality ---
        sensor = case_data.get("sensor_summary", {})
        if sensor:
            sensor_present = 1.0
            peak_g_force = _as_float(sensor.get("peak_g_force"))
            delta_v = _as_float(sensor.get("pre_impact_delta_v"))
            stabilization_time = _as_float(sensor.get("post_impact_stabilization_seconds"))
            lux_drop = 1.0 if sensor.get("optical_lux_drop") else 0.0
        else:
            # Option A Missingness: presence = 0.0, feature dimensions = 0.0
            sensor_present = 0.0
            peak_g_force = 0.0
            delta_v = 0.0
            stabilization_time = 0.0
            lux_drop = 0.0

        # Build fused dictionary representation
        fused_dict = {
            "vision": {
                "present": bool(vision_present),
                "classification_probabilities": {
                    "Cut": prob_cut, "Bruise": prob_bruise, "Swelling": prob_swelling, "Other": prob_other
                },
                "affected_ratio": affected_ratio
            },
            "questionnaire": {
                "present": bool(questionnaire_present),
                "pain_level": pain_level,
                "injury_mechanism_one_hot": {
                    "fall": mech_fall, "impact": mech_impact, "sports": mech_sports, "sharp_object": mech_sharp, "other": mech_other
                },
                "direct_impact": bool(direct_impact),
                "visible_bleeding": bool(visible_bleeding),
                "movement_limitation": movement_limitation,
                "weight_bearing": weight_bearing,
                "crack_pop": bool(crack_pop)
            },
            "sensor": {
                "present": bool(sensor_present),
                "peak_g_force": peak_g_force,
                "delta_v": delta_v,
                "stabilization_time": stabilization_time,
                "lux_drop": bool(lux_drop)
            }
        }

        # Build flat float array
        vector = np.array([
            vision_present,
            prob_cut, prob_bruise, prob_swelling, prob_other,
            affected_ratio,
            questionnaire_present,
            pain_level,
            mech_fall, mech_impact, mech_sports, mech_sharp, mech_other,
            direct_impact, visible_bleeding, movement_limitation, weight_bearing, crack_pop,
            sensor_present,
            peak_g_force, delta_v, stabilization_time, lux_drop
        ], dtype=np.float32)

        return fused_dict, vector, self.feature_names
