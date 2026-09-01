import pytest
import numpy as np
from ml.fusion.feature_fusion import MultimodalFeatureFusion

def test_multimodal_feature_fusion_all_modalities():
    """Verify that feature fusion successfully aggregates vision, voice, and sensor inputs."""
    fusion = MultimodalFeatureFusion()
    
    # Mock case dictionary with all streams populated
    case_data = {
        "case_id": "test_full_fusion",
        "vision_analysis": {
            "classification": {
                "Cut": 0.10, "Bruise": 0.15, "Swelling": 0.70, "Other": 0.05
            },
            "segmentation": {
                "affected_ratio": 0.18
            }
        },
        "questionnaire": {
            "answers": {
                "pain_level": 8,
                "injury_mechanism": "sports",
                "direct_impact": "yes",
                "visible_bleeding": "no",
                "movement_limitation": "mild",
                "weight_bearing": "partial",
                "crack_pop": "yes"
            }
        },
        "sensor_summary": {
            "peak_g_force": 4.5,
            "pre_impact_delta_v": 1.2,
            "post_impact_stabilization_seconds": 0.40,
            "optical_lux_drop": True
        }
    }
    
    fused_dict, vector, names = fusion.fuse_features(case_data)
    
    # 1. Shape validations
    assert len(names) == 23
    assert vector.shape == (23,)
    
    # 2. Presence flags validation
    assert vector[names.index("vision_present")] == 1.0
    assert vector[names.index("questionnaire_present")] == 1.0
    assert vector[names.index("sensor_present")] == 1.0
    
    # 3. Value calculations validation
    assert vector[names.index("prob_swelling")] == pytest.approx(0.70)
    assert vector[names.index("affected_ratio")] == pytest.approx(0.18)
    assert vector[names.index("pain_level")] == pytest.approx(0.80) # 8 / 10
    
    # One-hot mechanism matches sports
    assert vector[names.index("mech_sports")] == 1.0
    assert vector[names.index("mech_fall")] == 0.0
    
    # Limitation and weight-bearing mapping
    assert vector[names.index("movement_limitation")] == 0.5 # mild
    assert vector[names.index("weight_bearing")] == 0.5 # partial
    assert vector[names.index("direct_impact")] == 1.0
    assert vector[names.index("crack_pop")] == 1.0
    
    # Sensor validations
    assert vector[names.index("peak_g_force")] == pytest.approx(4.5)
    assert vector[names.index("delta_v")] == pytest.approx(1.2)
    assert vector[names.index("stabilization_time")] == pytest.approx(0.40)
    assert vector[names.index("lux_drop")] == 1.0

def test_multimodal_feature_fusion_missing_modalities():
    """Verify that feature fusion handles missing streams using Option A zero-padding."""
    fusion = MultimodalFeatureFusion()
    
    # Mock case dictionary with ONLY questionnaire (image and sensor skipped)
    case_data = {
        "case_id": "test_missing_fusion",
        "questionnaire": {
            "answers": {
                "pain_level": 5,
                "injury_mechanism": "sharp_object",
                "direct_impact": "no",
                "visible_bleeding": "yes"
            }
        }
    }
    
    fused_dict, vector, names = fusion.fuse_features(case_data)
    
    # 1. Dimension holds at 23
    assert len(names) == 23
    assert vector.shape == (23,)
    
    # 2. Presence flags check
    assert vector[names.index("vision_present")] == 0.0
    assert vector[names.index("questionnaire_present")] == 1.0
    assert vector[names.index("sensor_present")] == 0.0
    
    # 3. Vision Option A missingness check (presence=0, features=0)
    assert vector[names.index("prob_cut")] == 0.0
    assert vector[names.index("prob_bruise")] == 0.0
    assert vector[names.index("affected_ratio")] == 0.0
    
    # 4. Questionnaire mapping check
    assert vector[names.index("pain_level")] == 0.50
    assert vector[names.index("mech_sharp")] == 1.0
    assert vector[names.index("visible_bleeding")] == 1.0
    
    # 5. Sensor Option A missingness check (presence=0, features=0)
    assert vector[names.index("peak_g_force")] == 0.0
    assert vector[names.index("delta_v")] == 0.0
    assert vector[names.index("stabilization_time")] == 0.0
    assert vector[names.index("lux_drop")] == 0.0

def test_all_ten_modality_combinations_option_a():
    """Verify feature fusion and XGBoost prediction across all 10 supported modality combinations."""
    from ml.classifiers.xgboost_classifier import XGBoostClassifier
    fusion = MultimodalFeatureFusion()
    clf = XGBoostClassifier(model_path="ml/models/xgboost_best.json")

    v_data = {"classification": {"Cut": 0.8, "Bruise": 0.1, "Swelling": 0.05, "Other": 0.05}, "segmentation": {"affected_ratio": 0.12}}
    q_data = {"answers": {"pain_level": 7, "injury_mechanism": "sports"}}
    s_data = {"peak_g_force": 5.2, "pre_impact_delta_v": 2.1}

    combinations = [
        {"vision_analysis": v_data},  # 1. Vision only
        {"questionnaire": q_data},    # 2. Questionnaire only
        {"sensor_summary": s_data},   # 3. Sensor only
        {"vision_analysis": v_data, "questionnaire": q_data},  # 4. Vision + Quest
        {"vision_analysis": v_data, "sensor_summary": s_data},   # 5. Vision + Sensor
        {"questionnaire": q_data, "sensor_summary": s_data},    # 6. Quest + Sensor
        {"vision_analysis": v_data, "questionnaire": q_data, "sensor_summary": s_data}, # 7. All modalities
        {"questionnaire": q_data, "sensor_summary": s_data},    # 8. Missing vision
        {"vision_analysis": v_data, "sensor_summary": s_data},   # 9. Missing questionnaire
        {"vision_analysis": v_data, "questionnaire": q_data},  # 10. Missing sensor
    ]

    for idx, case_input in enumerate(combinations, 1):
        fused_dict, vector, names = fusion.fuse_features(case_input)
        assert vector.shape == (23,), f"Combo {idx} failed shape test"
        assert not np.isnan(vector).any(), f"Combo {idx} contained NaN"
        assert not np.isinf(vector).any(), f"Combo {idx} contained Inf"
        
        pred_idx, probs = clf.predict(vector)
        assert pred_idx in (0, 1, 2), f"Combo {idx} invalid prediction index"
        assert len(probs) == 3, f"Combo {idx} invalid probability output length"
        assert abs(sum(probs) - 1.0) < 1e-4, f"Combo {idx} probabilities do not sum to 1"


def test_frontend_vocab_maps_without_fabricating_missing_cause():
    fusion = MultimodalFeatureFusion()
    fused, vector, names = fusion.fuse_features({
        "questionnaire": {
            "answers": {
                "pain_level": 7,
                "cause": "direct_blow",
                "movement_limitation": "moderate",
                "weight_bearing": "unable",
                "crack_pop": "not_provided",
            }
        }
    })
    assert vector[names.index("mech_impact")] == 1.0
    assert vector[names.index("mech_other")] == 0.0
    assert vector[names.index("direct_impact")] == 1.0
    assert vector[names.index("movement_limitation")] == 0.5
    assert vector[names.index("weight_bearing")] == 1.0
    assert vector[names.index("crack_pop")] == 0.0

    missing_cause, vec2, names2 = fusion.fuse_features({
        "questionnaire": {"answers": {"pain_level": 4, "cause": None, "weight_bearing": None}}
    })
    assert vec2[names2.index("mech_fall")] == 0.0
    assert vec2[names2.index("mech_impact")] == 0.0
    assert vec2[names2.index("mech_sports")] == 0.0
    assert vec2[names2.index("mech_sharp")] == 0.0
    assert vec2[names2.index("mech_other")] == 0.0
    assert vec2[names2.index("weight_bearing")] == 0.0


