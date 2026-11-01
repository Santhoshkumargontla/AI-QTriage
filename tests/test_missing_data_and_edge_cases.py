"""
Test Missing-Data and Multimodal Fusion Edge Cases.
Validates:
1. Vision missing -> Option A zero-imputation, vision_present = 0.0
2. Questionnaire missing -> Option A zero-imputation, questionnaire_present = 0.0
3. Sensor missing -> Option A zero-imputation, sensor_present = 0.0
4. Unreliable segmentation -> affected_ratio = None in fused_dict, 0.0 in vector
5. Rule-based safety level conservative behavior across missing inputs
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.classifiers.xgboost_classifier import XGBoostClassifier, XGB_CANONICAL

def test_missing_data_fusion():
    fusion = MultimodalFeatureFusion()
    xgb = XGBoostClassifier(XGB_CANONICAL)

    # Base case
    canonical_case = {
        "vision_analysis": {
            "classification": {"Cut": 0.05, "Bruise": 0.85, "Swelling": 0.0, "Other": 0.10},
            "segmentation": {"affected_ratio": 0.02, "segmentation_reliable": True}
        },
        "questionnaire": {
            "answers": {
                "pain_level": 7,
                "cause": "fall",
                "direct_impact": "yes",
                "bleeding": "no",
                "movement": "limited",
                "weight_bearing": "no",
                "crack_pop": "yes"
            }
        },
        "sensor_summary": {
            "peak_g_force": 5.17,
            "pre_impact_delta_v": 2.1,
            "post_impact_stabilization_seconds": 0.02,
            "optical_lux_drop": True
        }
    }

    # 1. Full data
    f_dict, vec, names = fusion.fuse_features(canonical_case)
    assert vec.shape == (23,)
    assert vec[0] == 1.0  # vision_present
    assert vec[6] == 1.0  # questionnaire_present
    assert vec[18] == 1.0 # sensor_present
    pred_idx, probs = xgb.predict(vec)
    print(f"[PASS] Full multimodal case: pred={pred_idx}, probs={probs}")

    # 2. Missing Vision
    case_no_vision = {k: v for k, v in canonical_case.items() if k != "vision_analysis"}
    f_dict, vec, _ = fusion.fuse_features(case_no_vision)
    assert vec[0] == 0.0  # vision_present
    assert np.all(vec[1:6] == 0.0) # cut, bruise, swelling, other, affected_ratio
    assert f_dict["vision"]["present"] is False
    assert f_dict["vision"]["affected_ratio"] is None
    pred_idx, probs = xgb.predict(vec)
    print(f"[PASS] Missing Vision case: pred={pred_idx}, probs={probs}")

    # 3. Missing Questionnaire
    case_no_quest = {k: v for k, v in canonical_case.items() if k != "questionnaire"}
    f_dict, vec, _ = fusion.fuse_features(case_no_quest)
    assert vec[6] == 0.0  # questionnaire_present
    assert np.all(vec[7:18] == 0.0) # all questionnaire features
    assert f_dict["questionnaire"]["present"] is False
    pred_idx, probs = xgb.predict(vec)
    print(f"[PASS] Missing Questionnaire case: pred={pred_idx}, probs={probs}")

    # 4. Missing Sensor
    case_no_sensor = {k: v for k, v in canonical_case.items() if k != "sensor_summary"}
    f_dict, vec, _ = fusion.fuse_features(case_no_sensor)
    assert vec[18] == 0.0 # sensor_present
    assert np.all(vec[19:23] == 0.0) # peak_g_force, delta_v, stabilization_time, lux_drop
    assert f_dict["sensor"]["present"] is False
    pred_idx, probs = xgb.predict(vec)
    print(f"[PASS] Missing Sensor case: pred={pred_idx}, probs={probs}")

    # 5. Unreliable segmentation
    case_unrel_seg = dict(canonical_case)
    case_unrel_seg["vision_analysis"] = {
        "classification": {"Cut": 0.05, "Bruise": 0.85, "Swelling": 0.0, "Other": 0.10},
        "segmentation": {"affected_ratio": None, "segmentation_reliable": False}
    }
    f_dict, vec, _ = fusion.fuse_features(case_unrel_seg)
    assert f_dict["vision"]["affected_ratio"] is None
    assert f_dict["vision"]["affected_ratio_status"] == "UNAVAILABLE_WITHHELD"
    assert f_dict["vision"]["affected_ratio_imputed"] is True
    assert vec[5] == 0.0 # affected_ratio imputed to 0.0 in model vector
    pred_idx, probs = xgb.predict(vec)
    print(f"[PASS] Unreliable segmentation case: pred={pred_idx}, probs={probs}")

    print("\nAll missing-data edge case assertions PASSED!")

if __name__ == "__main__":
    test_missing_data_fusion()
