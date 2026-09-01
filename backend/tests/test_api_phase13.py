import os
import pytest
import numpy as np
import torch
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.explainability.evidence_consistency import EvidenceConsistencyAnalyzer

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_evidence_consistency_scoring():
    """Verify that logical contradictions trigger deductions and output correct alignment labels."""
    fusion = MultimodalFeatureFusion()
    analyzer = EvidenceConsistencyAnalyzer()
    
    # 1. Contradiction: Visual swelling but low kinetic impact (<1.5g)
    case_conflict = {
        "vision_analysis": {
            "classification": {"Swelling": 0.80},
            "segmentation": {"affected_ratio": 0.12}
        },
        "sensor_summary": {
            "peak_g_force": 1.1, # Too low for severe swelling
            "pre_impact_delta_v": 0.1,
            "post_impact_stabilization_seconds": 0.1
        }
    }
    _, vector, names = fusion.fuse_features(case_conflict)
    result = analyzer.calculate_consistency(vector, names)
    
    assert result["score"] == 75.0 # 100 - 25
    assert result["status"] == "Partially Consistent"
    assert len(result["conflicts"]) == 1
    assert "not supported by low sensor kinetic impact" in result["conflicts"][0]
    assert "does not validate clinical diagnostic accuracy" in result["explanation"]

    # 2. Consistent Case
    case_consistent = {
        "vision_analysis": {
            "classification": {"Swelling": 0.85},
            "segmentation": {"affected_ratio": 0.15}
        },
        "questionnaire": {
            "answers": {
                "pain_level": 7,
                "injury_mechanism": "sports",
                "direct_impact": "yes"
            }
        },
        "sensor_summary": {
            "peak_g_force": 4.2,
            "pre_impact_delta_v": 1.1,
            "post_impact_stabilization_seconds": 0.40
        }
    }
    _, vector_cons, names = fusion.fuse_features(case_consistent)
    result_cons = analyzer.calculate_consistency(vector_cons, names)
    assert result_cons["score"] == 100.0
    assert result_cons["status"] == "Highly Consistent"
    assert len(result_cons["agreements"]) >= 2

def test_counterfactual_sensitivity_analysis(temp_weights_dir):
    """Verify that sweeping variables identifies model decision boundaries."""
    fusion = MultimodalFeatureFusion()
    analyzer = EvidenceConsistencyAnalyzer()
    
    # 1. Fit mock models
    np.random.seed(42)
    X_train = np.random.randn(20, 23).astype(np.float32)
    y_train = np.random.randint(0, 3, 20).astype(np.int32)
    
    xgb_path = os.path.join(temp_weights_dir, "xgb.json")
    xgb = XGBoostClassifier()
    xgb.train(X_train, y_train)
    xgb.save_model(xgb_path)
    
    vqc = VQCClassifier()
    vqc.train(X_train, y_train, epochs=2)
    vqc.save_model(temp_weights_dir)

    # 2. Define test case vector
    case_data = {
        "questionnaire": {
            "answers": {
                "pain_level": 4,
                "injury_mechanism": "sports",
                "direct_impact": "yes"
            }
        },
        "sensor_summary": {
            "peak_g_force": 2.5,
            "pre_impact_delta_v": 0.5
        }
    }
    _, vector, names = fusion.fuse_features(case_data)

    # 3. Run counterfactual analysis
    cf_report = analyzer.analyze_counterfactuals(vector, names, xgb, vqc)
    
    assert "baseline_predictions" in cf_report
    assert "pain_sensitivity" in cf_report
    assert "g_force_sensitivity" in cf_report
    assert "classical_xgb" in cf_report["baseline_predictions"]
    assert "quantum_vqc" in cf_report["baseline_predictions"]
    assert "Counterfactual sensitivity analysis" in cf_report["explanation"]
