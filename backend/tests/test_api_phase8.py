import pytest
import numpy as np
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.fusion.rules_engine import RulesEngine

def test_rules_engine_high_category():
    """Verify that rules engine correctly identifies HIGH category cases."""
    fusion = MultimodalFeatureFusion()
    engine = RulesEngine()
    
    # 1. Test case: Crack & Pop + Severe limitation
    case_data = {
        "questionnaire": {
            "answers": {
                "pain_level": 5,
                "movement_limitation": "severe",
                "crack_pop": "yes"
            }
        }
    }
    _, vector, names = fusion.fuse_features(case_data)
    label, justification = engine.evaluate_rules(vector, names)
    assert label == "HIGH"
    assert "crack/pop accompanied by severe movement" in justification

    # 2. Test case: Bleeding & High Pain (Pain >= 7)
    case_data_bleeding = {
        "questionnaire": {
            "answers": {
                "pain_level": 7,
                "visible_bleeding": "yes"
            }
        }
    }
    _, vector_bleeding, names = fusion.fuse_features(case_data_bleeding)
    label, justification = engine.evaluate_rules(vector_bleeding, names)
    assert label == "HIGH"
    assert "bleeding" in justification

def test_rules_engine_moderate_category():
    """Verify that rules engine correctly identifies MODERATE category cases."""
    fusion = MultimodalFeatureFusion()
    engine = RulesEngine()
    
    # Test case: Moderate Pain (5/10) with direct impact
    case_data = {
        "questionnaire": {
            "answers": {
                "pain_level": 5,
                "direct_impact": "yes"
            }
        }
    }
    _, vector, names = fusion.fuse_features(case_data)
    label, justification = engine.evaluate_rules(vector, names)
    assert label == "MODERATE"
    assert "moderate pain level" in justification
    assert "direct impact" in justification

def test_rules_engine_low_category():
    """Verify that rules engine correctly identifies LOW category cases."""
    fusion = MultimodalFeatureFusion()
    engine = RulesEngine()
    
    # Test case: Low pain, no impact, baseline gravity
    case_data = {
        "questionnaire": {
            "answers": {
                "pain_level": 2,
                "injury_mechanism": "other",
                "direct_impact": "no"
            }
        }
    }
    _, vector, names = fusion.fuse_features(case_data)
    label, justification = engine.evaluate_rules(vector, names)
    assert label == "LOW"
    assert "LOW category" in justification
