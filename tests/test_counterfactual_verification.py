"""
Independent Verification Script for Counterfactual Decision Boundary.
Tests fine-grained sweep of peak_g_force while holding all other 22 features strictly constant.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import urllib.request
import numpy as np
from ml.classifiers.xgboost_classifier import XGBoostClassifier, XGB_CANONICAL
from ml.fusion.feature_fusion import MultimodalFeatureFusion

def verify_counterfactual():
    print("=" * 76)
    print("INDEPENDENT COUNTERFACTUAL BOUNDARY & PROBABILITY SWEEP")
    print("=" * 76)
    
    # Ingest production canonical case
    url = "http://127.0.0.1:8000/api/cases/bd890e11-530d-4d0e-8c4f-4bda296a8600"
    req = urllib.request.urlopen(url)
    case = json.loads(req.read())
    
    fusion = MultimodalFeatureFusion()
    case_data = {
        "vision_analysis": {
            "classification": case["visible_injury"]["classification"],
            "segmentation": {"affected_ratio": case["visible_injury"]["affected_ratio"]}
        },
        "questionnaire": case["questionnaire"],
        "sensor_summary": case["sensor_summary"]
    }
    _, baseline_vec, names = fusion.fuse_features(case_data)
    model = XGBoostClassifier(XGB_CANONICAL)
    
    g_idx = names.index("peak_g_force")
    labels = ["LOW", "MODERATE", "HIGH"]
    
    base_pred_idx, base_probs = model.predict(baseline_vec)
    print(f"Target Feature: Index {g_idx} -> '{names[g_idx]}'")
    print(f"Baseline Value: {baseline_vec[g_idx]:.2f}g")
    print(f"Baseline Category: {labels[base_pred_idx]} (Probability: {base_probs[base_pred_idx]*100:.2f}%)\n")
    print(f"{'G-Force':<10} | {'P(LOW)':<10} | {'P(MODERATE)':<12} | {'P(HIGH)':<10} | {'Predicted Category':<18}")
    print("-" * 72)
    
    transition_point = None
    prev_pred = None
    
    # Sweep from 0.00g to 2.00g in 0.05g increments
    for g in np.arange(0.00, 2.05, 0.05):
        test_vec = baseline_vec.copy()
        test_vec[g_idx] = float(g)
        
        # Verify all other 22 features remain identical
        diff_count = np.sum(test_vec != baseline_vec)
        assert diff_count == (1 if g != baseline_vec[g_idx] else 0), "Error: Multiple features changed!"
        
        pred_idx, probs = model.predict(test_vec)
        pred_label = labels[pred_idx]
        
        if prev_pred is not None and pred_idx != prev_pred and transition_point is None:
            transition_point = g
            print(f">>> BOUNDARY TRANSITION AT {g:.2f}g (Switched from {labels[prev_pred]} to {pred_label}) <<<")
        
        print(f"{g:<10.2f} | {probs[0]*100:<9.2f}% | {probs[1]*100:<11.2f}% | {probs[2]*100:<9.2f}% | {pred_label:<18}")
        prev_pred = pred_idx
        
    print("-" * 72)
    print(f"Verified Transition Point: Exactly at peak_g_force = {transition_point:.2f}g")
    print("=" * 76)

if __name__ == "__main__":
    verify_counterfactual()
