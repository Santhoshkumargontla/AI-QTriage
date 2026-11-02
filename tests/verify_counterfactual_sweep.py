import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database.connection import get_database
from ml.fusion.feature_fusion import MultimodalFeatureFusion, FEATURE_NAMES
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.models.canonical_paths import XGB_CANONICAL, resolve_existing

print("=" * 80)
print("CHECKLIST 10: COUNTERFACTUAL SENSITIVITY SWEEP & PROBABILITY NORMALIZATION")
print("=" * 80)

db = get_database()
case = db["cases"].find_one({"case_id": "a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"})
fusion = MultimodalFeatureFusion()
_, vector, names = fusion.fuse_features(case)

clf = XGBoostClassifier(model_path=resolve_existing(XGB_CANONICAL))
label_map = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

g_idx = list(names).index("peak_g_force")
sweep_vals = np.arange(0.0, 3.25, 0.25)

print(f"{'G-Force (g)':<12} | {'Class':<10} | {'P(LOW)':<10} | {'P(MOD)':<10} | {'P(HIGH)':<10} | {'Sum(P)':<10} | Status")
print("-" * 80)

transition_g = None
prev_class = None

for g in sweep_vals:
    test_vec = vector.copy()
    test_vec[g_idx] = float(g)
    pred_idx, probs = clf.predict(test_vec)
    pred_class = label_map[pred_idx]
    prob_sum = sum(probs)
    
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1.0 at {g}g: {prob_sum}"
    
    status = ""
    if prev_class is not None and pred_class != prev_class:
        status = f"TRANSITION -> {pred_class} (synthetic model decision boundary)"
        if transition_g is None:
            transition_g = g
    prev_class = pred_class
    
    print(f"{g:<12.2f} | {pred_class:<10} | {probs[0]:<10.4f} | {probs[1]:<10.4f} | {probs[2]:<10.4f} | {prob_sum:<10.4f} | {status}")

print("=" * 80)
print(f"Verified Transition Point: {transition_g:.2f}g")
print("Probability Normalization: 100.0% verified across all sweep steps.")
print("Scope Limitation: This boundary reflects synthetic XGBoost decision surface only.")
