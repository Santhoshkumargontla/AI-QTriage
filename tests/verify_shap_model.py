import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import xgboost as xgb
import shap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database.connection import get_database
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.models.canonical_paths import XGB_CANONICAL, resolve_existing

db = get_database()
case = db["cases"].find_one({"case_id": "a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"})
if not case:
    print("ERROR: Case not found in database.")
    sys.exit(1)

fusion = MultimodalFeatureFusion()
fused_dict, vector, names = fusion.fuse_features(case)

print(f"Feature vector shape: {vector.shape}, dtype: {vector.dtype}")
clf = XGBoostClassifier(model_path=resolve_existing(XGB_CANONICAL))

pred_idx, probs = clf.predict(vector)
print(f"Recalculated Prediction: pred_idx={pred_idx} (HIGH), probs={[round(p, 6) for p in probs]}")
print(f"Stored Probability in DB: {case.get('xgboost_prediction', {}).get('probability')}")

x_in = vector.reshape(1, -1)
booster = clf.model.get_booster()
dmat = xgb.DMatrix(x_in, feature_names=list(names))
raw_margins = booster.predict(dmat, output_margin=True)[0]
print(f"Booster Raw Margins (0=LOW, 1=MODERATE, 2=HIGH): {raw_margins}")

# Softmax verification
exp_m = np.exp(raw_margins - np.max(raw_margins))
softmax_probs = exp_m / np.sum(exp_m)
print(f"Softmax of Raw Margins: {[round(float(p), 6) for p in softmax_probs]}")
print(f"Class 2 Softmax: {softmax_probs[2]:.6f} vs Probs[2]: {probs[2]:.6f}")

explainer = shap.TreeExplainer(clf.model)
shap_vals = explainer.shap_values(x_in)
base_vals = explainer.expected_value
print(f"TreeExplainer Base Values: {base_vals}")

class_shap = shap_vals[2][0] if isinstance(shap_vals, list) else shap_vals[0, :, 2]
base_val = float(base_vals[2])

features = []
for i, name in enumerate(names):
    features.append({"feature": name, "val": float(class_shap[i])})

features.sort(key=lambda x: abs(x["val"]), reverse=True)

top6 = features[:6]
tail17 = features[6:]

top6_raw_sum = sum(f["val"] for f in top6)
tail17_raw_sum = sum(f["val"] for f in tail17)
total_shap_raw_sum = sum(f["val"] for f in features)
total_margin = base_val + total_shap_raw_sum

print("\n" + "=" * 60)
print("SHAP ADDITIVITY & ROUNDING RECONCILIATION")
print("=" * 60)
print(f"Base value (raw)      : {base_val:+.8f}  ->  {base_val:+.3f}")
print(f"Top 6 sum (raw)       : {top6_raw_sum:+.8f}  ->  {top6_raw_sum:+.3f}")
print(f"Tail 17 sum (raw)     : {tail17_raw_sum:+.8f}  ->  {tail17_raw_sum:+.3f}")
print(f"Total SHAP sum (raw)  : {total_shap_raw_sum:+.8f}  ->  {total_shap_raw_sum:+.3f}")
print(f"Calculated Margin     : {total_margin:+.8f}  ->  {total_margin:+.3f}")
print(f"Booster Direct Margin : {raw_margins[2]:+.8f}  ->  {raw_margins[2]:+.3f}")
print(f"Discrepancy           : {abs(total_margin - raw_margins[2]):.2e}")

# Also sum of individual rounded values
top6_rounded_sum = sum(round(f["val"], 3) for f in top6)
tail17_rounded_sum = sum(round(f["val"], 3) for f in tail17)
print(f"Sum of top 6 rounded  : {top6_rounded_sum:+.3f}")
print(f"Sum of tail 17 rounded: {tail17_rounded_sum:+.3f}")

print("\n" + "=" * 60)
print("STORED SHAP VALUES IN ACTIVE CASE (DATABASE RECORD)")
print("=" * 60)
stored_shaps = list(case.get("shap_explanations", []))
stored_shaps.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
s_top6 = stored_shaps[:6]
s_tail17 = stored_shaps[6:]
s_top6_sum = sum(x["shap_value"] for x in s_top6)
s_tail17_sum = sum(x["shap_value"] for x in s_tail17)
s_net_sum = sum(x["shap_value"] for x in stored_shaps)
s_margin = base_val + s_net_sum
print(f"Stored Base Value (raw) : {base_val:+.8f} -> rounded: {base_val:+.3f}")
print(f"Stored Top 6 sum (raw)  : {s_top6_sum:+.8f} -> rounded: {s_top6_sum:+.3f}")
print(f"Stored Tail 17 sum (raw): {s_tail17_sum:+.8f} -> rounded: {s_tail17_sum:+.3f}")
print(f"Stored Net sum (raw)    : {s_net_sum:+.8f} -> rounded: {s_net_sum:+.3f}")
print(f"Stored Margin (raw)     : {s_margin:+.8f} -> rounded: {s_margin:+.3f}")
print(f"Display Additivity: {s_top6_sum:+.3f} + ({s_tail17_sum:+.3f}) = {s_top6_sum + s_tail17_sum:+.3f}")
print(f"Display Margin Eq : {base_val:+.3f} + {s_net_sum:+.3f} = {base_val + s_net_sum:+.3f}")

print("\n" + "=" * 60)
print("ALL 23 FEATURES (CLASS 2: HIGH)")
print("=" * 60)
for i, f in enumerate(stored_shaps, 1):
    print(f"{i:2d}. {f['feature']:25s} : raw={f['shap_value']:+.8f}  display={f['shap_value']:+.3f}")

