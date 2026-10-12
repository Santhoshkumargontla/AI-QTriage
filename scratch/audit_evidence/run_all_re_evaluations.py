import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import joblib

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score
from ml.models.canonical_paths import (
    YOLO_CANONICAL, EFFNET_CANONICAL, UNET_CANONICAL, SENSOR_MODEL, SENSOR_SCALER,
    XGB_CANONICAL, VQC_DIR, VQC_WEIGHTS
)
from ml.training.train_xgboost import generate_multimodal_dataset, get_leakage_free_splits
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier

out_dir = "scratch/audit_evidence"
os.makedirs(out_dir, exist_ok=True)
results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "audit_status": "EXECUTED",
    "reproduced_metrics": {}
}

# -------------------------------------------------------------
# 1. SENSOR MOTION MODEL RE-EVALUATION
# -------------------------------------------------------------
print("--> Evaluating Sensor Motion Event Classifier...")
sensor_manifest = "data/datasets/manifests/sensor_real_manifest.csv"
if os.path.exists(sensor_manifest) and os.path.exists(SENSOR_MODEL) and os.path.exists(SENSOR_SCALER):
    import xgboost as xgb
    df_sensor = pd.read_csv(sensor_manifest)
    test_sensor = df_sensor[df_sensor["split"] == "test"].copy()
    label_map = {"normal_activity": 0, "fall": 1, "impact": 2}
    test_sensor["target"] = test_sensor["canonical_label"].map(label_map)
    test_sensor = test_sensor.dropna(subset=["target"])
    y_test_sensor = test_sensor["target"].astype(int).values
    
    from ml.data.prepare_sensor_real import FEATURE_COLS
    X_test_sensor = test_sensor[FEATURE_COLS].values
    
    scaler_sensor = joblib.load(SENSOR_SCALER)
    X_test_sensor_scaled = scaler_sensor.transform(X_test_sensor)
    
    model_sensor = xgb.XGBClassifier()
    model_sensor.load_model(SENSOR_MODEL)
    
    preds_sensor = model_sensor.predict(X_test_sensor_scaled)
    acc_sensor = float(accuracy_score(y_test_sensor, preds_sensor))
    f1_sensor = float(f1_score(y_test_sensor, preds_sensor, average="macro"))
    mcc_sensor = float(matthews_corrcoef(y_test_sensor, preds_sensor))
    cm_sensor = confusion_matrix(y_test_sensor, preds_sensor, labels=[0, 1, 2]).tolist()
    correct_sensor = int(sum(preds_sensor == y_test_sensor))
    
    print(f"    Sensor: {correct_sensor}/{len(y_test_sensor)} correct ({acc_sensor*100:.2f}%) | F1: {f1_sensor:.4f} | MCC: {mcc_sensor:+.4f}")
    results["reproduced_metrics"]["sensor_motion"] = {
        "dataset": "sisfall_uci_har_real",
        "n_test": len(y_test_sensor),
        "correct": correct_sensor,
        "accuracy": acc_sensor,
        "macro_f1": f1_sensor,
        "mcc": mcc_sensor,
        "confusion_matrix": cm_sensor,
        "impact_recall": float(cm_sensor[2][2] / sum(cm_sensor[2])) if sum(cm_sensor[2]) else 0.0
    }
else:
    print("    Sensor evaluation BLOCKED: missing files")
    results["reproduced_metrics"]["sensor_motion"] = {"status": "BLOCKED"}

# -------------------------------------------------------------
# 2. FULL MULTIMODAL XGBOOST RE-EVALUATION
# -------------------------------------------------------------
print("--> Evaluating Full Multimodal XGBoost Classifier...")
X, y = generate_multimodal_dataset(200, seed=42)
train_idx, val_idx, test_idx = get_leakage_free_splits(200)
X_test = X[test_idx]
y_test = y[test_idx]

xgb = XGBoostClassifier(model_path=XGB_CANONICAL)
preds_xgb = [xgb.predict(row)[0] for row in X_test]
acc_xgb = float(accuracy_score(y_test, preds_xgb))
f1_xgb = float(f1_score(y_test, preds_xgb, average="macro"))
mcc_xgb = float(matthews_corrcoef(y_test, preds_xgb))
cm_xgb = confusion_matrix(y_test, preds_xgb, labels=[0, 1, 2]).tolist()
correct_xgb = int(sum(np.array(preds_xgb) == y_test))

print(f"    XGBoost: {correct_xgb}/{len(y_test)} correct ({acc_xgb*100:.2f}%) | F1: {f1_xgb:.4f} | MCC: {mcc_xgb:+.4f}")
results["reproduced_metrics"]["xgboost_multimodal"] = {
    "n_test": len(y_test),
    "correct": correct_xgb,
    "accuracy": acc_xgb,
    "macro_f1": f1_xgb,
    "mcc": mcc_xgb,
    "confusion_matrix": cm_xgb
}

# -------------------------------------------------------------
# 3. PENNYLANE 4-QUBIT VQC RE-EVALUATION
# -------------------------------------------------------------
print("--> Evaluating PennyLane 4-Qubit VQC Classifier...")
vqc = VQCClassifier(model_dir=VQC_DIR)
preds_vqc = [vqc.predict(row)[0] for row in X_test]
acc_vqc = float(accuracy_score(y_test, preds_vqc))
f1_vqc = float(f1_score(y_test, preds_vqc, average="macro"))
mcc_vqc = float(matthews_corrcoef(y_test, preds_vqc))
cm_vqc = confusion_matrix(y_test, preds_vqc, labels=[0, 1, 2]).tolist()
correct_vqc = int(sum(np.array(preds_vqc) == y_test))

print(f"    VQC: {correct_vqc}/{len(y_test)} correct ({acc_vqc*100:.2f}%) | F1: {f1_vqc:.4f} | MCC: {mcc_vqc:+.4f}")
results["reproduced_metrics"]["vqc_quantum"] = {
    "n_test": len(y_test),
    "correct": correct_vqc,
    "accuracy": acc_vqc,
    "macro_f1": f1_vqc,
    "mcc": mcc_vqc,
    "confusion_matrix": cm_vqc
}

# -------------------------------------------------------------
# 4. 7 INDEPENDENT ABLATION RE-EVALUATIONS
# -------------------------------------------------------------
print("--> Evaluating 7 Independent Modality Ablation Models...")
import xgboost as xgb_lib
ablation_configs = [
    ("Vision Only", list(range(0, 6)), "xgb_vision_only.json"),
    ("Questionnaire Only", list(range(6, 18)), "xgb_questionnaire_only.json"),
    ("Sensor Only", list(range(18, 23)), "xgb_sensor_only.json"),
    ("Vision + Questionnaire", list(range(0, 18)), "xgb_vision_plus_questionnaire.json"),
    ("Vision + Sensor", list(range(0, 6)) + list(range(18, 23)), "xgb_vision_plus_sensor.json"),
    ("Questionnaire + Sensor", list(range(6, 23)), "xgb_questionnaire_plus_sensor.json"),
    ("Full Multimodal", list(range(0, 23)), "xgb_full_multimodal.json"),
]

results["reproduced_metrics"]["ablations"] = {}
for name, slice_idx, fname in ablation_configs:
    fpath = os.path.join("ml", "models", "ablation_independent", fname)
    if os.path.exists(fpath):
        m = xgb_lib.XGBClassifier()
        m.load_model(fpath)
        preds = m.predict(X_test[:, slice_idx])
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))
        mcc = float(matthews_corrcoef(y_test, preds))
        corr = int(sum(preds == y_test))
        print(f"    {name:<26}: {corr}/30 ({acc*100:.2f}%) | F1: {f1:.4f} | MCC: {mcc:+.4f}")
        results["reproduced_metrics"]["ablations"][name] = {
            "correct": corr, "accuracy": acc, "macro_f1": f1, "mcc": mcc
        }

# -------------------------------------------------------------
# 5. SAVE EVIDENCE OUTPUT
# -------------------------------------------------------------
with open(os.path.join(out_dir, "reproduced_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] All metrics saved to {os.path.join(out_dir, 'reproduced_metrics.json')}")
