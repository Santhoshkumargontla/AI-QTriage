import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.training.train_xgboost import generate_multimodal_dataset
from ml.models.canonical_paths import XGB_CANONICAL

# Load dataset and model
X, y = generate_multimodal_dataset(200, seed=42)
X_test, y_test = X[170:], y[170:]

xgb = XGBClassifier()
xgb.load_model(XGB_CANONICAL)

# Modality index masks:
# Vision: F0-F5 (indices 0..5)
# Quest:  F6-F17 (indices 6..17)
# Sensor: F18-F22 (indices 18..22)

def mask_features(X_data, keep_modalities):
    X_masked = np.zeros_like(X_data)
    if "vision" in keep_modalities:
        X_masked[:, 0:6] = X_data[:, 0:6]
    if "quest" in keep_modalities:
        X_masked[:, 6:18] = X_data[:, 6:18]
    if "sensor" in keep_modalities:
        X_masked[:, 18:23] = X_data[:, 18:23]
    return X_masked

configs = [
    ("Vision Only (F0–F5)", ["vision"]),
    ("Questionnaire Only (F6–F17)", ["quest"]),
    ("Sensor Only (F18–F22)", ["sensor"]),
    ("Vision + Questionnaire", ["vision", "quest"]),
    ("Vision + Sensor", ["vision", "sensor"]),
    ("Questionnaire + Sensor", ["quest", "sensor"]),
    ("Full Multimodal (F0–F22)", ["vision", "quest", "sensor"]),
]

results = []

for name, mods in configs:
    X_masked = mask_features(X_test, mods)
    preds = xgb.predict(X_masked)
    probs = xgb.predict_proba(X_masked)
    
    acc = float(accuracy_score(y_test, preds))
    prec, rec, f1, supp = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist()
    
    correct = int((preds == y_test).sum())
    
    results.append({
        "configuration": name,
        "modalities": mods,
        "correct": f"{correct} / {len(y_test)}",
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "mcc": round(mcc, 4),
        "confusion_matrix": cm
    })
    print(f"{name:30s} | Acc: {acc*100:5.2f}% ({correct}/{len(y_test)}) | F1: {f1:.4f} | MCC: {mcc:.4f}")

with open("data/results/canonical_ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nAblation results successfully written to data/results/canonical_ablation_results.json")
