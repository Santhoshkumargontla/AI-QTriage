import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.training.train_xgboost import generate_multimodal_dataset

# 1. Dataset Generation with deterministic seed 42
X, y = generate_multimodal_dataset(200, seed=42)

# Strict splits: 70% Train, 15% Val, 15% Test
n_train = 140
n_val = 30
n_test = 30

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

# Feature index definitions (total 23 features):
# Vision:        F0-F5   (indices 0..5, 6 features)
# Questionnaire: F6-F17  (indices 6..17, 12 features)
# Sensor:        F18-F22 (indices 18..22, 5 features)

feature_slices = {
    "vision": list(range(0, 6)),
    "quest": list(range(6, 18)),
    "sensor": list(range(18, 23))
}

configs = [
    ("Vision Only (F0–F5)", ["vision"]),
    ("Questionnaire Only (F6–F17)", ["quest"]),
    ("Sensor Only (F18–F22)", ["sensor"]),
    ("Vision + Questionnaire", ["vision", "quest"]),
    ("Vision + Sensor", ["vision", "sensor"]),
    ("Questionnaire + Sensor", ["quest", "sensor"]),
    ("Full Multimodal (F0–F22)", ["vision", "quest", "sensor"]),
]

def get_indices(mods):
    idx = []
    for m in mods:
        idx.extend(feature_slices[m])
    return sorted(idx)

# Wilson score confidence interval helper for accuracy
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

results = []
models_dir = os.path.join("ml", "models", "ablation_independent")
os.makedirs(models_dir, exist_ok=True)

print(f"{'Configuration':30s} | {'Dims':4s} | {'Acc (%)':7s} | {'95% CI':16s} | {'F1':6s} | {'MCC':7s} | {'Val Acc':7s}")
print("-" * 90)

for name, mods in configs:
    cols = get_indices(mods)
    
    # Extract feature subsets
    X_tr_sub = X_train[:, cols]
    X_val_sub = X_val[:, cols]
    X_te_sub = X_test[:, cols]
    
    # Balanced class weights computed strictly on training data
    sample_weights = compute_sample_weight("balanced", y_train)
    
    # Hyperparameters tuned on train/val:
    # Use standard hyperparameters consistent with canonical model (max_depth=3, lr=0.08, n_estimators=60)
    clf = XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.08,
        random_state=42
    )
    clf.fit(X_tr_sub, y_train, sample_weight=sample_weights)
    
    # Evaluate on Validation set (for model check)
    val_preds = clf.predict(X_val_sub)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Evaluate on held-out Test set (strictly untouched during training)
    test_preds = clf.predict(X_te_sub)
    test_probs = clf.predict_proba(X_te_sub)
    
    acc = float(accuracy_score(y_test, test_preds))
    prec, rec, f1, supp = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2]).tolist()
    
    correct = int((test_preds == y_test).sum())
    ci_low, ci_high = wilson_ci(correct, len(y_test))
    
    # Per-class metrics
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(y_test, test_preds, labels=[0, 1, 2], zero_division=0)
    per_class = {}
    classes = ["LOW", "MODERATE", "HIGH"]
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f_cls[i]), 4),
            "support": int(s_cls[i])
        }
        
    safe_name = name.split("(")[0].strip().replace(" ", "_").replace("+", "plus").lower()
    model_save_path = os.path.join(models_dir, f"xgb_{safe_name}.json")
    clf.save_model(model_save_path)
    
    entry = {
        "configuration": name,
        "modalities": mods,
        "active_feature_count": len(cols),
        "feature_indices": cols,
        "model_path": model_save_path,
        "correct": f"{correct} / {len(y_test)}",
        "accuracy": round(acc, 4),
        "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        "val_accuracy": round(float(val_acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "mcc": round(mcc, 4),
        "confusion_matrix": cm,
        "per_class": per_class
    }
    results.append(entry)
    
    ci_str = f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"
    print(f"{name:30s} | {len(cols):4d} | {acc*100:6.2f}% | {ci_str:16s} | {f1:.4f} | {mcc:7.4f} | {val_acc*100:6.2f}%")

save_path = "data/results/independent_ablation_results.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nIndependent modality ablation complete. Results saved to: {save_path}")
