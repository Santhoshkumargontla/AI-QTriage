"""Recompute EfficientNet kaggle-v1 held-out test metrics only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier

clf = EfficientNetV2Classifier()
classes = clf.classes
class_to_idx = {c: i for i, c in enumerate(classes)}
df = pd.read_csv(ROOT / "data/datasets/efficientnet_kaggle_v1/manifest.csv")
test_df = df[df["split"] == "test"]
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    img_path = ROOT / row["image_path"]
    if not img_path.exists():
        continue
    label = str(row["class"]).lower()
    raw = clf.predict_raw(np.array(Image.open(img_path).convert("RGB")))
    winner = str(raw["winner"]).lower().replace(" ", "_")
    if winner.startswith("ood"):
        winner = "ood_reject"
    if winner not in class_to_idx:
        continue
    y_true.append(class_to_idx[label])
    y_pred.append(class_to_idx[winner])

labels = list(range(len(classes)))
report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
out = {
    "n": len(y_true),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    "per_class": {c: report.get(c, {}) for c in classes},
    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
}
dest = ROOT / "scratch" / "effnet_kaggle_v1_recomputed_metrics.json"
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
