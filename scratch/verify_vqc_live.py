import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.classifiers.vqc_classifier import VQCClassifier
from ml.training.train_xgboost import generate_multimodal_dataset
from ml.models.canonical_paths import VQC_DIR

X, y = generate_multimodal_dataset(200, seed=42)
X_test, y_test = X[170:], y[170:]

vqc = VQCClassifier()
loaded = vqc.load_model(VQC_DIR)
print(f"VQC weights loaded: {loaded}", flush=True)

preds, probs = [], []
for row in X_test:
    idx, p = vqc.predict(row)
    preds.append(idx)
    probs.append(p)

preds = np.array(preds)
correct = int((preds == y_test).sum())
print(f"VQC live test accuracy: {correct}/{len(y_test)} = {correct/len(y_test):.4f}", flush=True)
