"""VERIFY-FIRST audit: metrics, hashes, OOD, YOLO, leakage. Never writes metrics files."""
import os
import sys
import json
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    brier_score_loss,
)

out = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ece_fn(y_probs, y_true, n_bins=5):
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = predictions == y_true
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = float(np.mean(in_bin))
        if prop > 0:
            ece += abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin]))) * prop
    return float(ece)


def multiclass_brier(probs, y_true, n_classes=3):
    y = np.asarray(y_true, dtype=int)
    onehot = np.zeros((len(y), n_classes), dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((np.asarray(probs) - onehot) ** 2, axis=1)))


print("=" * 70)
print("PHASE 1 LIVE METRICS")
print("=" * 70)
from ml.training.train_xgboost import generate_multimodal_dataset, compute_brier_and_ece
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.models.canonical_paths import XGB_CANONICAL, VQC_DIR, YOLO_CANONICAL, EFFNET_CANONICAL, UNET_CANONICAL

X, y = generate_multimodal_dataset(num_samples=200, seed=42)
X_train, y_train = X[:140], y[:140]
X_val, y_val = X[140:170], y[140:170]
X_test, y_test = X[170:], y[170:]
print("len(y_test)", len(y_test))
print("train/val/test", len(y_train), len(y_val), len(y_test))
print("index ranges train 0:140 val 140:170 test 170:200")

xgb = XGBoostClassifier(XGB_CANONICAL)
xgb_preds, xgb_probs = [], []
for row in X_test:
    idx, p = xgb.predict(row)
    xgb_preds.append(idx)
    xgb_probs.append(p)
xgb_preds = np.array(xgb_preds)
xgb_probs = np.array(xgb_probs)

vqc = VQCClassifier(VQC_DIR)
vqc_preds, vqc_probs = [], []
for row in X_test:
    idx, p = vqc.predict(row)
    vqc_preds.append(idx)
    vqc_probs.append(p)
vqc_preds = np.array(vqc_preds)
vqc_probs = np.array(vqc_probs)

print("len(y_pred_xgb)", len(xgb_preds))
print("len(y_pred_vqc)", len(vqc_preds))
assert len(y_test) == len(xgb_preds) == len(vqc_preds)

xgb_correct = int((xgb_preds == y_test).sum())
vqc_correct = int((vqc_preds == y_test).sum())
xgb_acc_ratio = xgb_correct / len(y_test)
vqc_acc_ratio = vqc_correct / len(y_test)
xgb_acc_sk = float(accuracy_score(y_test, xgb_preds))
vqc_acc_sk = float(accuracy_score(y_test, vqc_preds))
print("XGB CM\n", confusion_matrix(y_test, xgb_preds, labels=[0, 1, 2]))
print("VQC CM\n", confusion_matrix(y_test, vqc_preds, labels=[0, 1, 2]))
print("XGB correct", xgb_correct, "/", len(y_test), "ratio", xgb_acc_ratio, "sklearn", xgb_acc_sk)
print("VQC correct", vqc_correct, "/", len(y_test), "ratio", vqc_acc_ratio, "sklearn", vqc_acc_sk)
print("ratio==sklearn XGB", xgb_acc_ratio == xgb_acc_sk)
print("ratio==sklearn VQC", np.isclose(vqc_acc_ratio, vqc_acc_sk))
print("impossible 0.9444?", not np.isclose(xgb_acc_sk, 0.9444) and abs(xgb_acc_sk * 30 - round(xgb_acc_sk * 30)) < 1e-9)

brier_xgb_custom, ece_xgb_trainfn = compute_brier_and_ece(xgb_probs, y_test)
brier_vqc_custom, ece_vqc_trainfn = compute_brier_and_ece(vqc_probs, y_test)
brier_xgb_oh = multiclass_brier(xgb_probs, y_test)
brier_vqc_oh = multiclass_brier(vqc_probs, y_test)

metrics_live = {
    "test_sample_count": int(len(y_test)),
    "same_held_out_rows": True,
    "xgb_correct": f"{xgb_correct} / {len(y_test)}",
    "vqc_correct": f"{vqc_correct} / {len(y_test)}",
    "xgb_accuracy": round(xgb_acc_sk, 6),
    "vqc_accuracy": round(vqc_acc_sk, 6),
    "xgb_accuracy_matches_ratio": bool(np.isclose(xgb_acc_ratio, xgb_acc_sk)),
    "vqc_accuracy_matches_ratio": bool(np.isclose(vqc_acc_ratio, vqc_acc_sk)),
    "xgb_macro_precision": round(float(precision_score(y_test, xgb_preds, average="macro", zero_division=0)), 6),
    "vqc_macro_precision": round(float(precision_score(y_test, vqc_preds, average="macro", zero_division=0)), 6),
    "xgb_macro_recall": round(float(recall_score(y_test, xgb_preds, average="macro", zero_division=0)), 6),
    "vqc_macro_recall": round(float(recall_score(y_test, vqc_preds, average="macro", zero_division=0)), 6),
    "xgb_macro_f1": round(float(f1_score(y_test, xgb_preds, average="macro", zero_division=0)), 6),
    "vqc_macro_f1": round(float(f1_score(y_test, vqc_preds, average="macro", zero_division=0)), 6),
    "xgb_mcc": round(float(matthews_corrcoef(y_test, xgb_preds)), 6),
    "vqc_mcc": round(float(matthews_corrcoef(y_test, vqc_preds)), 6),
    "xgb_brier_onehot": round(brier_xgb_oh, 6),
    "vqc_brier_onehot": round(brier_vqc_oh, 6),
    "xgb_brier_train_helper": brier_xgb_custom,
    "vqc_brier_train_helper": brier_vqc_custom,
    "xgb_ece": round(ece_fn(xgb_probs, y_test), 6),
    "vqc_ece": round(ece_fn(vqc_probs, y_test), 6),
    "xgb_ece_train_helper": ece_xgb_trainfn,
    "vqc_ece_train_helper": ece_vqc_trainfn,
    "xgb_confusion_matrix": confusion_matrix(y_test, xgb_preds, labels=[0, 1, 2]).tolist(),
    "vqc_confusion_matrix": confusion_matrix(y_test, vqc_preds, labels=[0, 1, 2]).tolist(),
    "y_test": y_test.tolist(),
    "xgb_preds": xgb_preds.tolist(),
    "vqc_preds": vqc_preds.tolist(),
}
print(json.dumps({k: v for k, v in metrics_live.items() if k not in ("y_test", "xgb_preds", "vqc_preds")}, indent=2))
out["phase1"] = metrics_live

with open(os.path.join("data", "results", "vqc_xgb_comparison.json"), encoding="utf-8") as f:
    stored = json.load(f)
out["phase1_stored_vs_live"] = {
    "stored_xgb_acc": stored["xgboost"]["accuracy"],
    "live_xgb_acc": metrics_live["xgb_accuracy"],
    "xgb_match": stored["xgboost"]["accuracy"] == metrics_live["xgb_accuracy"],
    "stored_vqc_acc": stored["vqc"]["accuracy"],
    "live_vqc_acc": metrics_live["vqc_accuracy"],
    "vqc_match": stored["vqc"]["accuracy"] == metrics_live["vqc_accuracy"],
    "stored_xgb_correct": stored["xgboost"]["correct_predictions"],
    "live_xgb_correct": metrics_live["xgb_correct"],
}
print("STORED vs LIVE", out["phase1_stored_vs_live"])

print("=" * 70)
print("ARTIFACT HASHES")
print("=" * 70)
paths = {
    "yolo": YOLO_CANONICAL,
    "effnet": EFFNET_CANONICAL,
    "unet": UNET_CANONICAL,
    "xgb": XGB_CANONICAL,
    "vqc": os.path.join(VQC_DIR, "vqc_weights.npz"),
}
hashes = {k: {"path": v, "exists": os.path.exists(v), "sha256": sha256(v) if os.path.exists(v) else None} for k, v in paths.items()}
print(json.dumps(hashes, indent=2))
out["hashes"] = hashes

print("=" * 70)
print("YOLO RUNTIME")
print("=" * 70)
from ml.vision.yolo_wrapper import YOLO11Detector
det = YOLO11Detector()
info = det.get_info()
print(json.dumps(info, indent=2, default=str))
out["yolo"] = info
demo = os.path.join("data", "sample", "image", "football_injury.jpg")
if os.path.exists(demo) and det.model is not None:
    sweeps = {}
    for conf in (0.01, 0.05, 0.10, 0.25, 0.40, 0.50):
        raw = det.model(demo, conf=conf, verbose=False)[0]
        boxes = []
        for b in raw.boxes:
            boxes.append({
                "cls": int(b.cls[0]),
                "name": det.model.names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            })
        keep = det.detect(demo) if conf == det.infer_conf else None
        sweeps[str(conf)] = {"n": len(boxes), "boxes": boxes}
    print("DEMO SWEEP", json.dumps({k: {"n": v["n"], "confs": [b["conf"] for b in v["boxes"]]} for k, v in sweeps.items()}, indent=2))
    out["yolo_demo_sweep"] = {k: {"n": v["n"], "boxes": v["boxes"]} for k, v in sweeps.items()}
    out["yolo_detect_default"] = det.detect(demo)

print("=" * 70)
print("EFFNET / UNET RAW OOD")
print("=" * 70)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
import cv2

eff = EfficientNetV2Classifier()
unet = UNetSegmenter()
ood = {}
for name, arr in [
    ("gray", np.full((256, 256, 3), 128, np.uint8)),
    ("black", np.zeros((256, 256, 3), np.uint8)),
    ("white", np.full((256, 256, 3), 255, np.uint8)),
]:
    raw_p = eff.predict(arr)
    gated = None
    try:
        from ml.vision.efficientnet_wrapper import interpret_prediction
        gated = interpret_prediction(raw_p)
    except Exception as e:
        gated = {"error": str(e)}
    raw_mask = unet.predict_raw(arr) if hasattr(unet, "predict_raw") else None
    mask, pc, ratio, dbg = unet.segment(arr)
    ood[name] = {
        "effnet_raw": {k: float(v) for k, v in raw_p.items()} if isinstance(raw_p, dict) else raw_p,
        "effnet_gated": gated,
        "unet_display_positive_ratio": float(ratio),
        "unet_debug": {k: dbg.get(k) for k in ("status", "reason", "false_positive_area", "raw_positive_ratio", "mask_withheld") if isinstance(dbg, dict)},
    }
print(json.dumps(ood, indent=2, default=str))
out["ood"] = ood

print("=" * 70)
print("UNET MASK GEOMETRY SIM")
print("=" * 70)
h, w = 400, 700
full = np.zeros((h, w, 3), np.uint8)
bbox = [100, 50, 300, 220]
roi_h, roi_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
mask, pc, ratio, dbg = unet.segment(full, bbox)
out["unet_geometry"] = {
    "original": [h, w],
    "bbox": bbox,
    "roi_hw": [roi_h, roi_w],
    "returned_mask_shape": list(mask.shape) if mask is not None else None,
    "mask_equals_roi": list(mask.shape) == [roi_h, roi_w] if mask is not None else None,
    "mask_equals_full": list(mask.shape) == [h, w] if mask is not None else None,
    "debug_roi": {k: dbg.get(k) for k in ("preprocessed_roi_shape", "input_shape") if isinstance(dbg, dict)},
}
print(json.dumps(out["unet_geometry"], indent=2))

print("=" * 70)
print("DATASET MANIFESTS")
print("=" * 70)


def split_subjects(path, subj_col="subject_id"):
    import pandas as pd
    if not os.path.exists(path):
        return {"missing": path}
    df = pd.read_csv(path)
    res = {"n": len(df), "cols": list(df.columns)}
    if "split" in df.columns:
        res["split_counts"] = df["split"].value_counts().to_dict()
    if subj_col in df.columns:
        sets = {s: set(df[df["split"] == s][subj_col].astype(str)) for s in ("train", "val", "test") if "split" in df.columns}
        res["subject_counts"] = {k: len(v) for k, v in sets.items()}
        res["overlap_train_val"] = sorted(sets.get("train", set()) & sets.get("val", set()))
        res["overlap_train_test"] = sorted(sets.get("train", set()) & sets.get("test", set()))
        res["overlap_val_test"] = sorted(sets.get("val", set()) & sets.get("test", set()))
    return res

out["manifests"] = {
    "effnet": split_subjects("data/datasets/efficientnet_processed/manifest.csv"),
    "unet": split_subjects("data/datasets/unet_processed/manifest.csv"),
    "sensor": split_subjects("data/datasets/manifests/sensor_manifest.csv"),
}
print(json.dumps(out["manifests"], indent=2, default=str))

dest = os.path.join("scratch", "strict_verify_phase014.json")
# drop huge pred lists from file? keep them for proof
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("WROTE", dest)
