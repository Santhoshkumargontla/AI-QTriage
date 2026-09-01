"""HISTORICAL / ALTERNATE PATH — synthetic drawing EfficientNet trainer.

The CURRENT canonical runtime checkpoint is the Kaggle 8-class model
(ml/models/vision/efficientnetv2_injury_best.pt, version kaggle-v1).
Retrain/promote that head with:
  backend\\venv\\Scripts\\python.exe ml\\training\\train_efficientnet_kaggle_v1.py

This script trains on efficientnet_processed *drawings* and may overwrite
EFFNET_CANONICAL only if its own promotion gates pass. Prefer the Kaggle
trainer for production/research-demo weights.

Metrics are computed from the training loop and held-out predictions.
Blank/OOD injury-class collapse => not trustworthy, no promotion.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    EFFNET_CANDIDATE,
    EFFNET_CANDIDATE_DIR,
    EFFNET_CANONICAL,
    EFFNET_PROCESSED_MANIFEST,
    EFFNET_PROCESSED_ROOT,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact
from ml.training import prepare_efficientnet_processed_dataset as prep
from ml.training.prepare_efficientnet_processed_dataset import build as build_dataset

METADATA_SAVE_PATH = os.path.join("ml", "models", "vision", "efficientnetv2_metadata.json")
HISTORY_CSV = os.path.join(EFFNET_CANDIDATE_DIR, "results.csv")
MAX_EPOCHS = 16
PATIENCE = 5
BATCH_SIZE = 8
LR = 1e-3
SEED = 42
INJURY_CLASSES = {"cut", "bruise", "swelling"}
COLLAPSE_MAX = 0.95
CLASSES: list[str] = ["cut", "bruise", "swelling"]
OOD_WATCH = ("gray", "black", "white", "noisy_gray", "uniform_skin", "blank_skin", "dummy_test", "blue", "noise")


class WoundImageDataset(Dataset):
    def __init__(self, rows, image_size=224):
        self.rows = rows
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row["image_path"].replace("/", os.sep)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        tensor = img.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)
        label = CLASSES.index(row["class"])
        return tensor, label


def _load_split(split: str) -> list[dict]:
    rows = []
    with open(EFFNET_PROCESSED_MANIFEST, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["class"] in CLASSES and os.path.exists(row["image_path"].replace("/", os.sep)):
                rows.append(row)
    return rows


def _tensor_from_rgb(img_rgb: np.ndarray, device):
    img = cv2.resize(img_rgb, (224, 224))
    tensor = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy((tensor - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _probe(model, device) -> dict:
    model.eval()
    rng = np.random.default_rng(0)
    probes = {}
    specs = {
        "gray": np.full((224, 224, 3), 180, dtype=np.uint8),
        "black": np.zeros((224, 224, 3), dtype=np.uint8),
        "white": np.full((224, 224, 3), 255, dtype=np.uint8),
        "noisy_gray": np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8),
        "uniform_skin": np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8),
        "blue": np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8),
        "noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    cut = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(cut, (70, 40), (150, 180), (190, 20, 20), 7)
    specs["blurred_cut"] = cv2.GaussianBlur(cut, (31, 31), 8)
    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank):
        specs["blank_skin"] = cv2.cvtColor(cv2.imread(blank), cv2.COLOR_BGR2RGB)
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        specs["football_injury"] = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.exists(dummy):
        specs["dummy_test"] = cv2.cvtColor(cv2.imread(dummy), cv2.COLOR_BGR2RGB)
    qa = os.path.join("data", "sample", "qa", "qa_swelling_offcenter.jpg")
    if os.path.exists(qa):
        specs["qa_swelling_offcenter"] = cv2.cvtColor(cv2.imread(qa), cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        for name, img in specs.items():
            if img is None:
                continue
            prob = torch.softmax(model(_tensor_from_rgb(img, device)), dim=1).squeeze().cpu().numpy()
            arg = CLASSES[int(prob.argmax())]
            row = {CLASSES[i]: round(float(prob[i]), 4) for i in range(len(CLASSES))}
            row["max"] = round(float(prob.max()), 4)
            row["argmax"] = arg
            row["injury_class_predicted"] = arg in INJURY_CLASSES
            probes[name] = row
    return probes


def _injury_collapse(probes: dict) -> bool:
    return _collapse_count(probes) > 0


def _collapse_count(probes: dict) -> int:
    n = 0
    for name in OOD_WATCH:
        row = probes.get(name)
        if not row:
            continue
        if row.get("injury_class_predicted") and float(row.get("max") or 0) >= COLLAPSE_MAX:
            n += 1
    return n


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    if len(labels) == 0:
        return None
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float((pred[mask] == labels[mask]).mean())
        mean_conf = float(conf[mask].mean())
        ece += abs(acc - mean_conf) * (float(mask.sum()) / n)
    return round(float(ece), 6)


def _evaluate(model, loader, criterion, device):
    model.eval()
    ys, preds = [], []
    running = 0.0
    n = 0
    prob_rows = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred)
            prob_rows.append(prob)
    ys_a = np.array(ys)
    preds_a = np.array(preds)
    acc = float(accuracy_score(ys_a, preds_a)) if len(ys_a) else 0.0
    prec, rec, f1, support = precision_recall_fscore_support(
        ys_a, preds_a, labels=list(range(len(CLASSES))), zero_division=0
    )
    cm = confusion_matrix(ys_a, preds_a, labels=list(range(len(CLASSES)))).tolist()
    probs = np.vstack(prob_rows) if prob_rows else np.zeros((0, len(CLASSES)))
    per_class = {
        CLASSES[i]: {
            "precision": round(float(prec[i]), 6),
            "recall": round(float(rec[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i in range(len(CLASSES))
    }
    return {
        "n": int(len(ys_a)),
        "loss": round(running / max(n, 1), 6),
        "accuracy": round(acc, 6),
        "macro_precision": round(float(np.mean(prec)), 6) if len(prec) else 0.0,
        "macro_recall": round(float(np.mean(rec)), 6) if len(rec) else 0.0,
        "macro_f1": round(float(np.mean(f1)), 6) if len(f1) else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(CLASSES),
        "ece_10bin": _ece(probs, ys_a),
        "mean_max_softmax": None if len(probs) == 0 else round(float(probs.max(axis=1).mean()), 6),
        "predictions": preds_a.tolist(),
        "labels": ys_a.tolist(),
    }


def _baseline_probes():
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    clf = EfficientNetV2Classifier()
    if not clf.is_loaded:
        return {}
    rng = np.random.default_rng(0)
    specs = {
        "gray": np.full((224, 224, 3), 180, dtype=np.uint8),
        "black": np.zeros((224, 224, 3), dtype=np.uint8),
        "white": np.full((224, 224, 3), 255, dtype=np.uint8),
        "noisy_gray": np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8),
        "uniform_skin": np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8),
        "blue": np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8),
        "noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank):
        specs["blank_skin"] = cv2.cvtColor(cv2.imread(blank), cv2.COLOR_BGR2RGB)
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.exists(dummy):
        specs["dummy_test"] = cv2.cvtColor(cv2.imread(dummy), cv2.COLOR_BGR2RGB)
    qa = os.path.join("data", "sample", "qa", "qa_swelling_offcenter.jpg")
    if os.path.exists(qa):
        specs["qa_swelling_offcenter"] = cv2.cvtColor(cv2.imread(qa), cv2.COLOR_BGR2RGB)
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        specs["football_injury"] = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    out = {}
    for name, img in specs.items():
        if img is None:
            continue
        raw = clf.predict_raw(img, temperature=1.0)
        arg = str(raw["winner"]).lower()
        row = {k.lower(): round(float(v), 4) for k, v in raw["probs"].items()}
        row["max"] = round(float(raw["max_prob"]), 4)
        row["argmax"] = arg
        row["injury_class_predicted"] = arg in INJURY_CLASSES
        out[name] = row
    return out


def train_efficientnet(max_epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR, patience: int = PATIENCE):
    global CLASSES
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    build_dataset()
    CLASSES = list(prep.CLASSES)
    print(f"Taxonomy after honest unique-hash build: {CLASSES}")

    baseline_probes = _baseline_probes()
    baseline_collapse_n = _collapse_count(baseline_probes)
    production_sha_before = sha256_file(EFFNET_CANONICAL) if os.path.exists(EFFNET_CANONICAL) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"EfficientNetV2 REAL training on {device}  early_stopping patience={patience} max_epochs={max_epochs}")

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    train_counts = {c: sum(1 for r in train_rows if r["class"] == c) for c in CLASSES}
    missing_train = [c for c, n in train_counts.items() if n == 0]
    if missing_train:
        print(f"WARNING missing train classes: {missing_train}")

    train_loader = DataLoader(WoundImageDataset(train_rows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WoundImageDataset(val_rows), batch_size=batch_size)
    test_loader = DataLoader(WoundImageDataset(test_rows), batch_size=batch_size)

    import timm
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=len(CLASSES))
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name) or ("head" in name)
    model.to(device)

    counts = np.array([max(train_counts[c], 1) for c in CLASSES], dtype=np.float32)
    weights = torch.tensor(counts.sum() / counts, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))
        train_loss = running / max(n, 1)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{max_epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] + 1e-6 < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}; best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    val_best = _evaluate(model, val_loader, criterion, device)
    test_metrics = _evaluate(model, test_loader, criterion, device)
    probes = _probe(model, device)
    collapsed = _injury_collapse(probes)
    train_loss_decreased = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]

    os.makedirs(EFFNET_CANDIDATE_DIR, exist_ok=True)
    torch.save(model.state_dict(), EFFNET_CANDIDATE)
    classes_path = os.path.join(EFFNET_CANDIDATE_DIR, "efficientnet_classes.json")
    with open(classes_path, "w", encoding="utf-8") as handle:
        json.dump(CLASSES, handle, indent=2)
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)

    candidate_sha = sha256_file(EFFNET_CANDIDATE)
    overlap_ok = True
    with open(os.path.join(EFFNET_PROCESSED_ROOT, "processing_summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]
    overlap_ok = bool(summary.get("leakage_free"))

    candidate_collapse_n = _collapse_count(probes)
    ood_improved = candidate_collapse_n < baseline_collapse_n
    test_support_ok = all(int(test_metrics["per_class"][c]["support"]) > 0 for c in CLASSES)
    train_ok = all(train_counts[c] > 0 for c in CLASSES)
    copied_production = bool(production_sha_before) and candidate_sha == production_sha_before

    # A 1-count drop while remaining blanks still dump onto an injury class is not
    # a safety improvement. Accuracy 1.0 on unique drawings is also not sufficient.
    promote = (
        overlap_ok
        and train_loss_decreased
        and train_ok
        and test_metrics["n"] > 0
        and test_support_ok
        and ood_improved
        and not collapsed
        and not copied_production
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not overlap_ok:
        reasons.append("SPLIT_HASH_LEAKAGE")
    if not ood_improved:
        reasons.append(
            f"OOD_INJURY_COLLAPSE_NOT_IMPROVED baseline={baseline_collapse_n} candidate={candidate_collapse_n}"
        )
    if collapsed:
        reasons.append("CANDIDATE_STILL_COLLAPSES_OOD_TO_INJURY")
    if not train_loss_decreased:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if not train_ok:
        reasons.append("CLASS_HAS_NO_TRAINING_EXAMPLES")
    if not test_support_ok:
        reasons.append("CLASS_HAS_NO_HELD_OUT_SUPPORT")
    if copied_production:
        reasons.append("CANDIDATE_SHA_EQUALS_PRODUCTION")
    if promote:
        reasons.append(
            f"OOD_COLLAPSE_IMPROVED {baseline_collapse_n}->{candidate_collapse_n}; "
            "status remains NOT_TRUSTWORTHY (synthetic drawings; no trained reject class)"
        )

    promoted = False
    if promote:
        os.makedirs(os.path.dirname(EFFNET_CANONICAL), exist_ok=True)
        backup = EFFNET_CANONICAL + ".pre_processed_retrain_backup"
        if os.path.exists(EFFNET_CANONICAL) and not os.path.exists(backup):
            shutil.copy2(EFFNET_CANONICAL, backup)
        shutil.copy2(EFFNET_CANDIDATE, EFFNET_CANONICAL)
        shutil.copy2(classes_path, os.path.join("ml", "models", "vision", "efficientnetv2_injury_best_classes.json"))
        promoted = True

    production_sha_after = sha256_file(EFFNET_CANONICAL) if os.path.exists(EFFNET_CANONICAL) else None
    # Synthetic drawings + no legitimate negative training class: never clear this.
    status = "NOT_TRUSTWORTHY"
    known_limitations = list(summary.get("known_limitations") or [])
    known_limitations.extend([
        "Not clinically validated.",
        "Do not treat held-out accuracy as generalization to real injuries.",
        "No normal/reject class was trained (only two existing no-injury files; not fabricated).",
        "Closed-set softmax is not an OOD detector. Input-quality gates and confidence/margin/entropy rejection remain required.",
        "Gates do not certify photographs of real injuries.",
    ])
    if collapsed:
        known_limitations.append("Candidate still assigns an injury class at >=0.95 on blank/uniform/OOD probes.")

    def _slim_eval(block):
        return {k: block[k] for k in (
            "n", "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "per_class", "confusion_matrix", "confusion_matrix_labels", "ece_10bin", "mean_max_softmax",
        )}

    metrics = {
        "dataset_name": "efficientnet_processed",
        "dataset_type": "synthetic_drawings_unique_hash_no_fabricated_negatives",
        "dataset_size": summary.get("n"),
        "classes": CLASSES,
        "trained_normal_class": False,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_class_counts": train_counts,
        "class_counts_by_split": summary.get("class_counts_by_split"),
        "split_file": os.path.join(EFFNET_PROCESSED_ROOT, "split.csv").replace("\\", "/"),
        "max_epochs": max_epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "early_stopping_monitor": "val_loss",
        "optimizer": "Adam",
        "lr": lr,
        "batch_size": batch_size,
        "seed": SEED,
        "loss": "CrossEntropyLoss_class_weighted",
        "backbone": "tf_efficientnetv2_s.in21k_ft_in1k pretrained, classifier-head only",
        "device": device,
        "val_best_loss": round(best_val_loss, 6),
        "val": _slim_eval(val_best),
        "test": _slim_eval(test_metrics),
        "baseline_ood_probes": baseline_probes,
        "candidate_ood_probes": probes,
        "ood_watch_list": list(OOD_WATCH),
        "ood_collapse_max": COLLAPSE_MAX,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "ood_safety_improved": ood_improved,
        "blank_ood_injury_collapse": collapsed,
        "training_history": history,
        "metrics_source": "computed_from_training_loop_and_held_out_predictions",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "leakage_free": overlap_ok,
        "known_limitations": known_limitations,
        "not_clinical_accuracy": True,
        "did_not_fabricate_negatives": True,
    }
    metadata = {
        "model_name": "EfficientNetV2 Classification",
        "version": "v1.4.0",
        "status": status,
        "training_status": status,
        "readiness_status": status,
        "device": device,
        "classes": CLASSES,
        "training_date": metrics["trained_at"],
        "training_configuration": {
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "seed": SEED,
            "device": device,
            "init": "timm tf_efficientnetv2_s.in21k_ft_in1k pretrained, head only",
        },
        "metrics": metrics,
        "weights_path": EFFNET_CANONICAL if promoted else EFFNET_CANDIDATE,
        "candidate_path": EFFNET_CANDIDATE,
        "candidate_sha256": candidate_sha,
        "production_sha256": production_sha_after,
        "production_unchanged": production_sha_before == production_sha_after,
        "promotion": {
            "recommendation": recommendation,
            "reasons": reasons,
            "promoted_to_production": promoted,
            "rule": "Promote only if hash-disjoint splits, train loss decreased, held-out support for every class, candidate SHA differs from production, OOD injury-collapse count is strictly lower than baseline, AND remaining collapse count is 0. A small count drop while blanks still collapse is not promotion. Accuracy alone is not sufficient. Status stays NOT_TRUSTWORTHY.",
        },
        "training_was_real": True,
        "clinically_validated": False,
        "known_limitations": known_limitations,
        "data_provenance_class": "SYNTHETIC",
        "dataset_provenance": "efficientnet_processed unique-hash synthetic drawings. No fabricated negatives. Not clinical photography.",
    }

    os.makedirs(EFFNET_CANDIDATE_DIR, exist_ok=True)
    with open(os.path.join(EFFNET_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    pointer = {
        "latest_candidate": EFFNET_CANDIDATE.replace("\\", "/"),
        "latest_report": os.path.join(EFFNET_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json").replace("\\", "/"),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "ood_safety_improved": ood_improved,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "production_sha256": production_sha_after,
        "candidate_sha256": candidate_sha,
        "status_remains": "NOT_TRUSTWORTHY",
        "production_metadata_unchanged": not promoted,
    }
    with open(os.path.join(EFFNET_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump(pointer, handle, indent=2)
    if promoted:
        with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        register_model_artifact(
            model_name="EfficientNetV2 Classification",
            version="v1.4.0",
            artifact_path=EFFNET_CANONICAL,
            training_dataset="efficientnet_processed",
            sample_count=len(train_rows) + len(val_rows) + len(test_rows),
            classes=CLASSES,
            metrics=metrics,
            training_command=r"backend\venv\Scripts\python.exe ml\training\train_efficientnet.py",
            random_seed=SEED,
            notes="Real head-only training on unique-hash synthetic drawings. NOT_TRUSTWORTHY. Promoted only because OOD injury-collapse count improved versus baseline.",
        )
    print(f"RECOMMENDATION {recommendation}")
    print(f"STATUS {status} baseline_collapse={baseline_collapse_n} candidate_collapse={candidate_collapse_n} promoted={promoted}")
    print(f"test_acc={test_metrics['accuracy']} val_loss={val_best['loss']} ece={test_metrics.get('ece_10bin')}")
    return metadata


if __name__ == "__main__":
    train_efficientnet()
