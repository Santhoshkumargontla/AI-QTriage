"""Train EfficientNet on subject-aware cut/bruise + REAL normal patches.

Promotion only if OOD injury-class collapse count reaches 0.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    EFFNET_SUBJECT_CANDIDATE,
    EFFNET_SUBJECT_CANDIDATE_DIR,
    EFFNET_SUBJECT_MANIFEST,
    EFFNET_SUBJECT_ROOT,
    ROOT,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact
from ml.training.prepare_efficientnet_subject_normal import build as build_dataset

SEED = 42
MAX_EPOCHS = 14
PATIENCE = 5
BATCH_SIZE = 8
LR = 2e-4
CLASSES = ["cut", "bruise", "normal"]
INJURY_CLASSES = {"cut", "bruise", "swelling"}
COLLAPSE_MAX = 0.95
OOD_WATCH = ("gray", "black", "white", "noisy_gray", "uniform_skin", "blank_skin", "dummy_test", "blue", "noise")


class InjuryNormalDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment and self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        img = cv2.resize(img, (224, 224))
        tensor = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(tensor).permute(2, 0, 1), CLASSES.index(row["class"])


def _load_split(split: str):
    rows = []
    with open(os.path.join(ROOT, EFFNET_SUBJECT_MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["class"] in CLASSES:
                rows.append(row)
    return rows


def _tensor(img_rgb, device):
    img = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy((img - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)


def _probe(model, device):
    model.eval()
    rng = np.random.default_rng(0)
    specs = {
        "gray": np.full((224, 224, 3), 180, np.uint8),
        "black": np.zeros((224, 224, 3), np.uint8),
        "white": np.full((224, 224, 3), 255, np.uint8),
        "noisy_gray": np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8),
        "uniform_skin": np.full((224, 224, 3), (185, 145, 125), np.uint8),
        "blue": np.full((224, 224, 3), (20, 60, 200), np.uint8),
        "noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    for key, rel in (
        ("blank_skin", "data/datasets/yolo_injury/blank_skin.jpg"),
        ("dummy_test", "data/datasets/yolo_injury/dummy_test.jpg"),
        ("football_injury", "data/sample/image/football_injury.jpg"),
    ):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            specs[key] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    out = {}
    with torch.no_grad():
        for name, img in specs.items():
            prob = torch.softmax(model(_tensor(img, device)), dim=1).squeeze().cpu().numpy()
            arg = CLASSES[int(prob.argmax())]
            row = {CLASSES[i]: round(float(prob[i]), 4) for i in range(len(CLASSES))}
            row.update({"max": round(float(prob.max()), 4), "argmax": arg, "injury_class_predicted": arg in INJURY_CLASSES})
            out[name] = row
    return out


def _collapse_count(probes):
    return sum(
        1
        for name in OOD_WATCH
        if probes.get(name) and probes[name].get("injury_class_predicted") and float(probes[name].get("max") or 0) >= COLLAPSE_MAX
    )


def _baseline_probes():
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    clf = EfficientNetV2Classifier()
    if not clf.is_loaded:
        return {}
    rng = np.random.default_rng(0)
    specs = {
        "gray": np.full((224, 224, 3), 180, np.uint8),
        "black": np.zeros((224, 224, 3), np.uint8),
        "white": np.full((224, 224, 3), 255, np.uint8),
        "noisy_gray": np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8),
        "uniform_skin": np.full((224, 224, 3), (185, 145, 125), np.uint8),
        "blue": np.full((224, 224, 3), (20, 60, 200), np.uint8),
        "noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    for key, rel in (
        ("blank_skin", "data/datasets/yolo_injury/blank_skin.jpg"),
        ("dummy_test", "data/datasets/yolo_injury/dummy_test.jpg"),
    ):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            specs[key] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    out = {}
    for name, img in specs.items():
        raw = clf.predict_raw(img, temperature=1.0)
        arg = str(raw["winner"]).lower()
        row = {k.lower(): round(float(v), 4) for k, v in raw["probs"].items()}
        row.update({"max": round(float(raw["max_prob"]), 4), "argmax": arg, "injury_class_predicted": arg in INJURY_CLASSES})
        out[name] = row
    return out


def _evaluate(model, loader, criterion, device):
    model.eval()
    ys, preds = [], []
    running = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            running += float(criterion(logits, y).item()) * x.size(0)
            n += int(x.size(0))
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
    ys_a, preds_a = np.array(ys), np.array(preds)
    prec, rec, f1, support = precision_recall_fscore_support(ys_a, preds_a, labels=list(range(len(CLASSES))), zero_division=0)
    return {
        "n": int(len(ys_a)),
        "loss": round(running / max(n, 1), 6),
        "accuracy": round(float(accuracy_score(ys_a, preds_a)) if len(ys_a) else 0.0, 6),
        "macro_f1": round(float(np.mean(f1)), 6),
        "per_class": {
            CLASSES[i]: {"precision": round(float(prec[i]), 6), "recall": round(float(rec[i]), 6), "f1": round(float(f1[i]), 6), "support": int(support[i])}
            for i in range(len(CLASSES))
        },
        "confusion_matrix": confusion_matrix(ys_a, preds_a, labels=list(range(len(CLASSES)))).tolist(),
    }


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    summary = build_dataset()
    baseline = _baseline_probes()
    baseline_n = _collapse_count(baseline)
    production_abs = os.path.join(ROOT, EFFNET_CANONICAL)
    production_sha_before = sha256_file(production_abs) if os.path.exists(production_abs) else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    train_counts = {c: sum(1 for r in train_rows if r["class"] == c) for c in CLASSES}
    print("train_counts", train_counts, "baseline_collapse", baseline_n)
    weights = [1.0 / max(train_counts[r["class"]], 1) for r in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(InjuryNormalDataset(train_rows, augment=True), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(InjuryNormalDataset(val_rows), batch_size=BATCH_SIZE)
    test_loader = DataLoader(InjuryNormalDataset(test_rows), batch_size=BATCH_SIZE)

    import timm
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=len(CLASSES))
    for name, param in model.named_parameters():
        param.requires_grad = (
            ("classifier" in name)
            or ("head" in name)
            or ("blocks.5" in name)
            or ("blocks.4" in name)
            or ("blocks.3" in name)
        )
    model.to(device)
    counts = np.array([max(train_counts[c], 1) for c in CLASSES], np.float32)
    cw = torch.tensor(counts.sum() / counts, dtype=torch.float32, device=device)
    cw = cw / cw.mean()
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))
        val_m = _evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": round(running / max(n, 1), 6), "val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"]})
        print(f"epoch {epoch}/{MAX_EPOCHS} train_loss={history[-1]['train_loss']:.4f} val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f}")
        if val_m["loss"] + 1e-6 < best_val:
            best_val = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop @ {epoch}; best {best_epoch}")
                break

    model.load_state_dict(best_state)
    val_best = _evaluate(model, val_loader, criterion, device)
    test_m = _evaluate(model, test_loader, criterion, device)
    probes = _probe(model, device)
    cand_n = _collapse_count(probes)
    train_ok = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]

    os.makedirs(os.path.join(ROOT, EFFNET_SUBJECT_CANDIDATE_DIR), exist_ok=True)
    cand_abs = os.path.join(ROOT, EFFNET_SUBJECT_CANDIDATE)
    torch.save(model.state_dict(), cand_abs)
    classes_path = os.path.join(ROOT, EFFNET_SUBJECT_CANDIDATE_DIR, "efficientnet_classes.json")
    with open(classes_path, "w", encoding="utf-8") as handle:
        json.dump(CLASSES, handle, indent=2)
    candidate_sha = sha256_file(cand_abs)

    remaining_clear = cand_n == 0
    promote = (
        bool(summary.get("leakage_free"))
        and train_ok
        and all(train_counts[c] > 0 for c in CLASSES)
        and remaining_clear
        and cand_n < baseline_n
        and candidate_sha != production_sha_before
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not remaining_clear:
        reasons.append(f"STILL_COLLAPSES n={cand_n}")
    if not (cand_n < baseline_n):
        reasons.append(f"OOD_NOT_IMPROVED {baseline_n}->{cand_n}")
    if not train_ok:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if not summary.get("leakage_free"):
        reasons.append("LEAKAGE")

    promoted = False
    if promote:
        backup = production_abs + ".pre_subject_normal_backup"
        if os.path.exists(production_abs) and not os.path.exists(backup):
            shutil.copy2(production_abs, backup)
        shutil.copy2(cand_abs, production_abs)
        shutil.copy2(classes_path, os.path.join(ROOT, "ml", "models", "vision", "efficientnetv2_injury_best_classes.json"))
        meta = {
            "model_name": "EfficientNetV2 Classification",
            "version": "subject-normal-v1",
            "status": "READY_FOR_RESEARCH_DEMO",
            "training_status": "READY_FOR_RESEARCH_DEMO",
            "classes": CLASSES,
            "artifact_sha256": candidate_sha,
            "dataset_provenance": "synthetic cut/bruise drawings + REAL normal patches (AZH empty / empty-region crops); subject-aware normal split",
            "known_limitations": "Not clinical. cut/bruise still drawings. normal is peri-wound/clinic skin, not Mendeley healthy feet.",
            "metrics": {"test": test_m, "ood_collapse": cand_n, "baseline_ood_collapse": baseline_n},
        }
        with open(os.path.join(ROOT, "ml", "models", "vision", "efficientnetv2_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        register_model_artifact(
            model_name="EfficientNetV2 Classification",
            version="subject-normal-v1",
            artifact_path=EFFNET_CANONICAL,
            training_dataset="efficientnet_subject_normal",
            sample_count=summary["n"],
            classes=CLASSES,
            metrics=meta["metrics"],
            training_command="python -m ml.training.train_efficientnet_subject_normal",
            notes=meta["known_limitations"],
        )
        promoted = True

    status = "READY_FOR_RESEARCH_DEMO" if remaining_clear else "NOT_TRUSTWORTHY"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "reasons": reasons,
        "status": status,
        "baseline_ood_collapse_count": baseline_n,
        "candidate_ood_collapse_count": cand_n,
        "baseline_ood_probes": baseline,
        "candidate_ood_probes": probes,
        "val": val_best,
        "test": test_m,
        "history": history,
        "best_epoch": best_epoch,
        "production_sha256_before": production_sha_before,
        "candidate_sha256": candidate_sha,
        "production_sha256_after": sha256_file(production_abs) if os.path.exists(production_abs) else None,
        "dataset": summary,
        "classes": CLASSES,
        "device": device,
    }
    with open(os.path.join(ROOT, EFFNET_SUBJECT_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({k: report[k] for k in (
            "recommendation", "promoted_to_production", "reasons", "status",
            "baseline_ood_collapse_count", "candidate_ood_collapse_count",
            "production_sha256_before", "candidate_sha256", "production_sha256_after",
        )}, handle, indent=2)
    with open(os.path.join(ROOT, EFFNET_SUBJECT_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"recommendation": recommendation, "promoted": promoted, "status": status, "collapse": cand_n, "test_acc": test_m["accuracy"]}, indent=2))
    return report


if __name__ == "__main__":
    train()
