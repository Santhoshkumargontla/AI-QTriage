"""Train EfficientNet on Kaggle multi-class wound photos. Promote only if gates pass."""
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

from ml.models.canonical_paths import EFFNET_CANONICAL, ROOT, sha256_file
from ml.models.model_registry_manager import register_model_artifact
from ml.training.prepare_efficientnet_kaggle_v1 import CLASSES, OUT_ROOT, build as build_dataset

SEED = 42
# CPU training is slow; keep honest evaluation but allow fewer epochs via env.
_MAX_DEFAULT = 16 if torch.cuda.is_available() else 8
MAX_EPOCHS = int(os.environ.get("EFFNET_MAX_EPOCHS", str(_MAX_DEFAULT)))
PATIENCE = int(os.environ.get("EFFNET_PATIENCE", "4" if not torch.cuda.is_available() else "5"))
BATCH_SIZE = int(os.environ.get("EFFNET_BATCH_SIZE", "8"))
LR = 2e-4
COLLAPSE_MAX = 0.90
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")
CAND_DIR = os.path.join("ml", "models", "efficientnet_kaggle_v1_training")
CAND_PATH = os.path.join(CAND_DIR, "efficientnetv2_kaggle_candidate.pt")
CLASSES_JSON = os.path.join(CAND_DIR, "efficientnet_classes.json")
INJURY = {c for c in CLASSES if c not in {"normal", "ood_reject"}}


class InjuryDataset(Dataset):
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
    with open(os.path.join(ROOT, MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["class"] in CLASSES:
                rows.append(row)
    return rows


def _tensor(img_rgb, device):
    img = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy((img - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)


def _ood_collapse(model, device) -> int:
    model.eval()
    rng = np.random.default_rng(0)
    specs = {
        "gray": np.full((224, 224, 3), 180, np.uint8),
        "black": np.zeros((224, 224, 3), np.uint8),
        "white": np.full((224, 224, 3), 255, np.uint8),
        "noise": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "blue": np.full((224, 224, 3), (20, 60, 200), np.uint8),
        "skin": np.full((224, 224, 3), (185, 145, 125), np.uint8),
    }
    collapse = 0
    details = {}
    with torch.no_grad():
        for name, img in specs.items():
            logits = model(_tensor(img, device))[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            idx = int(probs.argmax())
            winner = CLASSES[idx]
            details[name] = {"winner": winner, "max_prob": float(probs.max()), "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}}
            if winner in INJURY and float(probs.max()) >= COLLAPSE_MAX:
                collapse += 1
    return collapse, details


def _eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            pred = logits.argmax(1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ps.extend(pred.tolist())
    acc = float(accuracy_score(ys, ps)) if ys else 0.0
    p, r, f1, _ = precision_recall_fscore_support(ys, ps, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(len(CLASSES)))).tolist()
    per = {}
    p_i, r_i, f_i, s_i = precision_recall_fscore_support(ys, ps, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    for i, name in enumerate(CLASSES):
        per[name] = {"precision": float(p_i[i]), "recall": float(r_i[i]), "f1": float(f_i[i]), "support": int(s_i[i])}
    return {"n": len(ys), "accuracy": acc, "macro_f1": float(f1), "macro_precision": float(p), "macro_recall": float(r), "per_class": per, "confusion_matrix": cm}


def train():
    os.makedirs(CAND_DIR, exist_ok=True)
    # Reuse prepared manifest when present (avoids multi-minute full rebuild on every train).
    prep_path = os.path.join(OUT_ROOT, "PREPARE_REPORT.json")
    if os.path.isfile(MANIFEST) and os.path.isfile(prep_path):
        with open(prep_path, encoding="utf-8") as handle:
            prep = json.load(handle)
        print(f"reusing prepared dataset: n_rows={prep.get('n_rows')} classes={prep.get('classes')}", flush=True)
    else:
        prep = build_dataset()
    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    assert train_rows and val_rows and test_rows, "empty splits"
    print(f"splits train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    import timm

    # Avoid network hang on pretrained download (SSL issues observed). Warm-start from
    # existing canonical checkpoint when present; otherwise random init (still real train).
    model = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=False,
        num_classes=len(CLASSES),
    )
    warm = os.path.join(ROOT, EFFNET_CANONICAL)
    if os.path.isfile(warm):
        try:
            state = torch.load(warm, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model_sd = model.state_dict()
            filtered = {
                k: v
                for k, v in state.items()
                if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
            }
            missing = model.load_state_dict(filtered, strict=False)
            print(
                f"warm_start keys={len(filtered)}/{len(model_sd)} "
                f"unexpected={len(getattr(missing, 'unexpected_keys', []) or [])}",
                flush=True,
            )
        except Exception as exc:
            print(f"warm_start_failed: {exc}", flush=True)
    else:
        print("warm_start skipped (no canonical weights)", flush=True)
    model.to(device)

    counts = [sum(1 for r in train_rows if r["class"] == c) for c in CLASSES]
    weights = [1.0 / max(c, 1) for c in counts]
    sample_w = [weights[CLASSES.index(r["class"])] for r in train_rows]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    train_loader = DataLoader(InjuryDataset(train_rows, True), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(InjuryDataset(val_rows), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(InjuryDataset(test_rows), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = _eval(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_m})
        print(f"epoch {epoch}/{MAX_EPOCHS} loss={np.mean(losses):.4f} val_f1={val_m['macro_f1']:.4f}", flush=True)
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CAND_PATH)
    with open(CLASSES_JSON, "w", encoding="utf-8") as handle:
        json.dump(CLASSES, handle, indent=2)

    test_m = _eval(model, test_loader, device)
    collapse, ood_details = _ood_collapse(model, device)
    cand_sha = sha256_file(CAND_PATH)

    # Baseline comparison: current canonical on overlapping cut/bruise if possible — informational only
    promote = collapse == 0 and test_m["macro_f1"] >= 0.45 and test_m["n"] >= 40
    reason = []
    if collapse != 0:
        reason.append(f"ood_injury_collapse={collapse}")
    if test_m["macro_f1"] < 0.45:
        reason.append(f"macro_f1_too_low={test_m['macro_f1']:.3f}")
    if test_m["n"] < 40:
        reason.append(f"test_n_too_small={test_m['n']}")
    if promote:
        reason.append("gates_passed")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "classes": CLASSES,
        "prepare": prep,
        "candidate_path": CAND_PATH.replace("\\", "/"),
        "candidate_sha256": cand_sha,
        "history": history,
        "test": test_m,
        "ood_collapse": collapse,
        "ood_details": ood_details,
        "promote": promote,
        "promotion_reason": "; ".join(reason),
        "baseline_canonical": EFFNET_CANONICAL.replace("\\", "/"),
        "baseline_sha256": sha256_file(os.path.join(ROOT, EFFNET_CANONICAL)) if os.path.isfile(os.path.join(ROOT, EFFNET_CANONICAL)) else None,
    }
    report_path = os.path.join(CAND_DIR, "TRAIN_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if promote:
        backup = os.path.join(ROOT, EFFNET_CANONICAL) + ".pre_kaggle_v1_backup"
        if os.path.isfile(os.path.join(ROOT, EFFNET_CANONICAL)):
            shutil.copy2(os.path.join(ROOT, EFFNET_CANONICAL), backup)
        shutil.copy2(CAND_PATH, os.path.join(ROOT, EFFNET_CANONICAL))
        classes_dest = os.path.splitext(os.path.join(ROOT, EFFNET_CANONICAL))[0] + "_classes.json"
        shutil.copy2(CLASSES_JSON, classes_dest)
        meta = {
            "model_name": "EfficientNetV2 Classification",
            "version": "kaggle-v1",
            "status": "READY_FOR_RESEARCH_DEMO",
            "training_status": "READY_FOR_RESEARCH_DEMO",
            "classes": CLASSES,
            "artifact_sha256": cand_sha,
            "metrics": {"test": test_m, "ood_collapse": collapse},
            "dataset_provenance": "Kaggle wound classification photos + normal/ood; licenses often unknown on card",
            "known_limitations": "Not clinical. Swelling unsupported. Fracture not in this head (X-ray separate). Small real-photo n.",
            "evaluation_artifact": report_path.replace("\\", "/"),
            "last_evaluated": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(ROOT, "ml", "models", "vision", "efficientnetv2_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        try:
            register_model_artifact(
                model_name="EfficientNetV2 Classification",
                version="kaggle-v1",
                artifact_path=os.path.join(ROOT, EFFNET_CANONICAL),
                training_dataset="efficientnet_kaggle_v1",
                sample_count=int(test_m["n"]),
                classes=CLASSES,
                metrics={"test": test_m, "ood_collapse": collapse},
                training_command="backend\\venv\\Scripts\\python.exe ml\\training\\train_efficientnet_kaggle_v1.py",
                notes=meta["known_limitations"],
            )
        except Exception as exc:
            report["registry_error"] = str(exc)
        report["promoted"] = True
        report["backup"] = backup.replace("\\", "/")
    else:
        report["promoted"] = False

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("promote", "promoted", "promotion_reason", "ood_collapse", "test", "candidate_sha256")}, indent=2))
    return report


if __name__ == "__main__":
    train()
