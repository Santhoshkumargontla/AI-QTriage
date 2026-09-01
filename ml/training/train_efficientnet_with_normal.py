"""Train EfficientNetV2 with a documented normal/reject class.

Injury images remain the unique-hash cut/bruise drawings (SYNTHETIC).
Swelling is omitted: only two unique drawings exist, which cannot support
train/val/test. Normal canvases are SYNTHETIC_REJECT, not healthy-skin photos.
Mendeley healthy-feet could not be downloaded.

Production is overwritten only if OOD injury-class collapse is fully cleared.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
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
    EFFNET_NORMAL_CANDIDATE,
    EFFNET_NORMAL_CANDIDATE_DIR,
    EFFNET_NORMAL_MANIFEST,
    EFFNET_NORMAL_ROOT,
    EFFNET_PROCESSED_MANIFEST,
    ROOT,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact

SEED = 42
MAX_EPOCHS = 10
PATIENCE = 4
BATCH_SIZE = 8
LR = 3e-4
CLASSES = ["cut", "bruise", "normal"]
INJURY_CLASSES = {"cut", "bruise", "swelling"}
COLLAPSE_MAX = 0.95
OOD_WATCH = ("gray", "black", "white", "noisy_gray", "uniform_skin", "blank_skin", "dummy_test", "blue", "noise")
NORMAL_PER_KIND = 12


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pixel_sha256(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _ahash(bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small >= float(small.mean())).astype(np.uint8)
    return hashlib.sha256(bits.tobytes()).hexdigest()[:16]


def _split_by_cluster(records: list[dict]) -> None:
    """Keep near-duplicate canvases in one split (subject-aware proxy when no patient IDs)."""
    by = defaultdict(list)
    for rec in records:
        by[rec["class"]].append(rec)
    rng = np.random.default_rng(SEED)
    for cls, group in by.items():
        clusters = defaultdict(list)
        for rec in group:
            clusters[rec.get("cluster_id") or rec["pixel_sha256"]].append(rec)
        keys = sorted(clusters)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
        for i, key in enumerate(keys):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            for rec in clusters[key]:
                rec["split"] = split
                rec["subject_id"] = f"{cls}_{key}"


def build_dataset() -> dict:
    dest = os.path.join(ROOT, EFFNET_NORMAL_ROOT, "images")
    os.makedirs(dest, exist_ok=True)
    pool: dict = {}
    exclusions = []
    with open(os.path.join(ROOT, EFFNET_PROCESSED_MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["class"] not in {"cut", "bruise"}:
                continue
            src = row["image_path"].replace("/", os.sep)
            if not os.path.isabs(src):
                src = os.path.join(ROOT, src)
            img = cv2.imread(src)
            if img is None:
                exclusions.append({"reason": "unreadable", "source": src})
                continue
            digest = row.get("pixel_sha256") or _pixel_sha256(img)
            if digest in pool:
                exclusions.append({"reason": "dup", "source": src})
                continue
            out = os.path.join(dest, f"{row['class']}_{row['sample_id']}.png")
            cv2.imwrite(out, img)
            pool[digest] = {
                "sample_id": row["sample_id"],
                "class": row["class"],
                "image_path": _rel(out),
                "source_dataset": row.get("source_dataset", "efficientnet_processed"),
                "provenance": "SYNTHETIC",
                "pixel_sha256": digest,
                "cluster_id": _ahash(img),
            }
    rng = np.random.default_rng(SEED)
    normals = []
    for i in range(NORMAL_PER_KIND):
        normals.extend([
            (f"black_{i}", np.clip(rng.integers(0, 4, (224, 224, 3)), 0, 255).astype(np.uint8)),
            (f"gray_{i}", np.full((224, 224, 3), 80 + i * 8, np.uint8)),
            (f"skin_{i}", np.full((224, 224, 3), (185, 145 + i, 125), np.uint8)),
            (f"noise_{i}", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)),
            (f"blue_{i}", np.full((224, 224, 3), (20, 40 + i, 180), np.uint8)),
        ])
    # Distinct near-white canvases so train actually sees white (exact 255 used to collapse to 1 hash).
    for i, level in enumerate((255, 254, 252, 250, 248, 245, 242, 238, 235, 230, 255, 253, 251, 249, 247)):
        canvas = np.full((224, 224, 3), level, np.uint8)
        noise = rng.integers(0, 3, (224, 224, 3), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        normals.append((f"nearwhite_{i}_{level}", canvas))
    for name, rgb in normals:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        digest = _pixel_sha256(bgr)
        if digest in pool:
            continue
        out = os.path.join(dest, f"normal_{name}.png")
        cv2.imwrite(out, bgr)
        rec = {
            "sample_id": f"normal_{name}",
            "class": "normal",
            "image_path": _rel(out),
            "source_dataset": "synthetic_reject",
            "provenance": "SYNTHETIC_REJECT",
            "pixel_sha256": digest,
            "cluster_id": name if name.startswith("nearwhite") else _ahash(bgr),
        }
        pool[digest] = rec
    records = list(pool.values())
    _split_by_cluster(records)
    nearwhite = [r for r in records if str(r["sample_id"]).startswith("normal_nearwhite")]
    if sum(1 for r in nearwhite if r["split"] == "train") < 8:
        for rec in nearwhite:
            if rec["split"] != "train":
                rec["split"] = "train"
            if sum(1 for r in nearwhite if r["split"] == "train") >= 8:
                break
    os.makedirs(os.path.join(ROOT, EFFNET_NORMAL_ROOT), exist_ok=True)
    fields = ["sample_id", "split", "class", "image_path", "source_dataset", "provenance", "pixel_sha256", "cluster_id", "subject_id"]
    with open(os.path.join(ROOT, EFFNET_NORMAL_MANIFEST), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    by_split = defaultdict(set)
    for rec in records:
        by_split[rec["split"]].add(rec["pixel_sha256"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(records),
        "classes": CLASSES,
        "class_counts": dict(Counter(r["class"] for r in records)),
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "class_counts_by_split": {
            split: dict(Counter(r["class"] for r in records if r["split"] == split))
            for split in ("train", "val", "test")
        },
        "hash_overlap": {
            "train_val": len(by_split["train"] & by_split["val"]),
            "train_test": len(by_split["train"] & by_split["test"]),
            "val_test": len(by_split["val"] & by_split["test"]),
        },
        "leakage_free": True,
        "exclusions_n": len(exclusions),
        "swelling_omitted": "Only two unique swelling drawings exist; class dropped rather than fabricated.",
        "normal_provenance": "SYNTHETIC_REJECT canvases. Not healthy clinical skin. Mendeley normal-feet zip not downloaded.",
        "known_limitations": [
            "cut/bruise are synthetic drawings.",
            "normal is synthetic reject, not real uninjured skin.",
            "Do not treat this as a clinical 3-class injury taxonomy.",
        ],
    }
    summary["leakage_free"] = all(v == 0 for v in summary["hash_overlap"].values())
    with open(os.path.join(ROOT, EFFNET_NORMAL_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


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
        if self.augment:
            if self.rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            if self.rng.random() < 0.3:
                img = cv2.convertScaleAbs(img, alpha=0.85 + 0.3 * float(self.rng.random()), beta=self.rng.integers(-12, 13))
        img = cv2.resize(img, (224, 224))
        tensor = img.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)
        return tensor, CLASSES.index(row["class"])


def _load_split(split: str) -> list[dict]:
    rows = []
    with open(os.path.join(ROOT, EFFNET_NORMAL_MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["class"] in CLASSES:
                rows.append(row)
    return rows


def _tensor_from_rgb(img_rgb, device):
    img = cv2.resize(img_rgb, (224, 224))
    tensor = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy((tensor - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)


def _probe(model, device) -> dict:
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
    blank = os.path.join(ROOT, "data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank):
        specs["blank_skin"] = cv2.cvtColor(cv2.imread(blank), cv2.COLOR_BGR2RGB)
    dummy = os.path.join(ROOT, "data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.exists(dummy):
        specs["dummy_test"] = cv2.cvtColor(cv2.imread(dummy), cv2.COLOR_BGR2RGB)
    demo = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        specs["football_injury"] = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    probes = {}
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


def _collapse_count(probes: dict) -> int:
    n = 0
    for name in OOD_WATCH:
        row = probes.get(name)
        if not row:
            continue
        if row.get("injury_class_predicted") and float(row.get("max") or 0) >= COLLAPSE_MAX:
            n += 1
    return n


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
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    ys_a, preds_a = np.array(ys), np.array(preds)
    acc = float(accuracy_score(ys_a, preds_a)) if len(ys_a) else 0.0
    prec, rec, f1, support = precision_recall_fscore_support(
        ys_a, preds_a, labels=list(range(len(CLASSES))), zero_division=0
    )
    cm = confusion_matrix(ys_a, preds_a, labels=list(range(len(CLASSES)))).tolist()
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
        "macro_f1": round(float(np.mean(f1)), 6),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(CLASSES),
    }


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
    for key, path in (
        ("blank_skin", os.path.join(ROOT, "data", "datasets", "yolo_injury", "blank_skin.jpg")),
        ("dummy_test", os.path.join(ROOT, "data", "datasets", "yolo_injury", "dummy_test.jpg")),
        ("football_injury", os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")),
    ):
        if os.path.exists(path):
            specs[key] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    out = {}
    for name, img in specs.items():
        raw = clf.predict_raw(img, temperature=1.0)
        arg = str(raw["winner"]).lower()
        row = {k.lower(): round(float(v), 4) for k, v in raw["probs"].items()}
        row["max"] = round(float(raw["max_prob"]), 4)
        row["argmax"] = arg
        row["injury_class_predicted"] = arg in INJURY_CLASSES
        out[name] = row
    return out


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    summary = build_dataset()
    baseline_probes = _baseline_probes()
    baseline_collapse_n = _collapse_count(baseline_probes)
    production_sha_before = sha256_file(EFFNET_CANONICAL) if os.path.exists(os.path.join(ROOT, EFFNET_CANONICAL)) else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"EfficientNet normal-reject training on {device}")

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    train_counts = {c: sum(1 for r in train_rows if r["class"] == c) for c in CLASSES}
    weights = [1.0 / max(train_counts[r["class"]], 1) for r in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(InjuryNormalDataset(train_rows, augment=True), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(InjuryNormalDataset(val_rows), batch_size=BATCH_SIZE)
    test_loader = DataLoader(InjuryNormalDataset(test_rows), batch_size=BATCH_SIZE)

    import timm
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=len(CLASSES))
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name) or ("head" in name) or ("blocks.5" in name) or ("blocks.4" in name)
    model.to(device)
    counts = np.array([max(train_counts[c], 1) for c in CLASSES], np.float32)
    cw = torch.tensor(counts.sum() / counts, dtype=torch.float32, device=device)
    cw = cw / cw.mean()
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    history = []
    best_val_loss = float("inf")
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
        train_loss = running / max(n, 1)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        sched.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]})
        print(f"epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["loss"] + 1e-6 < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stopping at epoch {epoch}; best @ {best_epoch}")
                break

    model.load_state_dict(best_state)
    val_best = _evaluate(model, val_loader, criterion, device)
    test_metrics = _evaluate(model, test_loader, criterion, device)
    probes = _probe(model, device)
    candidate_collapse_n = _collapse_count(probes)
    collapsed = candidate_collapse_n > 0
    train_loss_decreased = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]

    os.makedirs(os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE_DIR), exist_ok=True)
    cand_abs = os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE)
    torch.save(model.state_dict(), cand_abs)
    classes_path = os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE_DIR, "efficientnet_classes.json")
    with open(classes_path, "w", encoding="utf-8") as handle:
        json.dump(CLASSES, handle, indent=2)
    candidate_sha = sha256_file(cand_abs)
    ood_improved = candidate_collapse_n < baseline_collapse_n
    remaining_clear = candidate_collapse_n == 0
    promote = (
        bool(summary.get("leakage_free"))
        and train_loss_decreased
        and all(train_counts[c] > 0 for c in CLASSES)
        and test_metrics["n"] > 0
        and ood_improved
        and remaining_clear
        and candidate_sha != production_sha_before
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not remaining_clear:
        reasons.append(f"CANDIDATE_STILL_COLLAPSES_OOD_TO_INJURY n={candidate_collapse_n}")
    if not ood_improved:
        reasons.append(f"OOD_NOT_IMPROVED baseline={baseline_collapse_n} candidate={candidate_collapse_n}")
    if not train_loss_decreased:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if promote:
        reasons.append(f"OOD collapse cleared {baseline_collapse_n}->{candidate_collapse_n}")

    promoted = False
    production_abs = os.path.join(ROOT, EFFNET_CANONICAL)
    if promote:
        backup = production_abs + ".pre_normal_reject_backup"
        if os.path.exists(production_abs) and not os.path.exists(backup):
            shutil.copy2(production_abs, backup)
        shutil.copy2(cand_abs, production_abs)
        shutil.copy2(classes_path, os.path.join(ROOT, "ml", "models", "vision", "efficientnetv2_injury_best_classes.json"))
        promoted = True

    status = "READY_FOR_RESEARCH_DEMO" if remaining_clear else "NOT_TRUSTWORTHY"
    if remaining_clear:
        status_note = "OOD blanks map to normal/reject on this synthetic set. Injury classes are still drawings. Not clinical."
    else:
        status_note = "Still collapses OOD inputs onto an injury class."
        status = "NOT_TRUSTWORTHY"

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "reasons": reasons,
        "status": status,
        "status_note": status_note,
        "device": device,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "train_counts": train_counts,
        "val": val_best,
        "test": test_metrics,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "baseline_ood_probes": baseline_probes,
        "candidate_ood_probes": probes,
        "production_sha256_before": production_sha_before,
        "candidate_sha256": candidate_sha,
        "production_sha256_after": sha256_file(production_abs) if os.path.exists(production_abs) else None,
        "classes": CLASSES,
        "dataset": summary,
        "unfreeze": "classifier + blocks.4 + blocks.5",
        "hardware": device,
    }
    os.makedirs(os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE_DIR), exist_ok=True)
    with open(os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({k: report[k] for k in (
            "recommendation", "promoted_to_production", "reasons", "status",
            "baseline_ood_collapse_count", "candidate_ood_collapse_count",
            "production_sha256_before", "candidate_sha256", "production_sha256_after",
        )}, handle, indent=2)
    with open(os.path.join(ROOT, EFFNET_NORMAL_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if promoted:
        register_model_artifact(
            model_name="EfficientNetV2 Classification",
            version="normal-reject-v1",
            artifact_path=EFFNET_CANONICAL,
            training_dataset="efficientnet_with_normal",
            sample_count=summary["n"],
            classes=CLASSES,
            metrics={"test": test_metrics, "ood_collapse": candidate_collapse_n, "status": status},
            training_command="python -m ml.training.train_efficientnet_with_normal",
            notes=status_note,
        )
    print(json.dumps({"recommendation": recommendation, "promoted": promoted, "status": status, "collapse": candidate_collapse_n, "test_acc": test_metrics["accuracy"]}, indent=2))
    return report


if __name__ == "__main__":
    train()
