"""Train U-Net on deduped subject-aware public wound masks.

Promotion gate: CORE_WATCH blank/OOD positive_ratio must all be <= COLLAPSE_AREA.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    ROOT,
    UNET_CANONICAL,
    UNET_DEDUPED_CANDIDATE,
    UNET_DEDUPED_CANDIDATE_DIR,
    UNET_DEDUPED_MANIFEST,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact
from ml.training.prepare_unet_deduped_subject import build as build_dataset
from ml.training.train_unet import (
    COLLAPSE_AREA,
    CORE_WATCH,
    DiceBCELoss,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _baseline_probes,
    _collapse_names,
    _evaluate,
    _probe,
)

SEED = 42
MAX_EPOCHS = 8
PATIENCE = 3
BATCH_SIZE = 4
LR = 3e-4


class DedupedMaskDataset(Dataset):
    def __init__(self, rows, size=256, augment=False):
        self.rows = rows
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(os.path.join(ROOT, row["image_path"].replace("/", os.sep)))
        mask = cv2.imread(os.path.join(ROOT, row["mask_path"].replace("/", os.sep)), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment and self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        tensor = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        binary = (mask > 0).astype(np.float32)
        return torch.from_numpy(tensor).permute(2, 0, 1), torch.from_numpy(binary).unsqueeze(0)


def _load_split(split: str):
    rows = []
    with open(os.path.join(ROOT, UNET_DEDUPED_MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            if os.path.exists(os.path.join(ROOT, row["image_path"].replace("/", os.sep))):
                row["empty_mask"] = str(row.get("empty_mask", "")).lower() in {"1", "true", "yes"}
                rows.append(row)
    return rows


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    summary = build_dataset()
    baseline_probes = _baseline_probes()
    baseline_fail = _collapse_names(baseline_probes)
    baseline_n = len(baseline_fail)
    production_abs = os.path.join(ROOT, UNET_CANONICAL)
    production_sha_before = sha256_file(production_abs) if os.path.exists(production_abs) else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    n_train_empty = sum(1 for r in train_rows if r.get("empty_mask"))
    print(f"U-Net deduped train on {device} baseline_collapse={baseline_n} empty_train={n_train_empty}")

    train_loader = DataLoader(DedupedMaskDataset(train_rows, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DedupedMaskDataset(val_rows), batch_size=BATCH_SIZE)
    test_loader = DataLoader(DedupedMaskDataset(test_rows), batch_size=BATCH_SIZE)

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    for name, param in model.named_parameters():
        param.requires_grad = (not name.startswith("encoder")) or ("encoder.layer4" in name)
    model.to(device)
    criterion = DiceBCELoss(pos_weight=None)
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
        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(n, 1), 6),
            "val_loss": val_m["loss"],
            "val_dice": val_m["mean_dice"],
            "val_fp_area": val_m["mean_false_positive_area"],
        })
        print(
            f"epoch {epoch}/{MAX_EPOCHS} train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={val_m['loss']:.4f} val_dice={val_m['mean_dice']:.4f} val_fp={val_m['mean_false_positive_area']:.4f}"
        )
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
    fail = _collapse_names(probes)
    cand_n = len(fail)
    train_ok = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]

    os.makedirs(os.path.join(ROOT, UNET_DEDUPED_CANDIDATE_DIR), exist_ok=True)
    cand_abs = os.path.join(ROOT, UNET_DEDUPED_CANDIDATE)
    torch.save(model.state_dict(), cand_abs)
    candidate_sha = sha256_file(cand_abs)

    remaining_clear = cand_n == 0
    status = "READY_FOR_RESEARCH_DEMO" if remaining_clear else "MODEL_OUTPUT_NOT_TRUSTWORTHY"
    ood_improved = cand_n < baseline_n
    ood_holds_when_baseline_clear = baseline_n == 0 and cand_n == 0 and float(test_m.get("mean_dice") or 0) >= 0.55
    promote = (
        bool(summary.get("leakage_free"))
        and train_ok
        and test_m["n"] > 0
        and n_train_empty >= 1
        and remaining_clear
        and (ood_improved or ood_holds_when_baseline_clear)
        and candidate_sha != production_sha_before
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not remaining_clear:
        reasons.append(f"STILL_PAINTS {fail}")
    if not (ood_improved or ood_holds_when_baseline_clear):
        reasons.append(f"OOD_NOT_IMPROVED {baseline_n}->{cand_n}")
    if n_train_empty < 1:
        reasons.append("NO_EMPTY_TRAIN")
    if not train_ok:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")

    promoted = False
    if promote:
        backup = production_abs + ".pre_deduped_subject_backup"
        if os.path.exists(production_abs) and not os.path.exists(backup):
            shutil.copy2(production_abs, backup)
        shutil.copy2(cand_abs, production_abs)
        meta = {
            "model_name": "ResNet34-UNet Segmentation",
            "version": "deduped-subject-v1",
            "status": status,
            "training_status": status,
            "artifact_sha256": candidate_sha,
            "previous_sha256": production_sha_before,
            "dataset_provenance": "AZH+wseg+Medetec subject-aware, exact+near dedupe, synthetic empty canvases",
            "metrics": {"val": val_best, "test": test_m, "ood_collapse": cand_n, "collapse_area": COLLAPSE_AREA},
            "known_limitations": "Research-demo binary wound masks on ulcer/chronic photos. Not clinical. Not cut/bruise localization.",
            "training_was_real": True,
        }
        with open(os.path.join(ROOT, "ml", "models", "vision", "unet_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        register_model_artifact(
            model_name="ResNet34-UNet Segmentation",
            version="deduped-subject-v1",
            artifact_path=UNET_CANONICAL,
            training_dataset="unet_deduped_subject",
            sample_count=summary["n"],
            classes=["wound_binary"],
            metrics=meta["metrics"],
            training_command="python -m ml.training.train_unet_deduped",
            notes=meta["known_limitations"],
        )
        promoted = True

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "reasons": reasons,
        "status": status,
        "baseline_ood_collapse_count": baseline_n,
        "candidate_ood_collapse_count": cand_n,
        "baseline_fail": baseline_fail,
        "candidate_fail": fail,
        "baseline_ood_probes": baseline_probes,
        "candidate_ood_probes": probes,
        "val": val_best,
        "test": test_m,
        "history": history,
        "best_epoch": best_epoch,
        "production_sha256_before": production_sha_before,
        "candidate_sha256": candidate_sha,
        "production_sha256_after": sha256_file(production_abs) if os.path.exists(production_abs) else None,
        "dataset": summary,
        "device": device,
        "collapse_area": COLLAPSE_AREA,
        "core_watch": list(CORE_WATCH),
    }
    with open(os.path.join(ROOT, UNET_DEDUPED_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({k: report[k] for k in (
            "recommendation", "promoted_to_production", "reasons", "status",
            "baseline_ood_collapse_count", "candidate_ood_collapse_count", "candidate_fail",
            "production_sha256_before", "candidate_sha256", "production_sha256_after",
        )}, handle, indent=2)
    with open(os.path.join(ROOT, UNET_DEDUPED_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "recommendation": recommendation,
        "promoted": promoted,
        "status": status,
        "collapse": cand_n,
        "test_dice": test_m.get("mean_dice"),
    }, indent=2))
    return report


if __name__ == "__main__":
    train()
