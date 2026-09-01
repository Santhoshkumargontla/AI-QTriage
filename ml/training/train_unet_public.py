"""Train ResNet34-UNet on public wound photos + documented empty synthetics.

Production is overwritten only if blank/OOD painting is fully cleared
(positive ratio <= COLLAPSE_AREA on CORE_WATCH).
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
    UNET_PUBLIC_CANDIDATE,
    UNET_PUBLIC_CANDIDATE_DIR,
    UNET_PUBLIC_MANIFEST,
    UNET_PUBLIC_ROOT,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact
from ml.training.prepare_unet_public_real import build as build_dataset
from ml.training.train_unet import (
    COLLAPSE_AREA,
    CORE_WATCH,
    DiceBCELoss,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OOD_WATCH,
    THRESHOLD,
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


class PublicWoundMaskDataset(Dataset):
    def __init__(self, rows, size=256, augment=False):
        self.rows = rows
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        mask_path = os.path.join(ROOT, row["mask_path"].replace("/", os.sep))
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"{img_path} {mask_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment and self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        tensor = img.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)
        binary = (mask > 0).astype(np.float32)
        return tensor, torch.from_numpy(binary).unsqueeze(0)


def _load_split(split: str) -> list[dict]:
    rows = []
    with open(os.path.join(ROOT, UNET_PUBLIC_MANIFEST), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            img = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
            msk = os.path.join(ROOT, row["mask_path"].replace("/", os.sep))
            if os.path.exists(img) and os.path.exists(msk):
                row["empty_mask"] = str(row.get("empty_mask", "")).lower() in {"1", "true", "yes"}
                rows.append(row)
    return rows


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    summary = build_dataset()
    baseline_probes = _baseline_probes()
    baseline_fail = _collapse_names(baseline_probes)
    baseline_collapse_n = len(baseline_fail)
    production_abs = os.path.join(ROOT, UNET_CANONICAL)
    production_sha_before = sha256_file(production_abs) if os.path.exists(production_abs) else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"U-Net public training on {device} baseline_collapse={baseline_collapse_n} {baseline_fail}")

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    n_train_empty = sum(1 for r in train_rows if r.get("empty_mask"))
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} empty_train={n_train_empty}")

    train_loader = DataLoader(PublicWoundMaskDataset(train_rows, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PublicWoundMaskDataset(val_rows), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PublicWoundMaskDataset(test_rows), batch_size=BATCH_SIZE)

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    for name, param in model.named_parameters():
        param.requires_grad = (not name.startswith("encoder")) or ("encoder.layer4" in name)
    model.to(device)
    criterion = DiceBCELoss(pos_weight=None)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

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
        val_m = _evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_m["loss"],
            "val_dice": val_m["mean_dice"],
            "val_fp_area": val_m["mean_false_positive_area"],
        })
        print(
            f"epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} val_loss={val_m['loss']:.4f} "
            f"val_dice={val_m['mean_dice']:.4f} val_fp={val_m['mean_false_positive_area']:.4f}"
        )
        if val_m["loss"] + 1e-6 < best_val_loss:
            best_val_loss = val_m["loss"]
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
    test_m = _evaluate(model, test_loader, criterion, device)
    probes = _probe(model, device)
    candidate_fail = _collapse_names(probes)
    candidate_collapse_n = len(candidate_fail)
    train_loss_decreased = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]

    os.makedirs(os.path.join(ROOT, UNET_PUBLIC_CANDIDATE_DIR), exist_ok=True)
    cand_abs = os.path.join(ROOT, UNET_PUBLIC_CANDIDATE)
    torch.save(model.state_dict(), cand_abs)
    candidate_sha = sha256_file(cand_abs)
    ood_improved = candidate_collapse_n < baseline_collapse_n
    remaining_clear = candidate_collapse_n == 0
    status = "READY_FOR_RESEARCH_DEMO" if remaining_clear else "MODEL_OUTPUT_NOT_TRUSTWORTHY"
    promote = (
        bool(summary.get("leakage_free"))
        and train_loss_decreased
        and test_m["n"] > 0
        and n_train_empty >= 1
        and ood_improved
        and remaining_clear
        and candidate_sha != production_sha_before
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not remaining_clear:
        reasons.append(f"CANDIDATE_STILL_PAINTS_BLANKS {candidate_fail}")
    if not ood_improved:
        reasons.append(f"OOD_NOT_IMPROVED baseline={baseline_collapse_n} candidate={candidate_collapse_n}")
    if n_train_empty < 1:
        reasons.append("NO_EMPTY_MASK_TRAIN_EXAMPLES")
    if not train_loss_decreased:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if promote:
        reasons.append(f"OOD painting cleared {baseline_collapse_n}->{candidate_collapse_n}")

    promoted = False
    if promote:
        backup = production_abs + ".pre_public_real_backup"
        if os.path.exists(production_abs) and not os.path.exists(backup):
            shutil.copy2(production_abs, backup)
        shutil.copy2(cand_abs, production_abs)
        promoted = True
        meta_path = os.path.join(ROOT, "ml", "models", "vision", "unet_metadata.json")
        meta = {
            "model_name": "ResNet34-UNet Segmentation",
            "version": "public-wseg-medetec-v1",
            "status": status,
            "device": device,
            "metrics": {
                "dataset_name": "unet_public_real",
                "dataset_type": "public_wound_photos_plus_synthetic_empty",
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "test_samples": len(test_rows),
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "loss": "unweighted_BCE_plus_Dice",
                "encoder": "resnet34 imagenet; decoder+layer4 trained",
                "device": device,
                "val": val_best,
                "test": test_m,
                "uniform_image_probes": {k: probes[k] for k in ("black", "white", "gray") if k in probes},
                "blank_mask_untrustworthy": False,
                "training_history": history,
                "metrics_source": "computed_from_held_out_predictions",
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            "classes": ["wound_mask"],
            "weights_path": "ml/models/vision/unet_injury_best.pt",
            "canonical_path": "ml/models/vision/unet_injury_best.pt",
            "artifact_sha256": candidate_sha,
            "data_provenance_class": "PUBLIC_MIXED_SYNTHETIC_EMPTY",
            "dataset_provenance": "wseg CC-BY-NC-4.0 + Medetec 224 + SYNTHETIC_EMPTY_TARGET. Not sports-injury cut/bruise.",
            "training_status": status,
            "known_limitations": (
                "Research-demo binary wound segmentation on chronic/foot-ulcer photos. "
                "Not clinical. Not cut/bruise/swelling localization. Empty canvases are synthetic."
            ),
            "training_was_real": True,
            "previous_sha256": production_sha_before,
        }
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "reasons": reasons,
        "status": status,
        "device": device,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "val": val_best,
        "test": test_m,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "baseline_fail": baseline_fail,
        "candidate_fail": candidate_fail,
        "baseline_ood_probes": baseline_probes,
        "candidate_ood_probes": probes,
        "production_sha256_before": production_sha_before,
        "candidate_sha256": candidate_sha,
        "production_sha256_after": sha256_file(production_abs) if os.path.exists(production_abs) else None,
        "dataset": summary,
        "unfreeze": "decoder + encoder.layer4",
        "mask_binarize": "mask > 0 (Medetec is 0/1, wseg is 0/255)",
        "domain": "chronic/foot-ulcer photos + synthetic empty canvases; not sports-injury cut/bruise",
    }
    with open(os.path.join(ROOT, UNET_PUBLIC_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({k: report[k] for k in (
            "recommendation", "promoted_to_production", "reasons", "status",
            "baseline_ood_collapse_count", "candidate_ood_collapse_count",
            "candidate_fail", "production_sha256_before", "candidate_sha256",
            "production_sha256_after",
        )}, handle, indent=2)
    with open(os.path.join(ROOT, UNET_PUBLIC_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if promoted:
        register_model_artifact(
            model_name="ResNet34-UNet Segmentation",
            version="public-wseg-medetec-v1",
            artifact_path=UNET_CANONICAL,
            training_dataset="unet_public_real",
            sample_count=summary["n"],
            classes=["wound_binary"],
            metrics={"test": test_m, "ood_collapse": candidate_collapse_n, "status": status},
            training_command="python -m ml.training.train_unet_public",
            notes="Public wound photos + synthetic empty targets. Not clinical. Domain is ulcers, not acute cut/bruise.",
        )
    print(json.dumps({
        "recommendation": recommendation,
        "promoted": promoted,
        "status": status,
        "collapse": candidate_collapse_n,
        "test_dice": test_m.get("mean_dice"),
        "test_fp": test_m.get("mean_false_positive_area"),
    }, indent=2))
    return report


if __name__ == "__main__":
    train()
