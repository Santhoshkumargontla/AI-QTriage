"""Real ResNet34-UNet training on unique-hash image/mask pairs.

Does not fabricate empty medical masks. Production weights are overwritten
only if raw blank/OOD painting improves and remaining core probes stay
below COLLAPSE_AREA. In-domain Dice alone is not a promotion reason.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    UNET_CANDIDATE,
    UNET_CANDIDATE_DIR,
    UNET_CANONICAL,
    UNET_PROCESSED_MANIFEST,
    UNET_PROCESSED_ROOT,
    sha256_file,
)
from ml.models.model_registry_manager import register_model_artifact
from ml.training.prepare_unet_processed_dataset import build as build_dataset

METADATA_SAVE_PATH = os.path.join("ml", "models", "vision", "unet_metadata.json")
HISTORY_CSV = os.path.join(UNET_CANDIDATE_DIR, "results.csv")
MAX_EPOCHS = 16
PATIENCE = 5
BATCH_SIZE = 4
LR = 1e-3
SEED = 42
THRESHOLD = 0.5
# Promotion fails if any watched blank/OOD probe exceeds this positive area.
COLLAPSE_AREA = 0.05
CORE_WATCH = ("black", "white", "gray", "mid_gray", "blank_skin", "dummy_test")
OOD_WATCH = CORE_WATCH + ("noisy_gray", "uniform_skin", "blue", "green", "noise")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WoundMaskDataset(Dataset):
    def __init__(self, rows, size=256):
        self.rows = rows
        self.size = size
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = row["image_path"].replace("/", os.sep)
        mask_path = row["mask_path"].replace("/", os.sep)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Missing pair {img_path} {mask_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Match production wrapper: INTER_LINEAR image, INTER_NEAREST mask, ImageNet z-score.
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        tensor = img.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)
        binary = (mask > 127).astype(np.float32)
        return tensor, torch.from_numpy(binary).unsqueeze(0)


class DiceBCELoss(nn.Module):
    """Unweighted BCE + per-image soft Dice.

    pos_weight is left at 1.0. Inverse-frequency BCE would amplify the
    already-present foreground (~14% of pixels) and, with no empty-mask
    targets, would worsen blank-image painting. Empty-target Dice is
     pred.sum() / (pred.sum() + 1)  so all-zero maps are penalized when
    they exist; they are not fabricated for this run.
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * targets).sum(dim=dims)
        denom = prob.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return bce + dice.mean()


def _load_split(split: str) -> list[dict]:
    rows = []
    with open(UNET_PROCESSED_MANIFEST, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            img = row["image_path"].replace("/", os.sep)
            msk = row["mask_path"].replace("/", os.sep)
            if os.path.exists(img) and os.path.exists(msk):
                row["empty_mask"] = str(row.get("empty_mask", "")).lower() in {"1", "true", "yes"}
                rows.append(row)
    return rows


def _pair_metrics(pred: np.ndarray, target: np.ndarray, eps=1e-6):
    p = pred.reshape(-1).astype(np.float32)
    t = target.reshape(-1).astype(np.float32)
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    pred_sum = float(p.sum())
    tgt_sum = float(t.sum())
    empty_t = tgt_sum == 0
    empty_p = pred_sum == 0
    if empty_t and empty_p:
        dice = iou = prec = rec = 1.0
    elif empty_t and not empty_p:
        dice = iou = prec = 0.0
        rec = 1.0
    else:
        dice = (2 * tp + eps) / (pred_sum + tgt_sum + eps)
        union = pred_sum + tgt_sum - tp
        iou = (tp + eps) / (union + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
    fp_area = fp / max(p.size, 1)
    pos_ratio = pred_sum / max(p.size, 1)
    return dice, iou, prec, rec, fp_area, pos_ratio


def _summarize_pairs(rows_metrics: list[tuple]) -> dict:
    if not rows_metrics:
        return {
            "n": 0, "mean_dice": 0.0, "mean_iou": 0.0, "precision": 0.0,
            "recall": 0.0, "mean_false_positive_area": 0.0, "mean_positive_ratio": 0.0,
        }
    cols = list(zip(*rows_metrics))
    return {
        "n": len(rows_metrics),
        "mean_dice": round(float(np.mean(cols[0])), 6),
        "mean_iou": round(float(np.mean(cols[1])), 6),
        "precision": round(float(np.mean(cols[2])), 6),
        "recall": round(float(np.mean(cols[3])), 6),
        "mean_false_positive_area": round(float(np.mean(cols[4])), 6),
        "mean_positive_ratio": round(float(np.mean(cols[5])), 6),
    }


def _evaluate(model, loader, criterion, device, threshold=THRESHOLD):
    model.eval()
    pairs = []
    running = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))
            pred = (torch.sigmoid(logits) > threshold).float()
            for i in range(x.size(0)):
                pairs.append(_pair_metrics(pred[i, 0].cpu().numpy(), y[i, 0].cpu().numpy()))
    out = _summarize_pairs(pairs)
    out["loss"] = round(running / max(n, 1), 6)
    out["threshold"] = threshold
    return out


def _tensor_from_rgb(img_rgb: np.ndarray, device):
    img = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy((tensor - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _probe_specs() -> dict:
    rng = np.random.default_rng(0)
    specs = {
        "gray": np.full((256, 256, 3), 180, dtype=np.uint8),
        "mid_gray": np.full((256, 256, 3), 128, dtype=np.uint8),
        "black": np.zeros((256, 256, 3), dtype=np.uint8),
        "white": np.full((256, 256, 3), 255, dtype=np.uint8),
        "noisy_gray": np.clip(np.full((256, 256, 3), 180) + rng.normal(0, 5, (256, 256, 3)), 0, 255).astype(np.uint8),
        "uniform_skin": np.full((256, 256, 3), (185, 145, 125), dtype=np.uint8),
        "blue": np.full((256, 256, 3), (20, 60, 200), dtype=np.uint8),
        "green": np.full((256, 256, 3), (20, 180, 40), dtype=np.uint8),
        "noise": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
    }
    cut = np.full((256, 256, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(cut, (80, 45), (170, 205), (190, 20, 20), 8)
    specs["blurred_cut"] = cv2.GaussianBlur(cut, (31, 31), 8)
    specs["synthetic_cut"] = cut
    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank):
        specs["blank_skin"] = cv2.cvtColor(cv2.imread(blank), cv2.COLOR_BGR2RGB)
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.exists(dummy):
        specs["dummy_test"] = cv2.cvtColor(cv2.imread(dummy), cv2.COLOR_BGR2RGB)
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        specs["football_injury"] = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    return specs


def _pack_probe(prob: np.ndarray, threshold=THRESHOLD) -> dict:
    binary = (prob > threshold).astype(np.float32)
    pos = float(binary.mean())
    return {
        "mean_prob": round(float(prob.mean()), 4),
        "max_prob": round(float(prob.max()), 4),
        "min_prob": round(float(prob.min()), 4),
        "positive_ratio": round(pos, 4),
        "false_positive_area": round(pos, 4),
        "threshold": threshold,
    }


def _probe(model, device, threshold=THRESHOLD) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for name, img in _probe_specs().items():
            if img is None:
                continue
            prob = torch.sigmoid(model(_tensor_from_rgb(img, device))).squeeze().cpu().numpy()
            out[name] = _pack_probe(prob, threshold)
    return out


def _baseline_probes() -> dict:
    from ml.vision.unet_wrapper import UNetSegmenter
    seg = UNetSegmenter()
    if not seg.is_loaded:
        return {}
    out = {}
    for name, img in _probe_specs().items():
        if img is None:
            continue
        raw = seg.segment_raw(img)
        if raw.get("probs") is None:
            continue
        out[name] = _pack_probe(np.asarray(raw["probs"]), float(raw.get("threshold") or THRESHOLD))
    return out


def _collapse_names(probes: dict, watch=CORE_WATCH) -> list[str]:
    failures = []
    for name in watch:
        row = probes.get(name)
        if not row:
            continue
        if row["positive_ratio"] > COLLAPSE_AREA:
            failures.append(name)
    return failures


def _eval_rows_with_model(model, rows, device, threshold=THRESHOLD) -> dict:
    model.eval()
    pairs = []
    with torch.no_grad():
        for row in rows:
            img = cv2.imread(row["image_path"].replace("/", os.sep))
            mask = cv2.imread(row["mask_path"].replace("/", os.sep), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            target = (target > 127).astype(np.float32)
            prob = torch.sigmoid(model(_tensor_from_rgb(img, device))).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.float32)
            pairs.append(_pair_metrics(pred, target))
    return _summarize_pairs(pairs)


def _eval_rows_with_segmenter(seg, rows, threshold=THRESHOLD) -> dict:
    pairs = []
    for row in rows:
        img = cv2.imread(row["image_path"].replace("/", os.sep))
        mask = cv2.imread(row["mask_path"].replace("/", os.sep), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        target = (target > 127).astype(np.float32)
        raw = seg.segment_raw(img)
        if raw.get("binary") is None:
            continue
        pred = (np.asarray(raw["binary"]) > 0).astype(np.float32)
        pairs.append(_pair_metrics(pred, target))
    return _summarize_pairs(pairs)


def _eval_empty_files_with_model(model, paths, device, threshold=THRESHOLD) -> dict:
    model.eval()
    pairs = []
    per = {}
    with torch.no_grad():
        for stem, path in paths:
            if not os.path.isfile(path):
                continue
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            target = np.zeros((256, 256), dtype=np.float32)
            prob = torch.sigmoid(model(_tensor_from_rgb(img, device))).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.float32)
            m = _pair_metrics(pred, target)
            pairs.append(m)
            per[stem] = {
                "dice": round(m[0], 6), "iou": round(m[1], 6),
                "precision": round(m[2], 6), "recall": round(m[3], 6),
                "false_positive_area": round(m[4], 6), "positive_ratio": round(m[5], 6),
            }
    out = _summarize_pairs(pairs)
    out["per_image"] = per
    return out


def _eval_empty_files_with_segmenter(seg, paths) -> dict:
    pairs = []
    per = {}
    for stem, path in paths:
        if not os.path.isfile(path):
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        target = np.zeros((256, 256), dtype=np.float32)
        raw = seg.segment_raw(img)
        if raw.get("binary") is None:
            continue
        pred = (np.asarray(raw["binary"]) > 0).astype(np.float32)
        m = _pair_metrics(pred, target)
        pairs.append(m)
        per[stem] = {
            "dice": round(m[0], 6), "iou": round(m[1], 6),
            "precision": round(m[2], 6), "recall": round(m[3], 6),
            "false_positive_area": round(m[4], 6), "positive_ratio": round(m[5], 6),
        }
    out = _summarize_pairs(pairs)
    out["per_image"] = per
    return out


def _threshold_sweep(model, loader, criterion, device) -> dict:
    return {str(t): _evaluate(model, loader, criterion, device, threshold=t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)}


def train_unet(max_epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR, patience: int = PATIENCE):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    production_sha_before = sha256_file(UNET_CANONICAL) if os.path.exists(UNET_CANONICAL) else None
    baseline_probes = _baseline_probes()
    baseline_fail = _collapse_names(baseline_probes)
    baseline_collapse_n = len(baseline_fail)

    from ml.vision.unet_wrapper import UNetSegmenter
    baseline_seg = UNetSegmenter()

    build_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"U-Net REAL training on {device}  early_stopping patience={patience} max_epochs={max_epochs}")

    train_rows, val_rows, test_rows = _load_split("train"), _load_split("val"), _load_split("test")
    n_train_empty = sum(1 for r in train_rows if r["empty_mask"])
    n_test_empty = sum(1 for r in test_rows if r["empty_mask"])
    train_pos_area = float(np.mean([float(r.get("mask_area") or 0) for r in train_rows])) if train_rows else 0.0
    # Unweighted BCE. Inverse-frequency pos_weight would be (1-fg)/fg ≈ 6x here.
    pos_weight_used = None
    print(
        f"train n={len(train_rows)} empty={n_train_empty} mean_fg={train_pos_area:.4f} "
        f"pos_weight={pos_weight_used or 1.0} (unweighted on purpose)"
    )

    train_loader = DataLoader(WoundMaskDataset(train_rows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WoundMaskDataset(val_rows), batch_size=batch_size)
    test_loader = DataLoader(WoundMaskDataset(test_rows), batch_size=batch_size)

    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            param.requires_grad = False
    model.to(device)
    criterion = DiceBCELoss(pos_weight=pos_weight_used)
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
        val_m = _evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_m["loss"],
            "val_dice": val_m["mean_dice"],
            "val_iou": val_m["mean_iou"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_fp_area": val_m["mean_false_positive_area"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{max_epochs} train_loss={train_loss:.4f} val_loss={val_m['loss']:.4f} "
            f"val_dice={val_m['mean_dice']:.4f} val_fp={val_m['mean_false_positive_area']:.4f}"
        )
        if val_m["loss"] + 1e-6 < best_val_loss:
            best_val_loss = val_m["loss"]
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
    test_m = _evaluate(model, test_loader, criterion, device)
    probes = _probe(model, device)
    candidate_fail = _collapse_names(probes)
    candidate_collapse_n = len(candidate_fail)
    collapsed = candidate_collapse_n > 0
    train_loss_decreased = len(history) >= 2 and history[-1]["train_loss"] < history[0]["train_loss"]
    val_sweep = _threshold_sweep(model, val_loader, criterion, device)

    existing_neg_paths = [
        ("blank_skin", os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")),
        ("dummy_test", os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")),
    ]
    candidate_injury = _eval_rows_with_model(model, test_rows, device)
    candidate_noinj = _eval_empty_files_with_model(model, existing_neg_paths, device)
    baseline_injury = _eval_rows_with_segmenter(baseline_seg, test_rows) if baseline_seg.is_loaded else {}
    baseline_noinj = _eval_empty_files_with_segmenter(baseline_seg, existing_neg_paths) if baseline_seg.is_loaded else {}

    os.makedirs(UNET_CANDIDATE_DIR, exist_ok=True)
    torch.save(model.state_dict(), UNET_CANDIDATE)
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch", "train_loss", "val_loss", "val_dice", "val_iou",
                "val_precision", "val_recall", "val_fp_area",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    with open(os.path.join(UNET_PROCESSED_ROOT, "processing_summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]
    overlap_ok = bool(summary.get("leakage_free"))
    candidate_sha = sha256_file(UNET_CANDIDATE)
    copied_production = bool(production_sha_before) and candidate_sha == production_sha_before

    ood_improved = candidate_collapse_n < baseline_collapse_n
    remaining_clear = candidate_collapse_n == 0
    # Do not promote on in-domain Dice while blanks still paint injuries.
    promote = (
        overlap_ok
        and train_loss_decreased
        and test_m["n"] > 0
        and ood_improved
        and remaining_clear
        and not copied_production
    )
    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if not overlap_ok:
        reasons.append("SPLIT_HASH_LEAKAGE")
    if not ood_improved:
        reasons.append(
            f"OOD_BLANK_PAINTING_NOT_IMPROVED baseline={baseline_collapse_n} candidate={candidate_collapse_n}"
        )
    if collapsed:
        reasons.append(f"CANDIDATE_STILL_PAINTS_BLANKS {candidate_fail}")
    if not train_loss_decreased:
        reasons.append("TRAIN_LOSS_DID_NOT_DECREASE")
    if n_train_empty < 1:
        reasons.append("NO_EMPTY_MASK_TRAIN_EXAMPLES_NOT_FABRICATED")
    if copied_production:
        reasons.append("CANDIDATE_SHA_EQUALS_PRODUCTION")
    if promote:
        reasons.append(
            f"OOD_COLLAPSE_CLEARED {baseline_collapse_n}->{candidate_collapse_n}. "
            "Status remains MODEL_OUTPUT_NOT_TRUSTWORTHY (synthetic drawings; empty class not trained)."
        )

    promoted = False
    if promote:
        os.makedirs(os.path.dirname(UNET_CANONICAL), exist_ok=True)
        backup = UNET_CANONICAL + ".pre_processed_retrain_backup"
        if os.path.exists(UNET_CANONICAL) and not os.path.exists(backup):
            shutil.copy2(UNET_CANONICAL, backup)
        shutil.copy2(UNET_CANDIDATE, UNET_CANONICAL)
        promoted = True
    production_sha_after = sha256_file(UNET_CANONICAL) if os.path.exists(UNET_CANONICAL) else None

    # Unique-hash drawings + no empty-mask class: never clear this.
    status = "MODEL_OUTPUT_NOT_TRUSTWORTHY"
    final_status = "REQUIRES_MORE_DATA"
    known_limitations = list(summary.get("known_limitations") or [])
    known_limitations.extend([
        "Not clinically validated.",
        "Do not treat held-out Dice as generalization to real wounds.",
        "Blank/OOD probes must be read alongside Dice/IoU. Gates hide uniforms; they do not certify photographs.",
        "Only two existing no-injury files were available. They were eval-only. Empty medical scenes were not generated.",
        "Unweighted BCE + per-image Dice. Threshold kept at 0.5 for comparison with production; a val sweep is reported but not used to hide OOD painting.",
    ])
    if collapsed:
        known_limitations.append(
            f"Pathological masks remain on probes: {', '.join(candidate_fail)} (positive_ratio > {COLLAPSE_AREA})."
        )

    metric_keys = ("n", "loss", "mean_dice", "mean_iou", "precision", "recall", "mean_false_positive_area", "mean_positive_ratio")
    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "dataset_name": "unet_processed",
        "dataset_type": "synthetic_drawings_unique_hash_no_fabricated_empty_masks",
        "dataset_size": summary.get("n"),
        "positives": summary.get("positives"),
        "negatives": summary.get("negatives"),
        "empty_mask_class_trained": False,
        "did_not_fabricate_negatives": True,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_empty_masks": n_train_empty,
        "test_empty_masks": n_test_empty,
        "train_mean_foreground_area": round(train_pos_area, 6),
        "split_file": os.path.join(UNET_PROCESSED_ROOT, "split.csv").replace("\\", "/"),
        "max_epochs": max_epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "early_stopping_monitor": "val_loss",
        "optimizer": "Adam",
        "lr": lr,
        "batch_size": batch_size,
        "seed": SEED,
        "loss": "unweighted_BCE_plus_per_image_Dice",
        "bce_pos_weight": 1.0,
        "encoder": "resnet34 imagenet frozen, decoder trained",
        "preprocess": "resize_256 INTER_LINEAR image / INTER_NEAREST mask, ImageNet z-score",
        "threshold": THRESHOLD,
        "threshold_sweep_val": val_sweep,
        "device": device,
        "val_best_loss": round(best_val_loss, 6),
        "val": {k: val_best[k] for k in metric_keys if k in val_best},
        "test": {k: test_m[k] for k in metric_keys if k in test_m},
        "held_out_injury_baseline": baseline_injury,
        "held_out_injury_candidate": candidate_injury,
        "held_out_no_injury_baseline": baseline_noinj,
        "held_out_no_injury_candidate": candidate_noinj,
        "baseline_ood_probes": baseline_probes,
        "candidate_ood_probes": probes,
        "ood_watch_list": list(OOD_WATCH),
        "core_watch_list": list(CORE_WATCH),
        "collapse_area": COLLAPSE_AREA,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "baseline_pathological_probes": baseline_fail,
        "pathological_probes": candidate_fail,
        "ood_safety_improved": ood_improved,
        "blank_ood_pathological": collapsed,
        "training_history": history,
        "metrics_source": "computed_from_training_loop_and_held_out_predictions",
        "trained_at": trained_at,
        "leakage_free": overlap_ok,
        "known_limitations": known_limitations,
        "not_clinical_accuracy": True,
    }
    metadata = {
        "model_name": "ResNet34-UNet Segmentation",
        "version": "v1.4.0",
        "status": status,
        "training_status": status,
        "readiness_status": final_status,
        "final_status": final_status,
        "device": device,
        "training_date": trained_at,
        "training_configuration": {
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "seed": SEED,
            "device": device,
            "init": "smp Unet resnet34 imagenet, encoder frozen",
            "loss": "unweighted_BCE_plus_per_image_Dice",
            "bce_pos_weight": 1.0,
            "threshold": THRESHOLD,
            "preprocess": "256 LINEAR/NEAREST ImageNet-norm",
        },
        "metrics": metrics,
        "weights_path": UNET_CANONICAL if promoted else UNET_CANDIDATE,
        "candidate_path": UNET_CANDIDATE,
        "candidate_sha256": candidate_sha,
        "production_sha256": production_sha_after,
        "production_unchanged": production_sha_before == production_sha_after,
        "promotion": {
            "recommendation": recommendation,
            "reasons": reasons,
            "promoted_to_production": promoted,
            "rule": (
                "Promote only if hash-disjoint splits, train loss decreased, candidate SHA differs "
                "from production, core blank/OOD positive ratio is strictly better than baseline, "
                "AND remaining core collapse count is 0 (positive_ratio <= 0.05 on black/white/gray/"
                "mid_gray/blank_skin/dummy_test). In-domain Dice improvement with blank painting is not promotion."
            ),
        },
        "training_was_real": True,
        "clinically_validated": False,
        "known_limitations": known_limitations,
        "data_provenance_class": "SYNTHETIC",
        "dataset_provenance": "unet_processed unique-hash synthetic drawings. No fabricated empty masks. Not clinical photography.",
    }

    with open(os.path.join(UNET_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    pointer = {
        "latest_candidate": UNET_CANDIDATE.replace("\\", "/"),
        "latest_report": os.path.join(UNET_CANDIDATE_DIR, "TRAINING_EVAL_REPORT.json").replace("\\", "/"),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "ood_safety_improved": ood_improved,
        "baseline_ood_collapse_count": baseline_collapse_n,
        "candidate_ood_collapse_count": candidate_collapse_n,
        "production_sha256": production_sha_after,
        "candidate_sha256": candidate_sha,
        "status_remains": status,
        "final_status": final_status,
        "production_metadata_unchanged": not promoted,
    }
    with open(os.path.join(UNET_CANDIDATE_DIR, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump(pointer, handle, indent=2)
    if promoted:
        with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        register_model_artifact(
            model_name="ResNet34-UNet Segmentation",
            version="v1.4.0",
            artifact_path=UNET_CANONICAL,
            training_dataset="unet_processed",
            sample_count=len(train_rows) + len(val_rows) + len(test_rows),
            classes=["wound_mask"],
            metrics=metrics,
            training_command=r"backend\venv\Scripts\python.exe ml\training\train_unet.py",
            random_seed=SEED,
            notes="Real decoder fine-tune on unique-hash pairs. Empty masks not fabricated. MODEL_OUTPUT_NOT_TRUSTWORTHY.",
        )

    print(f"RECOMMENDATION {recommendation}")
    print(f"STATUS {status} FINAL {final_status} baseline_collapse={baseline_collapse_n} candidate_collapse={candidate_collapse_n} promoted={promoted}")
    print(f"test_dice={test_m['mean_dice']} test_fp_area={test_m.get('mean_false_positive_area')} val_loss={val_best['loss']}")
    return metadata


if __name__ == "__main__":
    train_unet()
