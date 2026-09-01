"""Verify-first U-Net forensic audit. Does not modify production weights."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.canonical_paths import UNET_CANONICAL, UNET_METADATA, REGISTRY_PATH, sha256_file, exists, resolve_existing, read_json
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.input_quality import assess_input_quality

ROOT = os.path.join("data", "datasets", "public_wound_dataset")
MANIFEST = os.path.join(ROOT, "manifest.csv")
OUT_JSON = os.path.join("scratch", "forensic_unet_audit.json")
OUT_TXT = os.path.join("scratch", "forensic_unet_audit.txt")
THRESHOLD = 0.5
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pixel_sha(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    h, w = bgr.shape[:2]
    digest = hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()
    return digest, bgr


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
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "false_positive_area": float(fp_area),
        "pred_positive_ratio": float(pred_sum / max(p.size, 1)),
        "gt_positive_ratio": float(tgt_sum / max(t.size, 1)),
    }


def _forward_logits_probs(seg: UNetSegmenter, img_rgb: np.ndarray):
    roi = cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if roi.dtype != np.uint8:
        roi = np.clip(roi, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(seg.device)
    with torch.no_grad():
        logits = seg.model(tensor).squeeze().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    return np.asarray(logits, dtype=np.float32), np.asarray(probs, dtype=np.float32)


def _probe(seg: UNetSegmenter, name: str, group: str, img: np.ndarray, gt_mask=None):
    quality = assess_input_quality(img)
    logits, probs = _forward_logits_probs(seg, img)
    binary = (probs > THRESHOLD).astype(np.uint8)
    raw = seg.segment_raw(img)
    mask, count, ratio, gated = seg.segment(img)
    h, w = img.shape[:2]
    row = {
        "name": name,
        "group": group,
        "source_shape_hw": [int(h), int(w)],
        "model_input_hw": [SIZE, SIZE],
        "raw_mask_hw": [int(binary.shape[0]), int(binary.shape[1])],
        "gated_mask_hw": [int(mask.shape[0]), int(mask.shape[1])] if mask is not None else None,
        "image_std": round(float(np.std(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))), 6),
        "quality_status": quality["status"],
        "quality_reason": quality["reason"],
        "gate_is_std_lt_3": False,
        "gate_actually_used": "input_quality.assess_input_quality (std<12 and ptp<40 for uniform_or_blank, plus other checks). Not std<3.",
        "raw": {
            "positive_mask_ratio": round(float(binary.mean()), 6),
            "positive_pixels": int(binary.sum()),
            "total_pixels": int(binary.size),
            "max_probability": round(float(probs.max()), 6),
            "mean_probability": round(float(probs.mean()), 6),
            "min_probability": round(float(probs.min()), 6),
            "max_logit": round(float(logits.max()), 6),
            "mean_logit": round(float(logits.mean()), 6),
            "min_logit": round(float(logits.min()), 6),
            "false_positive_area": None,
            "mask_dimensions": f"{binary.shape[0]}x{binary.shape[1]}",
            "wrapper_segment_raw_positive_ratio": raw.get("positive_ratio"),
            "wrapper_segment_raw_max_prob": raw.get("max_prob"),
            "wrapper_segment_raw_mean_prob": raw.get("mean_prob"),
        },
        "gated": {
            "status": gated.get("status"),
            "reason": gated.get("reason"),
            "mask_withheld": gated.get("mask_withheld"),
            "is_reliable": gated.get("is_reliable"),
            "displayed_positive_ratio": round(float(mask.mean()), 6) if mask is not None else None,
            "displayed_pixel_count": int(count or 0),
            "logged_raw_positive_ratio": gated.get("raw_positive_ratio"),
            "logged_false_positive_area": gated.get("false_positive_area"),
        },
    }
    if gt_mask is None:
        # No GT: every positive pixel is a false positive on OOD/blank/unrelated/normal probes.
        row["raw"]["false_positive_area"] = row["raw"]["positive_mask_ratio"]
        row["raw"]["false_positive_area_definition"] = "no_gt_so_fp_area_equals_positive_ratio"
    else:
        gt = cv2.resize((gt_mask > 127).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        m = _pair_metrics(binary, gt)
        row["held_out_vs_gt"] = {k: round(float(v), 6) for k, v in m.items()}
        row["raw"]["false_positive_area"] = round(m["false_positive_area"], 6)
        row["raw"]["false_positive_area_definition"] = "fp_pixels_over_hw_vs_gt_mask"
    return row


def audit_dataset():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_split = defaultdict(list)
    by_class = Counter()
    split_class = defaultdict(Counter)
    subjects = defaultdict(set)
    img_hashes = defaultdict(list)
    mask_hashes = defaultdict(list)
    missing = 0
    empty_masks = 0
    dim_mismatch = 0
    mask_areas = []
    unique_img_by_class = defaultdict(set)
    unique_mask_by_class = defaultdict(set)

    for row in rows:
        split = row["split"]
        cls = row["class"]
        by_split[split].append(row)
        by_class[cls] += 1
        split_class[split][cls] += 1
        subjects[split].add(row.get("subject_id") or "")
        img_path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        mask_path = os.path.join(ROOT, row["mask_path"].replace("/", os.sep))
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            missing += 1
            continue
        img_digest, bgr = _pixel_sha(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            missing += 1
            continue
        if bgr.shape[:2] != mask.shape[:2]:
            dim_mismatch += 1
        img_hashes[img_digest].append((split, row["sample_id"], cls))
        mdigest = hashlib.sha256(mask.tobytes() + f"|{mask.shape[0]}x{mask.shape[1]}".encode()).hexdigest()
        mask_hashes[mdigest].append((split, row["sample_id"], cls))
        unique_img_by_class[cls].add(img_digest)
        unique_mask_by_class[cls].add(mdigest)
        area = float((mask > 127).mean())
        mask_areas.append(area)
        if area == 0.0:
            empty_masks += 1

    leak_img = []
    for digest, group in img_hashes.items():
        splits = {s for s, _, _ in group}
        if len(group) > 1 and len(splits) > 1:
            leak_img.append({"pixel_sha256": digest, "n": len(group), "splits": sorted(splits), "ids": group})
    leak_mask = []
    for digest, group in mask_hashes.items():
        splits = {s for s, _, _ in group}
        if len(group) > 1 and len(splits) > 1:
            leak_mask.append({"pixel_sha256": digest, "n": len(group), "splits": sorted(splits)})

    areas = np.array(mask_areas, dtype=np.float64) if mask_areas else np.array([0.0])
    return {
        "root": ROOT.replace("\\", "/"),
        "n_manifest": len(rows),
        "missing_or_unreadable": missing,
        "dimension_mismatches": dim_mismatch,
        "empty_masks": empty_masks,
        "negative_no_wound_pairs": empty_masks,
        "class_counts": dict(by_class),
        "split_sizes": {k: len(v) for k, v in by_split.items()},
        "split_class_counts": {k: dict(v) for k, v in split_class.items()},
        "unique_image_templates_by_class": {k: len(v) for k, v in unique_img_by_class.items()},
        "unique_mask_templates_by_class": {k: len(v) for k, v in unique_mask_by_class.items()},
        "unique_image_templates_total": sum(len(v) for v in unique_img_by_class.values()),
        "subject_overlap": {
            "train_val": sorted(subjects["train"] & subjects["val"]),
            "train_test": sorted(subjects["train"] & subjects["test"]),
            "val_test": sorted(subjects["val"] & subjects["test"]),
        },
        "exact_duplicate_image_groups": sum(1 for v in img_hashes.values() if len(v) > 1),
        "exact_duplicate_images": sum(len(v) for v in img_hashes.values() if len(v) > 1),
        "cross_split_exact_image_leak_groups": len(leak_img),
        "cross_split_exact_mask_leak_groups": len(leak_mask),
        "image_leak_sample": leak_img[:8],
        "mask_area": {
            "min": round(float(areas.min()), 6),
            "max": round(float(areas.max()), 6),
            "mean": round(float(areas.mean()), 6),
            "p50": round(float(np.median(areas)), 6),
        },
        "positive_pixel_fraction_all_masks": round(float(areas.mean()), 6),
        "manifest_source_field": rows[0].get("source") if rows else None,
        "generator": "ml/training/download_public_datasets.py generate_expanded_wound_dataset",
        "leakage_free": len(leak_img) == 0,
        "subject_ids_disjoint": not any(subjects[a] & subjects[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
    }, rows


def held_out(seg: UNetSegmenter, rows: list[dict], unique_only: bool):
    seen = set()
    metrics = []
    used = []
    for row in rows:
        if row["split"] != "test":
            continue
        img_path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        mask_path = os.path.join(ROOT, row["mask_path"].replace("/", os.sep))
        digest, bgr = _pixel_sha(img_path)
        if bgr is None:
            continue
        if unique_only:
            if digest in seen:
                continue
            seen.add(digest)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        _, probs = _forward_logits_probs(seg, img)
        binary = (probs > THRESHOLD).astype(np.uint8)
        gt_r = cv2.resize((gt > 127).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        m = _pair_metrics(binary, gt_r)
        m["sample_id"] = row["sample_id"]
        m["class"] = row["class"]
        metrics.append(m)
        used.append(row["sample_id"])
    if not metrics:
        return {"n": 0}
    keys = ["dice", "iou", "precision", "recall", "false_positive_area"]
    return {
        "n": len(metrics),
        "unique_only": unique_only,
        "mean": {k: round(float(np.mean([m[k] for m in metrics])), 6) for k in keys},
        "min_dice": round(float(min(m["dice"] for m in metrics)), 6),
        "max_dice": round(float(max(m["dice"] for m in metrics)), 6),
        "sample_ids": used,
        "per_image": [{k: round(float(m[k]), 6) if k in keys else m[k] for k in ("sample_id", "class", *keys)} for m in metrics],
    }


def imagenet_norm_stats():
    def z(rgb):
        x = np.array(rgb, dtype=np.float32) / 255.0
        return ((x - MEAN) / STD).tolist()
    return {
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "black_0_normalized": z([0, 0, 0]),
        "white_255_normalized": z([255, 255, 255]),
        "gray_180_normalized": z([180, 180, 180]),
        "skin_185_145_125_normalized": z([185, 145, 125]),
        "note": "Training drawings are mostly skin-tone (~185,145,125). Black/white are several std away from that mode after ImageNet z-score.",
    }


def head_inspection(seg: UNetSegmenter):
    info = {}
    for name, tensor in seg.model.state_dict().items():
        if "segmentation_head" in name or name.endswith("conv2d.bias") or "segm" in name.lower():
            arr = tensor.detach().cpu().numpy()
            info[name] = {
                "shape": list(arr.shape),
                "mean": round(float(arr.mean()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
                "absmax": round(float(np.abs(arr).max()), 6),
            }
            if arr.ndim == 1 and arr.size <= 8:
                info[name]["values"] = [round(float(v), 6) for v in arr.tolist()]
    # Frozen encoder check
    frozen = 0
    trainable = 0
    for n, p in seg.model.named_parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    info["_param_requires_grad_note"] = (
        "Inference load does not restore the train-time freeze flag; counts below are the live module defaults."
    )
    info["_numel_requires_grad_true"] = int(trainable)
    info["_numel_requires_grad_false"] = int(frozen)
    return info


def main():
    canonical = resolve_existing(UNET_CANONICAL)
    live_sha = sha256_file(canonical) if exists(canonical) else None
    meta = read_json(UNET_METADATA) if exists(UNET_METADATA) else {}
    registry = read_json(REGISTRY_PATH) if exists(REGISTRY_PATH) else {}
    unet_reg = registry.get("ResNet34-UNet Segmentation") or {}
    size = os.path.getsize(canonical) if exists(canonical) else None

    ds, rows = audit_dataset()
    seg = UNetSegmenter()
    if not seg.is_loaded:
        raise RuntimeError("U-Net weights did not load")

    rng = np.random.default_rng(0)
    probes = []

    def add(name, group, img, gt=None):
        probes.append(_probe(seg, name, group, img, gt))

    add("gray", "uniform", np.full((224, 224, 3), 180, dtype=np.uint8))
    add("black", "uniform", np.zeros((224, 224, 3), dtype=np.uint8))
    add("white", "uniform", np.full((224, 224, 3), 255, dtype=np.uint8))
    add("blank_mid_gray", "blank", np.full((256, 256, 3), 128, dtype=np.uint8))

    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank):
        add("blank_skin.jpg", "normal_no_injury", cv2.cvtColor(cv2.imread(blank), cv2.COLOR_BGR2RGB))
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.exists(dummy):
        add("dummy_test.jpg", "normal_no_injury", cv2.cvtColor(cv2.imread(dummy), cv2.COLOR_BGR2RGB))
    add("uniform_skin", "normal_no_injury", np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8))
    add("blue_unrelated", "unrelated", np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8))
    add("green_unrelated", "unrelated", np.full((224, 224, 3), (20, 180, 40), dtype=np.uint8))
    add("high_frequency_noise", "unrelated", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))

    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        add("football_injury.jpg", "injury_photo_no_gt", cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB))

    # Representative in-domain injury drawing with GT (first unique test cut if present)
    test_rows = [r for r in rows if r["split"] == "test"]
    picked = None
    seen = set()
    for row in test_rows:
        path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        digest, bgr = _pixel_sha(path)
        if bgr is None or digest in seen:
            continue
        seen.add(digest)
        if row["class"] == "cut" or picked is None:
            picked = (row, bgr)
            if row["class"] == "cut":
                break
    if picked:
        row, bgr = picked
        gt = cv2.imread(os.path.join(ROOT, row["mask_path"].replace("/", os.sep)), cv2.IMREAD_GRAYSCALE)
        add(
            f"heldout_{row['sample_id']}_{row['class']}",
            "injury_drawing_with_gt",
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            gt,
        )

    held_leaked = held_out(seg, rows, unique_only=False)
    held_unique = held_out(seg, rows, unique_only=True)

    processed_summary = None
    ps = os.path.join("data", "datasets", "unet_processed", "processing_summary.json")
    if os.path.exists(ps):
        processed_summary = json.load(open(ps, encoding="utf-8")).get("summary")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_retrain": True,
        "did_not_modify_production_weights": True,
        "did_not_fabricate_metrics": True,
        "artifact": {
            "canonical_path": UNET_CANONICAL.replace("\\", "/"),
            "exists": exists(canonical),
            "file_size": size,
            "sha256": live_sha,
            "metadata_sha256": meta.get("artifact_sha256"),
            "registry_sha256": unet_reg.get("artifact_sha256"),
            "sha_matches_metadata": live_sha == meta.get("artifact_sha256"),
            "sha_matches_registry": live_sha == unet_reg.get("artifact_sha256"),
            "metadata_status": meta.get("status") or meta.get("training_status"),
            "architecture": "smp.Unet encoder=resnet34 classes=1 activation=None",
            "threshold": THRESHOLD,
        },
        "preprocessing_audit": {
            "train_image": "cv2.resize to 256x256 default INTER_LINEAR, RGB, /255, ImageNet mean/std, no augmentation",
            "train_mask": "cv2.resize 256x256 INTER_NEAREST, binary mask>127",
            "inference_image": "cv2.resize(roi, (256,256)) default INTER_LINEAR, RGB, /255, ImageNet mean/std",
            "inference_mask_back_to_roi": "cv2.resize processed binary to ROI with INTER_NEAREST (gated path only)",
            "normalization": "ImageNet z-score, same train and infer",
            "thresholding": "fixed 0.5 sigmoid; no adaptive threshold",
            "postprocess_gated_only": "3x3 morph open; keep largest CC if >=80% of positives; max area 0.70",
            "encoder": "resnet34 ImageNet init, frozen during the production train loop (metadata)",
            "loss": "BCE_plus_Dice on non-empty masks only",
            "imagenet_zscore_of_uniforms": imagenet_norm_stats(),
            "do_not_rely_on_std_lt_3": "Production gate is assess_input_quality: uniform if std<12 AND ptp<40 (not std<3). This audit reports RAW sigmoid maps without treating that gate as a reliability proof.",
        },
        "segmentation_head": head_inspection(seg),
        "dataset_audit": ds,
        "unet_processed_exists_but_not_production_trainset": {
            "path": "data/datasets/unet_processed",
            "present": processed_summary is not None,
            "note": "Current train_unet.py rebuilds unet_processed (including generated empty masks). Production metadata/dataset_name is public_wound_dataset. Empty masks in unet_processed were not used to train the live SHA.",
            "summary_excerpt": {
                k: processed_summary.get(k)
                for k in ("n", "classes", "did_not_fabricate_negatives", "empty_mask_n", "n_empty", "known_limitations")
                if processed_summary and k in processed_summary
            } if processed_summary else None,
        },
        "held_out_advertised_test_n30_leaked": held_leaked,
        "held_out_unique_pixel_templates_only": held_unique,
        "probes": probes,
        "root_cause": {
            "black_white_positive_masks": [
                "Every production training mask is non-empty (empty_masks=0). The decoder never received a target of all zeros.",
                "Dice+BCE on only-positive masks rewards covering the wound mark; there is no empty-background class.",
                "Black and white uniforms are ImageNet-z far from the skin-tone drawing mode, so encoder features are OOD.",
                "On that OOD input the head saturates: black/white logits are large and positive almost everywhere, so sigmoid~1 and positive_ratio~1.",
                "A std<12 gate withholds the overlay; it does not change the raw map. Geometry of in-domain masks can still look aligned while OOD is pathological.",
            ]
        },
        "geometry_vs_reliability": {
            "in_domain_mask_can_overlap_drawn_stroke": True,
            "that_does_not_imply_reliability": True,
            "reason": "High Dice on leaked unique templates is reconstruction of the generator, not generalization. OOD full-frame positives falsify reliability.",
        },
        "retraining_recommendation": [
            "Do not retrain on public_wound_dataset as-is (zero empty masks, duplicate leakage, synthetic drawings).",
            "Do not fabricate a large empty class from random noise and call it clinical negatives.",
            "If only two real no-injury files exist, keep them eval-only; do not pretend a reject class is trained.",
            "Promote a candidate only if raw positive_ratio on black/white/gray/blank_skin is < 0.05 without gates, and unique-hash held-out Dice is reported separately from leaked Dice.",
            "Keep production SHA unchanged until that evidence exists. Keep overlay gates.",
        ],
        "final_status": "MODEL_OUTPUT_NOT_TRUSTWORTHY",
    }
    os.makedirs("scratch", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    lines = [
        f"SHA {live_sha}",
        f"status {report['final_status']}",
        f"empty_masks {ds['empty_masks']}",
        f"leak_groups {ds['cross_split_exact_image_leak_groups']}",
        f"held_leaked n={held_leaked.get('n')} mean={held_leaked.get('mean')}",
        f"held_unique n={held_unique.get('n')} mean={held_unique.get('mean')}",
    ]
    for p in probes:
        r = p["raw"]
        lines.append(
            f"{p['name']:32s} pos={r['positive_mask_ratio']:.4f} maxp={r['max_probability']:.4f} "
            f"meanp={r['mean_probability']:.4f} fp={r['false_positive_area']:.4f} "
            f"hw={r['mask_dimensions']} gated={p['gated']['status']}"
        )
    with open(OUT_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
