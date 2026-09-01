"""EfficientNet dataset audit + blank/OOD reliability evaluation.

Reports raw softmax (before gates) and gated inference (after). Failures are not hidden.
"""
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.input_quality import assess_input_quality

ROOT = os.path.join("data", "datasets", "public_wound_dataset")
MANIFEST = os.path.join(ROOT, "manifest.csv")
OUT_DIR = os.path.join("ml", "models", "efficientnet_reliability")
OUT_JSON = os.path.join(OUT_DIR, "EFFICIENTNET_RELIABILITY_REPORT.json")
CLASSES = ["cut", "bruise", "swelling"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dhash(img, size=16) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())


def audit_dataset():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(MANIFEST)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_split = defaultdict(list)
    by_class = Counter()
    split_class = defaultdict(Counter)
    subjects = defaultdict(set)
    hashes = defaultdict(list)
    dhashes = defaultdict(list)
    missing = 0
    for row in rows:
        split = row["split"]
        cls = row["class"]
        by_split[split].append(row)
        by_class[cls] += 1
        split_class[split][cls] += 1
        subjects[split].add(row.get("subject_id") or "")
        path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        if not os.path.exists(path):
            missing += 1
            continue
        with open(path, "rb") as handle:
            digest = _sha256_bytes(handle.read())
        hashes[digest].append((split, row["sample_id"], cls))
        img = cv2.imread(path)
        if img is not None:
            dhashes[_dhash(img)].append((split, row["sample_id"], cls))

    exact_dup_groups = {h: v for h, v in hashes.items() if len(v) > 1}
    near_dup_groups = {h: v for h, v in dhashes.items() if len(v) > 1}
    leak_exact = []
    for group in exact_dup_groups.values():
        splits = {s for s, _, _ in group}
        if len(splits) > 1:
            leak_exact.append(group)
    leak_near = []
    for group in near_dup_groups.values():
        splits = {s for s, _, _ in group}
        if len(splits) > 1:
            leak_near.append({"count": len(group), "splits": sorted(splits), "classes": sorted({c for _, _, c in group})})

    train_subj = subjects["train"]
    val_subj = subjects["val"]
    test_subj = subjects["test"]
    return {
        "root": ROOT,
        "provenance": "SYNTHETIC_PIL_DRAWINGS — generator remaps abrasion/laceration/burn to swelling; manifest source field is not a real download.",
        "n_manifest": len(rows),
        "missing_images": missing,
        "class_balance_all": dict(by_class),
        "split_sizes": {k: len(v) for k, v in by_split.items()},
        "split_class_counts": {k: dict(v) for k, v in split_class.items()},
        "subject_counts": {k: len(v) for k, v in subjects.items()},
        "subject_overlap": {
            "train_val": sorted(train_subj & val_subj),
            "train_test": sorted(train_subj & test_subj),
            "val_test": sorted(val_subj & test_subj),
        },
        "exact_pixel_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_images": int(sum(len(v) for v in exact_dup_groups.values())),
        "cross_split_exact_duplicate_groups": len(leak_exact),
        "near_duplicate_dhash_groups": len(near_dup_groups),
        "cross_split_near_duplicate_groups": leak_near[:20],
        "cross_split_near_duplicate_group_count": len(leak_near),
        "negative_or_no_injury_labeled": int(by_class.get("none", 0) + by_class.get("normal", 0) + by_class.get("other", 0)),
        "needs_negative_class": True,
        "note": "Held-out accuracy is not evidence of generalization: templates are near-duplicates across splits.",
    }


def _summary(pred: dict) -> dict:
    return {
        "status": pred.get("__status"),
        "reason": pred.get("__reason"),
        "winner": pred.get("__winner"),
        "is_confident": pred.get("__is_confident"),
        "max_prob": pred.get("__max_prob"),
        "raw_winner": pred.get("__raw_winner"),
        "raw_max_prob": pred.get("__raw_max_prob"),
        "entropy": pred.get("__entropy"),
        "Cut": pred.get("Cut"),
        "Bruise": pred.get("Bruise"),
        "Swelling": pred.get("Swelling"),
    }


def evaluate_probes(clf: EfficientNetV2Classifier):
    rng = np.random.default_rng(42)
    probes = []

    def add(name, group, img):
        quality = assess_input_quality(img)
        raw = clf.predict_raw(img, temperature=1.0)
        gated = clf.predict(img)
        probes.append({
            "name": name,
            "group": group,
            "quality_status": quality["status"],
            "quality_reason": quality["reason"],
            "quality_metrics": quality.get("metrics", {}),
            "before_raw_softmax": raw,
            "after_gated": _summary(gated),
            "failure_hidden": False,
        })

    add("gray", "uniform", np.full((224, 224, 3), 180, dtype=np.uint8))
    add("black", "uniform", np.zeros((224, 224, 3), dtype=np.uint8))
    add("white", "uniform", np.full((224, 224, 3), 255, dtype=np.uint8))
    add("noisy_gray_std5", "uniform", np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8))
    add("skin_tone_uniform", "no_injury", np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8))
    add("skin_tone_noisy", "no_injury", np.clip(np.full((224, 224, 3), (185, 145, 125)) + rng.normal(0, 4, (224, 224, 3)), 0, 255).astype(np.uint8))

    blank_path = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank_path):
        bgr = cv2.imread(blank_path)
        add("blank_skin.jpg", "no_injury", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # In-distribution template (should remain VALID)
    id_img = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(id_img, (70, 40), (150, 180), (190, 20, 20), 7)
    add("synthetic_cut_template", "in_distribution", id_img)

    blurred = cv2.GaussianBlur(id_img, (31, 31), 8)
    add("blurred_cut_template", "blurred", blurred)
    heavy_blur = cv2.GaussianBlur(id_img, (51, 51), 16)
    add("heavily_blurred_cut_template", "blurred", heavy_blur)

    add("blue_unrelated", "unrelated", np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8))
    add("green_unrelated", "unrelated", np.full((224, 224, 3), (20, 180, 40), dtype=np.uint8))
    striped = np.zeros((224, 224, 3), dtype=np.uint8)
    striped[:, :, 2] = 200
    striped[::6, :, 1] = 180
    add("blue_striped_unrelated", "unrelated", striped)
    add("high_frequency_noise", "unrelated", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))

    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        bgr = cv2.imread(demo)
        add("football_injury.jpg", "real_photo", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # One real val image if present
    val_manifest = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "val"]
    if val_manifest:
        path = os.path.join(ROOT, val_manifest[0]["image_path"].replace("/", os.sep))
        if os.path.exists(path):
            bgr = cv2.imread(path)
            add(f"val_{val_manifest[0]['sample_id']}_{val_manifest[0]['class']}", "held_out_val", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return probes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = audit_dataset()
    clf = EfficientNetV2Classifier()
    probes = evaluate_probes(clf) if clf.is_loaded else []

    groups = defaultdict(list)
    for row in probes:
        groups[row["group"]].append({
            "name": row["name"],
            "before_winner": row["before_raw_softmax"].get("winner"),
            "before_max": row["before_raw_softmax"].get("max_prob"),
            "after_status": row["after_gated"].get("status"),
            "after_winner": row["after_gated"].get("winner"),
            "after_confident": row["after_gated"].get("is_confident"),
        })

    still_confident_invalid = [
        row["name"]
        for row in probes
        if row["group"] in {"uniform", "no_injury", "unrelated", "blurred"}
        and row["after_gated"].get("is_confident")
    ]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "not_clinical_accuracy": True,
        "do_not_trust_held_out_accuracy": True,
        "dataset_audit": audit,
        "inference": {
            "min_confidence": clf.min_confidence if clf.is_loaded else None,
            "temperature": clf.temperature if clf.is_loaded else None,
            "model_loaded": clf.is_loaded,
            "classes": list(clf.classes) if clf.is_loaded else [],
        },
        "probe_groups": dict(groups),
        "probes": probes,
        "still_confident_on_invalid": still_confident_invalid,
        "retraining_requirements": [
            "Add a real negative / no-injury class (blank skin, uniform, cluttered non-injury photos). Current set has 0.",
            "Stop remapping abrasion/laceration/burn into swelling.",
            "Replace deterministic PIL templates; current train/val/test share identical or near-identical drawings.",
            "Enforce uniqueness: no exact or dHash duplicates across splits.",
            "Train with uniform/blank/blur augmentations labeled as negative or rejected, not as swelling.",
            "Do not treat 1.0 test accuracy on 30 near-duplicate drawings as generalization.",
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {OUT_JSON}")
    print("class_balance", audit["class_balance_all"])
    print("splits", audit["split_sizes"], audit["split_class_counts"])
    print("subject_overlap", audit["subject_overlap"])
    print("cross_split_exact", audit["cross_split_exact_duplicate_groups"], "near", audit["cross_split_near_duplicate_group_count"])
    print("still_confident_on_invalid", still_confident_invalid)
    for row in probes:
        b = row["before_raw_softmax"]
        a = row["after_gated"]
        print(
            f"{row['name']:32s} before={b.get('winner')} {b.get('max_prob'):.3f}  "
            f"after={a.get('status')} winner={a.get('winner')} conf={a.get('is_confident')}"
        )


if __name__ == "__main__":
    main()
