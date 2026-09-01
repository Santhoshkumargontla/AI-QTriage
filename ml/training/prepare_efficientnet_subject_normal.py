"""Build EfficientNet set: synthetic cut/bruise + REAL normal (subject-aware).

Normal sources (honest labels, not remapped ulcers→swelling):
  - AZH patches whose mask area == 0 (empty wound mask = background skin patch)
  - Largest empty-mask crops from positive AZH/Medetec/wseg images (peri-wound skin)
  - Existing blank_skin/dummy_test as eval-only OOD files (not training)

Injury sources remain unique-hash cut/bruise drawings (SYNTHETIC) — no public
cut/bruise photo set was downloadable without a valid Roboflow key.

Splits are SUBJECT-level for real normals and hash-level for drawings.
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

from ml.models.canonical_paths import ROOT

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "efficientnet_subject_normal")
CLASSES = ["cut", "bruise", "normal"]
NORMAL_TARGET = 180
CROP_SIZE = 160
MIN_EMPTY_CROP_FRAC = 0.98


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pixel_sha(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _phash(bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(cv2.resize(bgr, (8, 8)), cv2.COLOR_BGR2GRAY)
    avg = float(gray.mean())
    bits = 0
    for i, v in enumerate(gray.flatten()):
        if v >= avg:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _write(path: str, bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)


def _add_unique(pool: dict, rec: dict, exclusions: list, near: list, near_hamming: int | None = 5) -> bool:
    digest = rec["pixel_sha256"]
    if digest in pool:
        exclusions.append({"reason": "exact_dup", "source": rec["source_path"], "kept": pool[digest]["source_path"]})
        return False
    if near_hamming is not None:
        ph = rec["phash"]
        for other in pool.values():
            if other["class"] == rec["class"] and _hamming(ph, other["phash"]) <= near_hamming:
                near.append({"reason": "near_dup", "source": rec["source_path"], "kept": other["source_path"]})
                return False
    pool[digest] = rec
    return True


def _load_drawing_injuries(pool: dict, exclusions: list, near: list, dest: str) -> None:
    man = os.path.join(ROOT, "data", "datasets", "efficientnet_processed", "manifest.csv")
    if not os.path.exists(man):
        return
    with open(man, newline="", encoding="utf-8") as handle:
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
            out = os.path.join(dest, f"{row['class']}_{row['sample_id']}.png")
            rec = {
                "sample_id": row["sample_id"],
                "class": row["class"],
                "subject_id": f"drawing_{row['sample_id']}",
                "image_path": _rel(out),
                "source_dataset": row.get("source_dataset", "efficientnet_processed"),
                "source_path": _rel(src),
                "provenance": "SYNTHETIC",
                "pixel_sha256": row.get("pixel_sha256") or _pixel_sha(img),
                "phash": _phash(img),
            }
            # Drawings are near-identical by design; exact-hash only (no phash cull).
            if _add_unique(pool, rec, exclusions, near, near_hamming=None):
                _write(out, img)


def _iter_mask_pairs():
    azh = os.path.join(ROOT, "data", "datasets", "external", "azh_patches")
    for split in ("train", "test"):
        img_dir = os.path.join(azh, split, "images")
        lab_dir = os.path.join(azh, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            yield (
                os.path.join(img_dir, name),
                os.path.join(lab_dir, name),
                name.rsplit("_", 1)[0],
                "azh_patches",
                "PUBLIC_REAL_PHOTOS",
            )
    med = os.path.join(ROOT, "data", "datasets", "external", "wound-segmentation", "data", "Medetec_foot_ulcer_224")
    for split in ("train", "test"):
        img_dir = os.path.join(med, split, "images")
        lab_dir = os.path.join(med, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            yield (
                os.path.join(img_dir, name),
                os.path.join(lab_dir, name),
                f"medetec_{os.path.splitext(name)[0]}",
                "medetec_foot_ulcer_224",
                "PUBLIC_REAL_PHOTOS",
            )


def _largest_empty_crop(img: np.ndarray, mask: np.ndarray):
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    if binary.mean() < 0.001:
        return img
    # sliding windows for empty crop
    size = min(CROP_SIZE, h, w)
    best = None
    best_score = -1.0
    step = max(16, size // 4)
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            patch = binary[y : y + size, x : x + size]
            empty = 1.0 - float(patch.mean())
            if empty < MIN_EMPTY_CROP_FRAC:
                continue
            # prefer higher variance (not pure black)
            crop = img[y : y + size, x : x + size]
            score = empty * 10 + float(crop.std()) / 255.0
            if score > best_score:
                best_score = score
                best = crop
    return best


def _load_synthetic_flat_normals(pool: dict, exclusions: list, near: list, dest: str) -> None:
    """Flat / near-uniform patches labeled normal so OOD gray/blank_skin stop collapsing to injury."""
    rng = np.random.default_rng(SEED + 7)
    tones = [
        (180, 180, 180),
        (200, 200, 200),
        (160, 160, 160),
        (185, 145, 125),
        (210, 170, 150),
        (150, 110, 95),
        (230, 200, 180),
        (120, 90, 75),
        (40, 40, 40),
        (250, 250, 250),
    ]
    for i, tone in enumerate(tones):
        for j in range(3):
            img = np.full((224, 224, 3), tone, dtype=np.uint8)
            noise = rng.normal(0, 2.0 + j, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            sample_id = f"normal_flat_{i:02d}_{j}"
            out = os.path.join(dest, f"{sample_id}.png")
            rec = {
                "sample_id": sample_id,
                "class": "normal",
                "subject_id": f"synth_flat:{i}",
                "image_path": _rel(out),
                "source_dataset": "synthetic_flat_normal",
                "source_path": sample_id,
                "provenance": "SYNTHETIC_NORMAL_OOD_GUARD",
                "normal_kind": "flat_tone_guard",
                "pixel_sha256": _pixel_sha(img),
                "phash": _phash(img),
            }
            if _add_unique(pool, rec, exclusions, near, near_hamming=None):
                _write(out, img)


def _load_real_normals(pool: dict, exclusions: list, near: list, dest: str) -> None:
    n_added = 0
    # Prefer empty-mask patches first, then empty-region crops — scan all pairs.
    pairs = list(_iter_mask_pairs())
    empty_first = []
    crops_later = []
    for ip, mp, subject, source, prov in pairs:
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if float((mask > 0).mean()) < 0.001:
            empty_first.append((ip, mp, subject, source, prov))
        else:
            crops_later.append((ip, mp, subject, source, prov))
    for ip, mp, subject, source, prov in empty_first + crops_later:
        if n_added >= NORMAL_TARGET:
            break
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            exclusions.append({"reason": "unreadable", "source": ip})
            continue
        if img.shape[:2] != mask.shape[:2]:
            exclusions.append({"reason": "dim_mismatch", "source": ip})
            continue
        area = float((mask > 0).mean())
        if area < 0.001:
            crop = img
            kind = "empty_mask_patch"
        else:
            crop = _largest_empty_crop(img, mask)
            kind = "empty_region_crop"
            if crop is None:
                continue
        if float(crop.std()) < 3.0:
            exclusions.append({"reason": "near_uniform_crop", "source": ip})
            continue
        sample_id = f"normal_{source}_{subject}_{n_added:04d}"
        out = os.path.join(dest, f"{sample_id}.png")
        rec = {
            "sample_id": sample_id,
            "class": "normal",
            "subject_id": f"{source}:{subject}",
            "image_path": _rel(out),
            "source_dataset": source,
            "source_path": _rel(ip),
            "provenance": prov,
            "normal_kind": kind,
            "pixel_sha256": _pixel_sha(crop),
            "phash": _phash(crop),
        }
        if _add_unique(pool, rec, exclusions, near, near_hamming=3):
            _write(out, crop)
            n_added += 1


def _subject_aware_split(records: list[dict]) -> None:
    """Drawings: hash split within class. Normals: subject-level split.

    Guarantee each injury class has ≥1 train sample when n≥3.
    """
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class"]].append(rec)

    for cls, group in by_class.items():
        if cls == "normal":
            subjects = sorted({r["subject_id"] for r in group})
            rng.shuffle(subjects)
            n = len(subjects)
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
            # leave at least ~50% subjects for train when possible
            while n_test + n_val >= n and (n_test > 1 or n_val > 1):
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            test_s = set(subjects[:n_test])
            val_s = set(subjects[n_test : n_test + n_val])
            for rec in group:
                if rec["subject_id"] in test_s:
                    rec["split"] = "test"
                elif rec["subject_id"] in val_s:
                    rec["split"] = "val"
                else:
                    rec["split"] = "train"
        else:
            group.sort(key=lambda r: r["pixel_sha256"])
            n = len(group)
            idx = np.arange(n)
            rng.shuffle(idx)
            if n < 3:
                for rec in group:
                    rec["split"] = "train"
                continue
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
            while n_test + n_val >= n - 1:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            for i, rec in enumerate(group):
                rank = int(np.where(idx == i)[0][0])
                if rank < n_test:
                    rec["split"] = "test"
                elif rank < n_test + n_val:
                    rec["split"] = "val"
                else:
                    rec["split"] = "train"


def build() -> dict:
    dest = os.path.join(ROOT, OUT_ROOT, "images")
    os.makedirs(dest, exist_ok=True)
    pool: dict = {}
    exclusions: list = []
    near: list = []
    _load_drawing_injuries(pool, exclusions, near, dest)
    _load_real_normals(pool, exclusions, near, dest)
    # Flat OOD guards hurt earlier (collapse 2→3); keep real normals only.
    records = list(pool.values())
    _subject_aware_split(records)

    # subject leakage check for normals
    by_split_subj = defaultdict(set)
    for rec in records:
        if rec["class"] == "normal":
            by_split_subj[rec["split"]].add(rec["subject_id"])
    subj_overlap = {
        "train_val": sorted(by_split_subj["train"] & by_split_subj["val"]),
        "train_test": sorted(by_split_subj["train"] & by_split_subj["test"]),
        "val_test": sorted(by_split_subj["val"] & by_split_subj["test"]),
    }
    by_split_hash = defaultdict(set)
    for rec in records:
        by_split_hash[rec["split"]].add(rec["pixel_sha256"])
    hash_overlap = {
        "train_val": len(by_split_hash["train"] & by_split_hash["val"]),
        "train_test": len(by_split_hash["train"] & by_split_hash["test"]),
        "val_test": len(by_split_hash["val"] & by_split_hash["test"]),
    }

    os.makedirs(os.path.join(ROOT, OUT_ROOT), exist_ok=True)
    fields = [
        "sample_id", "split", "class", "subject_id", "image_path",
        "source_dataset", "source_path", "provenance", "normal_kind",
        "pixel_sha256",
    ]
    man = os.path.join(ROOT, OUT_ROOT, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    ood_rows = []
    for path, sid, src in (
        ("data/datasets/yolo_injury/blank_skin.jpg", "blank_skin", "yolo_injury"),
        ("data/datasets/yolo_injury/dummy_test.jpg", "dummy_test", "yolo_injury"),
    ):
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            ood_rows.append({
                "sample_id": sid,
                "path": path,
                "source_dataset": src,
                "used_as_training_label": False,
                "note": "Eval-only OOD; not a training normal class member.",
            })
    with open(os.path.join(ROOT, OUT_ROOT, "ood_eval.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ood_rows[0].keys()) if ood_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(ood_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": OUT_ROOT,
        "manifest": _rel(man),
        "classes": CLASSES,
        "n": len(records),
        "class_counts": dict(Counter(r["class"] for r in records)),
        "provenance_counts": dict(Counter(r["provenance"] for r in records)),
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "class_counts_by_split": {
            split: dict(Counter(r["class"] for r in records if r["split"] == split))
            for split in ("train", "val", "test")
        },
        "normal_subjects": len({r["subject_id"] for r in records if r["class"] == "normal"}),
        "subject_overlap_normal": {k: len(v) for k, v in subj_overlap.items()},
        "hash_overlap": hash_overlap,
        "leakage_free": all(len(v) == 0 for v in subj_overlap.values()) and all(v == 0 for v in hash_overlap.values()),
        "exclusions_n": len(exclusions),
        "near_dup_excluded_n": len(near),
        "swelling_omitted": "Only 2 unique swelling drawings exist; not fabricated.",
        "known_limitations": [
            "cut/bruise remain SYNTHETIC drawings (no Roboflow key for public cut/bruise photos).",
            "normal is REAL photo patches: empty AZH masks and empty-region crops near wounds — not healthy-feet Mendeley (download blocked).",
            "Domain: wound-clinic / ulcer photography, not sports injury.",
            "Subject-aware split applies to normal subjects; drawings use unique-hash split.",
        ],
    }
    with open(os.path.join(ROOT, OUT_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:30], "near_sample": near[:20]}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
