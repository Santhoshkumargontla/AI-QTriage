"""Build EfficientNet reject-v2 dataset: cut/bruise + real normal + ood_reject.

Classes:
  cut, bruise  — SYNTHETIC drawings (honest; no Roboflow cut/bruise photos available)
  normal       — PUBLIC_REAL peri-wound / empty-mask skin patches (AZH, Medetec)
  ood_reject   — SYNTHETIC blank/gray/noise/color canvases (explicit reject class)

Does NOT invent swelling labels. Does NOT use Mendeley (download blocked).
Does NOT use Roboflow (no API key).

Leakage controls: exact pixel-hash uniqueness; subject-aware split for real normals;
ood_reject hash-split; no exact-hash cross-split.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import ROOT

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "efficientnet_reject_v2")
CLASSES = ["cut", "bruise", "normal", "ood_reject"]
NORMAL_TARGET = 220
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
            crop = img[y : y + size, x : x + size]
            score = empty * 10 + float(crop.std()) / 255.0
            if score > best_score:
                best_score = score
                best = crop
    return best


def _load_real_normals(pool: dict, exclusions: list, near: list, dest: str) -> None:
    n_added = 0
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


def _load_ood_reject(pool: dict, exclusions: list, near: list, dest: str) -> None:
    """Explicit reject class — not remapped to an injury label."""
    rng = np.random.default_rng(SEED + 99)
    canvases = []
    for v in (0, 32, 64, 96, 128, 160, 192, 224, 255):
        canvases.append(("gray", np.full((224, 224, 3), v, np.uint8)))
    canvases.append(("black", np.zeros((224, 224, 3), np.uint8)))
    canvases.append(("white", np.full((224, 224, 3), 255, np.uint8)))
    for name, color in (
        ("blue", (200, 60, 20)),
        ("green", (40, 180, 40)),
        ("red", (30, 30, 220)),
    ):
        canvases.append((name, np.full((224, 224, 3), color, np.uint8)))
    for i in range(24):
        canvases.append((f"noise_{i}", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)))
    for i in range(8):
        base = np.full((224, 224, 3), 180, np.uint8)
        noise = rng.normal(0, 3 + i, base.shape)
        canvases.append((f"noisy_gray_{i}", np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)))
    # mild gradient unrelated scenes
    yy, xx = np.mgrid[0:224, 0:224]
    canvases.append(
        (
            "green_gradient",
            np.stack(
                [
                    (xx * 0.3).astype(np.uint8),
                    (80 + yy * 0.4).astype(np.uint8),
                    (40 + xx * 0.1).astype(np.uint8),
                ],
                axis=-1,
            ),
        )
    )
    for i, (tag, img) in enumerate(canvases):
        sample_id = f"ood_reject_{tag}_{i:03d}"
        out = os.path.join(dest, f"{sample_id}.png")
        rec = {
            "sample_id": sample_id,
            "class": "ood_reject",
            "subject_id": f"ood_reject:{tag}",
            "image_path": _rel(out),
            "source_dataset": "synthetic_ood_reject_v2",
            "source_path": sample_id,
            "provenance": "SYNTHETIC_OOD_REJECT",
            "normal_kind": "ood_reject_canvas",
            "pixel_sha256": _pixel_sha(img),
            "phash": _phash(img),
        }
        if _add_unique(pool, rec, exclusions, near, near_hamming=None):
            _write(out, img)


def _subject_aware_split(records: list[dict]) -> None:
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class"]].append(rec)

    for cls, group in by_class.items():
        if cls in {"normal", "ood_reject"}:
            subjects = sorted({r["subject_id"] for r in group})
            rng.shuffle(subjects)
            n = len(subjects)
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
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
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    pool: dict = {}
    exclusions: list = []
    near: list = []
    _load_drawing_injuries(pool, exclusions, near, dest)
    _load_real_normals(pool, exclusions, near, dest)
    _load_ood_reject(pool, exclusions, near, dest)
    records = list(pool.values())
    _subject_aware_split(records)

    by_split_subj = defaultdict(set)
    for rec in records:
        if rec["class"] in {"normal", "ood_reject"}:
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
        for rec in sorted(records, key=lambda r: (r["split"], r["class"], r["sample_id"])):
            writer.writerow(rec)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": OUT_ROOT,
        "classes": CLASSES,
        "n": len(records),
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "class_counts": dict(Counter(r["class"] for r in records)),
        "provenance_counts": dict(Counter(r["provenance"] for r in records)),
        "subject_overlap": {k: len(v) for k, v in subj_overlap.items()},
        "hash_overlap": hash_overlap,
        "exclusions_n": len(exclusions),
        "near_dup_n": len(near),
        "leakage_free": (
            hash_overlap["train_val"] == 0
            and hash_overlap["train_test"] == 0
            and hash_overlap["val_test"] == 0
            and all(len(v) == 0 for v in subj_overlap.values())
        ),
        "known_limitations": [
            "cut/bruise remain SYNTHETIC drawings — Roboflow/Kaggle credentials absent.",
            "swelling has ZERO labeled samples — not included in this classifier.",
            "normal is peri-wound / empty ulcer-patch skin, not healthy sports-injury skin.",
            "ood_reject is synthetic canvases — validated as abstention, not clinical OOD.",
            "Mendeley healthy-feet download blocked (403/404).",
        ],
    }
    with open(os.path.join(ROOT, OUT_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:20]}, handle, indent=2)
    # also mirror under data/manifests for the required pipeline layout
    man_dir = os.path.join(ROOT, "data", "manifests")
    os.makedirs(man_dir, exist_ok=True)
    shutil.copy2(man, os.path.join(man_dir, "efficientnet_reject_v2_manifest.csv"))
    with open(os.path.join(man_dir, "efficientnet_reject_v2_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
