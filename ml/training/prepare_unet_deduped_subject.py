"""Rebuild U-Net set: AZH + wseg + Medetec, exact+near dedupe, subject-aware split.

Keeps SYNTHETIC_EMPTY_TARGET canvases for blank-collapse training.
Does not remap ulcer→cut/bruise/swelling.
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
OUT_ROOT = os.path.join("data", "datasets", "unet_deduped_subject")
MAX_POS_AREA = 0.85
MIN_POS_AREA = 0.005
WSEG_CAP = 350
EMPTY_PER_KIND = 6
NEAR_HAMMING = 5


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


def _binarize(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)


def _write(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr)


def _add(pool: dict, rec: dict, exclusions: list) -> bool:
    if rec["pixel_sha256"] in pool:
        exclusions.append({"reason": "exact_dup", "source": rec["source_path"]})
        return False
    for other in pool.values():
        if _hamming(rec["phash"], other["phash"]) <= NEAR_HAMMING:
            exclusions.append({"reason": "near_dup", "source": rec["source_path"], "kept": other["source_path"]})
            return False
    pool[rec["pixel_sha256"]] = rec
    return True


def _collect_azh(pool: dict, exclusions: list, dest_i: str, dest_m: str) -> None:
    root = os.path.join(ROOT, "data", "datasets", "external", "azh_patches")
    for split in ("train", "test"):
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.lower().endswith((".png", ".jpg")):
                continue
            ip = os.path.join(img_dir, name)
            mp = os.path.join(lab_dir, name)
            img = cv2.imread(ip)
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
                exclusions.append({"reason": "bad_pair", "source": ip})
                continue
            binary = _binarize(mask)
            area = float((binary > 0).mean())
            stem = os.path.splitext(name)[0]
            subject = stem.rsplit("_", 1)[0]
            empty = area < MIN_POS_AREA
            if not empty and (area < MIN_POS_AREA or area > MAX_POS_AREA):
                exclusions.append({"reason": "area", "source": ip, "area": area})
                continue
            if empty:
                binary = np.zeros_like(binary)
                area = 0.0
            out_i = os.path.join(dest_i, f"azh_{stem}.png")
            out_m = os.path.join(dest_m, f"azh_{stem}.png")
            rec = {
                "sample_id": f"azh_{stem}",
                "kind": "empty" if empty else "positive",
                "class": "none" if empty else "wound_unspecified",
                "empty_mask": empty,
                "mask_area": round(area, 6),
                "subject_id": f"azh:{subject}",
                "source_dataset": "azh_patches",
                "source_path": _rel(ip),
                "source_mask": _rel(mp),
                "provenance": "PUBLIC_REAL_PHOTOS",
                "license": "AZH/UWM redistribution",
                "pixel_sha256": _pixel_sha(img),
                "phash": _phash(img),
                "image_path": _rel(out_i),
                "mask_path": _rel(out_m),
            }
            if _add(pool, rec, exclusions):
                _write(out_i, img)
                _write(out_m, binary)


def _collect_wseg(pool: dict, exclusions: list, dest_i: str, dest_m: str) -> None:
    sample_dir = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "sample")
    mask_dir = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "mask")
    if not os.path.isdir(sample_dir):
        return
    names = sorted(f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png")))
    stride = max(1, len(names) // WSEG_CAP)
    for name in names[::stride][:WSEG_CAP]:
        ip = os.path.join(sample_dir, name)
        mp = os.path.join(mask_dir, name)
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            exclusions.append({"reason": "bad_pair", "source": ip})
            continue
        binary = _binarize(mask)
        area = float((binary > 0).mean())
        if area < MIN_POS_AREA or area > MAX_POS_AREA:
            exclusions.append({"reason": "area", "source": ip, "area": area})
            continue
        stem = os.path.splitext(name)[0]
        out_i = os.path.join(dest_i, f"wseg_{stem}.png")
        out_m = os.path.join(dest_m, f"wseg_{stem}.png")
        rec = {
            "sample_id": f"wseg_{stem}",
            "kind": "positive",
            "class": "wound_unspecified",
            "empty_mask": False,
            "mask_area": round(area, 6),
            "subject_id": f"wseg:{stem}",
            "source_dataset": "hf_wseg_dataset",
            "source_path": _rel(ip),
            "source_mask": _rel(mp),
            "provenance": "PUBLIC_REAL_PHOTOS",
            "license": "CC-BY-NC-4.0",
            "pixel_sha256": _pixel_sha(img),
            "phash": _phash(img),
            "image_path": _rel(out_i),
            "mask_path": _rel(out_m),
        }
        if _add(pool, rec, exclusions):
            _write(out_i, img)
            _write(out_m, binary)


def _collect_medetec(pool: dict, exclusions: list, dest_i: str, dest_m: str) -> None:
    root = os.path.join(ROOT, "data", "datasets", "external", "wound-segmentation", "data", "Medetec_foot_ulcer_224")
    for split in ("train", "test"):
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.lower().endswith((".png", ".jpg")):
                continue
            ip = os.path.join(img_dir, name)
            mp = os.path.join(lab_dir, name)
            img = cv2.imread(ip)
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
                exclusions.append({"reason": "bad_pair", "source": ip})
                continue
            binary = _binarize(mask)
            area = float((binary > 0).mean())
            if area < MIN_POS_AREA or area > MAX_POS_AREA:
                exclusions.append({"reason": "area", "source": ip, "area": area})
                continue
            stem = f"medetec_{split}_{os.path.splitext(name)[0]}"
            out_i = os.path.join(dest_i, f"{stem}.png")
            out_m = os.path.join(dest_m, f"{stem}.png")
            rec = {
                "sample_id": stem,
                "kind": "positive",
                "class": "foot_ulcer",
                "empty_mask": False,
                "mask_area": round(area, 6),
                "subject_id": f"medetec:{os.path.splitext(name)[0]}",
                "source_dataset": "medetec_foot_ulcer_224",
                "source_path": _rel(ip),
                "source_mask": _rel(mp),
                "provenance": "PUBLIC_REAL_PHOTOS",
                "license": "Medetec + UWM annotated 224px",
                "pixel_sha256": _pixel_sha(img),
                "phash": _phash(img),
                "image_path": _rel(out_i),
                "mask_path": _rel(out_m),
            }
            if _add(pool, rec, exclusions):
                _write(out_i, img)
                _write(out_m, binary)


def _collect_empty_synth(pool: dict, dest_i: str, dest_m: str) -> None:
    rng = np.random.default_rng(SEED)
    specs = []
    for i in range(EMPTY_PER_KIND):
        specs.append((f"black_{i}", np.zeros((256, 256, 3), np.uint8)))
        specs.append((f"white_{i}", np.full((256, 256, 3), 255, np.uint8)))
        specs.append((f"gray_{i}", np.full((256, 256, 3), 100 + i * 12, np.uint8)))
        specs.append((f"skin_{i}", np.full((256, 256, 3), (180 + i, 140, 120), np.uint8)))
        specs.append((f"noise_{i}", rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)))
    for name, rgb in specs:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((256, 256), np.uint8)
        out_i = os.path.join(dest_i, f"empty_{name}.png")
        out_m = os.path.join(dest_m, f"empty_{name}.png")
        digest = _pixel_sha(bgr)
        if digest in pool:
            continue
        rec = {
            "sample_id": f"empty_{name}",
            "kind": "empty",
            "class": "none",
            "empty_mask": True,
            "mask_area": 0.0,
            "subject_id": f"synth_empty:{name}",
            "source_dataset": "synthetic_empty_target",
            "source_path": _rel(out_i),
            "source_mask": _rel(out_m),
            "provenance": "SYNTHETIC_EMPTY_TARGET",
            "license": "internal",
            "pixel_sha256": digest,
            "phash": _phash(bgr),
            "image_path": _rel(out_i),
            "mask_path": _rel(out_m),
        }
        pool[digest] = rec
        _write(out_i, bgr)
        _write(out_m, mask)


def _subject_split(records: list[dict]) -> None:
    rng = np.random.default_rng(SEED)
    by_kind = defaultdict(list)
    for rec in records:
        by_kind[rec["kind"]].append(rec)
    for kind, group in by_kind.items():
        subjects = sorted({r["subject_id"] for r in group})
        rng.shuffle(subjects)
        n = len(subjects)
        n_test = max(1, int(round(n * 0.15))) if n >= 6 else max(1, n // 5 or (1 if n > 2 else 0))
        n_val = max(1, int(round(n * 0.15))) if n >= 6 else max(1, n // 5 or (1 if n > 2 else 0))
        if n_test + n_val >= n:
            n_test = 1 if n > 2 else 0
            n_val = 1 if n > 3 else 0
        test_s = set(subjects[:n_test])
        val_s = set(subjects[n_test : n_test + n_val])
        for rec in group:
            if rec["subject_id"] in test_s:
                rec["split"] = "test"
            elif rec["subject_id"] in val_s:
                rec["split"] = "val"
            else:
                rec["split"] = "train"


def build() -> dict:
    dest_i = os.path.join(ROOT, OUT_ROOT, "images")
    dest_m = os.path.join(ROOT, OUT_ROOT, "masks")
    os.makedirs(dest_i, exist_ok=True)
    os.makedirs(dest_m, exist_ok=True)
    pool: dict = {}
    exclusions: list = []
    _collect_azh(pool, exclusions, dest_i, dest_m)
    _collect_wseg(pool, exclusions, dest_i, dest_m)
    _collect_medetec(pool, exclusions, dest_i, dest_m)
    _collect_empty_synth(pool, dest_i, dest_m)
    records = list(pool.values())
    _subject_split(records)

    by_split_subj = defaultdict(set)
    by_split_hash = defaultdict(set)
    for rec in records:
        by_split_subj[rec["split"]].add(rec["subject_id"])
        by_split_hash[rec["split"]].add(rec["pixel_sha256"])
    subj_overlap = {
        "train_val": sorted(by_split_subj["train"] & by_split_subj["val"]),
        "train_test": sorted(by_split_subj["train"] & by_split_subj["test"]),
        "val_test": sorted(by_split_subj["val"] & by_split_subj["test"]),
    }
    hash_overlap = {
        "train_val": len(by_split_hash["train"] & by_split_hash["val"]),
        "train_test": len(by_split_hash["train"] & by_split_hash["test"]),
        "val_test": len(by_split_hash["val"] & by_split_hash["test"]),
    }

    os.makedirs(os.path.join(ROOT, OUT_ROOT), exist_ok=True)
    fields = [
        "sample_id", "split", "kind", "class", "empty_mask", "mask_area", "subject_id",
        "image_path", "mask_path", "source_dataset", "source_path", "source_mask",
        "provenance", "license", "pixel_sha256",
    ]
    man = os.path.join(ROOT, OUT_ROOT, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": OUT_ROOT,
        "manifest": _rel(man),
        "n": len(records),
        "kind_counts": dict(Counter(r["kind"] for r in records)),
        "source_counts": dict(Counter(r["source_dataset"] for r in records)),
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "split_by_kind": {
            split: dict(Counter(r["kind"] for r in records if r["split"] == split))
            for split in ("train", "val", "test")
        },
        "subjects": len({r["subject_id"] for r in records}),
        "subject_overlap": {k: len(v) for k, v in subj_overlap.items()},
        "hash_overlap": hash_overlap,
        "near_dup_hamming": NEAR_HAMMING,
        "leakage_free": all(len(v) == 0 for v in subj_overlap.values()) and all(v == 0 for v in hash_overlap.values()),
        "exclusions_n": len(exclusions),
        "blank_collapse_gate": "CORE_WATCH positive_ratio <= 0.05 required for promotion",
        "known_limitations": [
            "Binary wound vs background on chronic/ulcer photos + synthetic empty canvases.",
            "Not sports-injury cut/bruise segmentation.",
            "wseg is CC-BY-NC-4.0.",
        ],
    }
    with open(os.path.join(ROOT, OUT_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:40]}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
