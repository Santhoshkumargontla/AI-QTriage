"""Build a U-Net set from public wound photos plus documented empty synthetics.

Sources:
  - HuggingFace subbareddyoota/wseg_dataset (CC-BY-NC-4.0) sample/ + mask/
  - UWM Medetec_foot_ulcer_224 (public redistribution; foot-ulcer domain)
  - SYNTHETIC_EMPTY_TARGET canvases (not medical scenes)

Does not remap ulcers to cut/bruise/swelling. Hash-disjoint splits.
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

from ml.models.canonical_paths import ROOT, UNET_PUBLIC_MANIFEST, UNET_PUBLIC_ROOT

SEED = 42
MAX_POS_AREA = 0.85
MIN_POS_AREA = 0.005
WSEG_CAP = 400
EMPTY_PER_KIND = 8


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pixel_sha256(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _binarize(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)


def _write_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def _add(pool: dict, rec: dict, exclusions: list) -> bool:
    digest = rec["pixel_sha256"]
    if digest in pool:
        exclusions.append({"reason": "exact_pixel_duplicate", "kept": pool[digest]["source_path"], "source": rec["source_path"]})
        return False
    pool[digest] = rec
    return True


def _split_by_hash(records: list[dict]) -> None:
    by = defaultdict(list)
    for rec in records:
        by[rec["kind"]].append(rec)
    rng = np.random.default_rng(SEED)
    for kind, group in by.items():
        group.sort(key=lambda r: r["pixel_sha256"])
        n = len(group)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * 0.15))) if n >= 8 else max(1, n // 6 or (1 if n > 2 else 0))
        n_val = max(1, int(round(n * 0.15))) if n >= 8 else max(1, n // 6 or (1 if n > 2 else 0))
        if n_test + n_val >= n:
            n_test = 1 if n > 2 else 0
            n_val = 1 if n > 3 else 0
        for i, rec in enumerate(group):
            rank = int(np.where(idx == i)[0][0])
            if rank < n_test:
                rec["split"] = "test"
            elif rank < n_test + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "train"


def _collect_wseg(pool: dict, exclusions: list, dest_img: str, dest_mask: str) -> None:
    sample_dir = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "sample")
    mask_dir = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "mask")
    if not os.path.isdir(sample_dir):
        print("wseg sample dir missing")
        return
    names = sorted(f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    stride = max(1, len(names) // WSEG_CAP)
    picked = names[::stride][:WSEG_CAP]
    print(f"wseg considering {len(picked)}/{len(names)}")
    for name in picked:
        ip = os.path.join(sample_dir, name)
        mp = os.path.join(mask_dir, name)
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            exclusions.append({"reason": "unreadable", "source": ip})
            continue
        if img.shape[:2] != mask.shape[:2]:
            exclusions.append({"reason": "dimension_mismatch", "source": ip})
            continue
        binary = _binarize(mask)
        area = float((binary > 0).mean())
        if area < MIN_POS_AREA:
            exclusions.append({"reason": "empty_mask_on_positive", "source": ip, "area": area})
            continue
        if area > MAX_POS_AREA:
            exclusions.append({"reason": "unreasonable_full_mask", "source": ip, "area": area})
            continue
        stem = os.path.splitext(name)[0]
        out_i = os.path.join(dest_img, f"wseg_{stem}.png")
        out_m = os.path.join(dest_mask, f"wseg_{stem}.png")
        rec = {
            "sample_id": f"wseg_{stem}",
            "kind": "positive",
            "class": "wound_unspecified",
            "empty_mask": False,
            "mask_area": round(area, 6),
            "source_dataset": "hf_wseg_dataset",
            "source_path": _rel(ip),
            "source_mask": _rel(mp),
            "provenance": "PUBLIC_REAL_PHOTOS",
            "license": "CC-BY-NC-4.0",
            "pixel_sha256": _pixel_sha256(img),
            "image_path": _rel(out_i),
            "mask_path": _rel(out_m),
        }
        if _add(pool, rec, exclusions):
            _write_png(out_i, img)
            _write_png(out_m, binary)


def _collect_medetec(pool: dict, exclusions: list, dest_img: str, dest_mask: str) -> None:
    root = os.path.join(ROOT, "data", "datasets", "external", "wound-segmentation", "data", "Medetec_foot_ulcer_224")
    for split in ("train", "test"):
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ip = os.path.join(img_dir, name)
            mp = os.path.join(lab_dir, name)
            img = cv2.imread(ip)
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                exclusions.append({"reason": "unreadable", "source": ip})
                continue
            if img.shape[:2] != mask.shape[:2]:
                exclusions.append({"reason": "dimension_mismatch", "source": ip})
                continue
            binary = _binarize(mask)
            area = float((binary > 0).mean())
            if area < MIN_POS_AREA:
                exclusions.append({"reason": "empty_mask_on_positive", "source": ip, "area": area})
                continue
            if area > MAX_POS_AREA:
                exclusions.append({"reason": "unreasonable_full_mask", "source": ip, "area": area})
                continue
            stem = f"medetec_{split}_{os.path.splitext(name)[0]}"
            out_i = os.path.join(dest_img, f"{stem}.png")
            out_m = os.path.join(dest_mask, f"{stem}.png")
            rec = {
                "sample_id": stem,
                "kind": "positive",
                "class": "foot_ulcer",
                "empty_mask": False,
                "mask_area": round(area, 6),
                "source_dataset": "medetec_foot_ulcer_224",
                "source_path": _rel(ip),
                "source_mask": _rel(mp),
                "provenance": "PUBLIC_REAL_PHOTOS",
                "license": "Medetec stock terms + UWM annotated 224px redistribution",
                "pixel_sha256": _pixel_sha256(img),
                "image_path": _rel(out_i),
                "mask_path": _rel(out_m),
            }
            if _add(pool, rec, exclusions):
                _write_png(out_i, img)
                _write_png(out_m, binary)


def _collect_empty_synth(pool: dict, dest_img: str, dest_mask: str) -> None:
    rng = np.random.default_rng(SEED)
    specs = []
    for i in range(EMPTY_PER_KIND):
        specs.append((f"black_{i}", np.zeros((256, 256, 3), np.uint8)))
        specs.append((f"white_{i}", np.full((256, 256, 3), 255, np.uint8)))
        specs.append((f"gray_{i}", np.full((256, 256, 3), 128 + (i % 5) * 10, np.uint8)))
        specs.append((f"skin_{i}", np.full((256, 256, 3), (180 + i, 140, 120), np.uint8)))
        noise = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        specs.append((f"noise_{i}", noise))
        grad = np.tile(np.linspace(40, 200, 256, dtype=np.uint8)[:, None], (1, 256))
        specs.append((f"grad_{i}", cv2.merge([grad, grad, np.full_like(grad, 90 + i)])))
    for name, rgb in specs:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((256, 256), np.uint8)
        out_i = os.path.join(dest_img, f"empty_{name}.png")
        out_m = os.path.join(dest_mask, f"empty_{name}.png")
        rec = {
            "sample_id": f"empty_{name}",
            "kind": "empty",
            "class": "none",
            "empty_mask": True,
            "mask_area": 0.0,
            "source_dataset": "synthetic_empty_target",
            "source_path": _rel(out_i),
            "source_mask": _rel(out_m),
            "provenance": "SYNTHETIC_EMPTY_TARGET",
            "license": "internal research (not medical data)",
            "pixel_sha256": _pixel_sha256(bgr),
            "image_path": _rel(out_i),
            "mask_path": _rel(out_m),
        }
        if rec["pixel_sha256"] in pool:
            continue
        pool[rec["pixel_sha256"]] = rec
        _write_png(out_i, bgr)
        _write_png(out_m, mask)


def build() -> dict:
    dest_img = os.path.join(ROOT, UNET_PUBLIC_ROOT, "images")
    dest_mask = os.path.join(ROOT, UNET_PUBLIC_ROOT, "masks")
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_mask, exist_ok=True)
    pool: dict = {}
    exclusions: list = []
    _collect_wseg(pool, exclusions, dest_img, dest_mask)
    _collect_medetec(pool, exclusions, dest_img, dest_mask)
    _collect_empty_synth(pool, dest_img, dest_mask)
    records = list(pool.values())
    _split_by_hash(records)
    os.makedirs(os.path.join(ROOT, UNET_PUBLIC_ROOT), exist_ok=True)
    fields = [
        "sample_id", "split", "kind", "class", "empty_mask", "mask_area",
        "image_path", "mask_path", "source_dataset", "source_path", "source_mask",
        "provenance", "license", "pixel_sha256",
    ]
    with open(os.path.join(ROOT, UNET_PUBLIC_MANIFEST), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    overlap = {"train_val": [], "train_test": [], "val_test": []}
    by_split = defaultdict(set)
    for rec in records:
        by_split[rec["split"]].add(rec["pixel_sha256"])
    overlap["train_val"] = sorted(by_split["train"] & by_split["val"])
    overlap["train_test"] = sorted(by_split["train"] & by_split["test"])
    overlap["val_test"] = sorted(by_split["val"] & by_split["test"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(records),
        "kind_counts": dict(Counter(r["kind"] for r in records)),
        "source_counts": dict(Counter(r["source_dataset"] for r in records)),
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "split_by_kind": {
            split: dict(Counter(r["kind"] for r in records if r["split"] == split))
            for split in ("train", "val", "test")
        },
        "hash_overlap": {k: len(v) for k, v in overlap.items()},
        "leakage_free": all(len(v) == 0 for v in overlap.values()),
        "exclusions_n": len(exclusions),
        "domain_note": "Binary wound-vs-background. Not cut/bruise/swelling. Foot-ulcer/chronic-wound photographs plus synthetic empty canvases.",
        "known_limitations": [
            "wseg is CC-BY-NC-4.0 (academic/non-commercial).",
            "Medetec is foot-ulcer stock photography, not sports injury.",
            "Empty canvases are SYNTHETIC_EMPTY_TARGET, not healthy-skin photography.",
            "Mendeley healthy-feet zip was not downloadable (no public direct URL / 403).",
        ],
    }
    with open(os.path.join(ROOT, UNET_PUBLIC_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:40]}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
