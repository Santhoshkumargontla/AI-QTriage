"""Build YOLO detect set with real wound boxes from public masks + existing cut/bruise.

Class map (fixed):
  0 cut    — from yolo_retrain_v2 / yolo_processed (mostly SYNTHETIC drawings)
  1 bruise — same
  2 wound  — bbox from AZH / Medetec / wseg binary masks (PUBLIC_REAL_PHOTOS)

Subject-aware split for wound subjects. Cut/bruise keep hash-disjoint stems.
Does not invent cut/bruise photo labels. Roboflow download remains blocked without API key.
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
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import ROOT

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "yolo_wound_boxes_v1")
NAMES = {0: "cut", 1: "bruise", 2: "wound"}
WSEG_CAP = 250
MIN_AREA = 0.005
MAX_AREA = 0.85


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pixel_sha(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _mask_to_yolo_line(mask: np.ndarray, class_id: int = 2) -> str | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.mean())
    if area < MIN_AREA or area > MAX_AREA:
        return None
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    nw = bw / w
    nh = bh / h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _copy_cut_bruise(records: list, exclusions: list) -> None:
    src_root = os.path.join(ROOT, "data", "datasets", "yolo_retrain_v2")
    man = os.path.join(src_root, "manifest.csv")
    if not os.path.exists(man):
        return
    with open(man, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        img = row.get("dest_image") or ""
        lbl = row.get("dest_label") or ""
        img_path = os.path.join(ROOT, img.replace("/", os.sep))
        lbl_path = os.path.join(ROOT, lbl.replace("/", os.sep))
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            exclusions.append({"reason": "missing", "source": img})
            continue
        lines = []
        for ln in open(lbl_path, encoding="utf-8"):
            parts = ln.split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if cid in (0, 1):  # cut, bruise only — drop legacy wound if any
                lines.append(ln.strip())
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        records.append({
            "sample_id": f"legacy_{stem}",
            "subject_id": f"drawing:{stem}",
            "class_ids": sorted({int(float(ln.split()[0])) for ln in lines}) if lines else [],
            "lines": lines,
            "image_bgr": bgr,
            "provenance": row.get("provenance") or "SYNTHETIC",
            "source_dataset": "yolo_retrain_v2",
            "source_path": _rel(img_path),
            "pixel_sha256": _pixel_sha(bgr),
            "empty": len(lines) == 0,
        })


def _add_wound_from_masks(records: list, exclusions: list, seen_hash: set) -> None:
    pairs = []
    azh = os.path.join(ROOT, "data", "datasets", "external", "azh_patches")
    for split in ("train", "test"):
        img_dir = os.path.join(azh, split, "images")
        lab_dir = os.path.join(azh, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith(".png"):
                continue
            pairs.append((
                os.path.join(img_dir, name),
                os.path.join(lab_dir, name),
                f"azh:{name.rsplit('_', 1)[0]}",
                "azh_patches",
            ))
    med = os.path.join(ROOT, "data", "datasets", "external", "wound-segmentation", "data", "Medetec_foot_ulcer_224")
    for split in ("train", "test"):
        img_dir = os.path.join(med, split, "images")
        lab_dir = os.path.join(med, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith((".png", ".jpg")):
                continue
            pairs.append((
                os.path.join(img_dir, name),
                os.path.join(lab_dir, name),
                f"medetec:{os.path.splitext(name)[0]}",
                "medetec_foot_ulcer_224",
            ))
    wseg_s = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "sample")
    wseg_m = os.path.join(ROOT, "data", "datasets", "external", "hf_wseg_dataset", "extracted", "wseg_dataset", "mask")
    if os.path.isdir(wseg_s):
        names = sorted(f for f in os.listdir(wseg_s) if f.lower().endswith((".jpg", ".png")))
        stride = max(1, len(names) // WSEG_CAP)
        for name in names[::stride][:WSEG_CAP]:
            pairs.append((
                os.path.join(wseg_s, name),
                os.path.join(wseg_m, name),
                f"wseg:{os.path.splitext(name)[0]}",
                "hf_wseg_dataset",
            ))

    for ip, mp, subject, source in pairs:
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            exclusions.append({"reason": "unreadable", "source": ip})
            continue
        if img.shape[:2] != mask.shape[:2]:
            exclusions.append({"reason": "dim", "source": ip})
            continue
        digest = _pixel_sha(img)
        if digest in seen_hash:
            exclusions.append({"reason": "exact_dup", "source": ip})
            continue
        line = _mask_to_yolo_line(mask, class_id=2)
        if line is None:
            exclusions.append({"reason": "no_valid_box", "source": ip})
            continue
        seen_hash.add(digest)
        stem = os.path.splitext(os.path.basename(ip))[0]
        records.append({
            "sample_id": f"wound_{source}_{stem}",
            "subject_id": subject,
            "class_ids": [2],
            "lines": [line],
            "image_bgr": img,
            "provenance": "PUBLIC_REAL_PHOTOS",
            "source_dataset": source,
            "source_path": _rel(ip),
            "pixel_sha256": digest,
            "empty": False,
        })


def _subject_split(records: list) -> None:
    rng = np.random.default_rng(SEED)
    # Split wound subjects separately from drawing stems
    wound = [r for r in records if 2 in r["class_ids"]]
    other = [r for r in records if 2 not in r["class_ids"]]

    subjects = sorted({r["subject_id"] for r in wound})
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test : n_test + n_val])
    for r in wound:
        if r["subject_id"] in test_s:
            r["split"] = "test"
        elif r["subject_id"] in val_s:
            r["split"] = "val"
        else:
            r["split"] = "train"

    other.sort(key=lambda r: r["pixel_sha256"])
    idx = np.arange(len(other))
    rng.shuffle(idx)
    n = len(other)
    n_test = max(2, int(round(n * 0.15))) if n else 0
    n_val = max(2, int(round(n * 0.15))) if n else 0
    for i, r in enumerate(other):
        rank = int(np.where(idx == i)[0][0]) if n else 0
        if rank < n_test:
            r["split"] = "test"
        elif rank < n_test + n_val:
            r["split"] = "val"
        else:
            r["split"] = "train"


def build() -> dict:
    records: list = []
    exclusions: list = []
    seen: set = set()
    _copy_cut_bruise(records, exclusions)
    for r in records:
        seen.add(r["pixel_sha256"])
    _add_wound_from_masks(records, exclusions, seen)
    _subject_split(records)

    # write YOLO layout
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(ROOT, OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ROOT, OUT_ROOT, "labels", split), exist_ok=True)

    man_rows = []
    box_counts = Counter()
    for rec in records:
        split = rec["split"]
        stem = rec["sample_id"]
        img_out = os.path.join(ROOT, OUT_ROOT, "images", split, f"{stem}.jpg")
        lbl_out = os.path.join(ROOT, OUT_ROOT, "labels", split, f"{stem}.txt")
        cv2.imwrite(img_out, rec["image_bgr"])
        with open(lbl_out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rec["lines"]) + ("\n" if rec["lines"] else ""))
        for cid in rec["class_ids"]:
            box_counts[NAMES[cid]] += sum(1 for ln in rec["lines"] if int(float(ln.split()[0])) == cid)
        man_rows.append({
            "sample_id": stem,
            "split": split,
            "subject_id": rec["subject_id"],
            "dest_image": _rel(img_out),
            "dest_label": _rel(lbl_out),
            "source_dataset": rec["source_dataset"],
            "source_path": rec["source_path"],
            "provenance": rec["provenance"],
            "pixel_sha256": rec["pixel_sha256"],
            "n_boxes": len(rec["lines"]),
            "class_ids": ",".join(str(c) for c in rec["class_ids"]),
        })

    yaml_path = os.path.join(ROOT, OUT_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({
            "path": os.path.abspath(os.path.join(ROOT, OUT_ROOT)).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": NAMES,
            "nc": 3,
        }, handle, sort_keys=False)

    man = os.path.join(ROOT, OUT_ROOT, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(man_rows[0].keys()))
        writer.writeheader()
        writer.writerows(man_rows)

    by_split_subj = defaultdict(set)
    for r in man_rows:
        by_split_subj[r["split"]].add(r["subject_id"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": OUT_ROOT,
        "yaml": _rel(yaml_path),
        "n": len(man_rows),
        "split_sizes": dict(Counter(r["split"] for r in man_rows)),
        "box_counts": dict(box_counts),
        "provenance_counts": dict(Counter(r["provenance"] for r in man_rows)),
        "subjects": len({r["subject_id"] for r in man_rows}),
        "subject_overlap": {
            "train_val": len(by_split_subj["train"] & by_split_subj["val"]),
            "train_test": len(by_split_subj["train"] & by_split_subj["test"]),
            "val_test": len(by_split_subj["val"] & by_split_subj["test"]),
        },
        "exclusions_n": len(exclusions),
        "names": NAMES,
        "known_limitations": [
            "wound boxes are axis-aligned envelopes of public ulcer/wound masks — not sports injuries.",
            "cut/bruise remain largely SYNTHETIC drawings from yolo_retrain_v2.",
            "Roboflow cut/bruise photo download blocked (no valid API key).",
            "Do not claim clinical detection performance.",
        ],
    }
    summary["leakage_free"] = all(v == 0 for v in summary["subject_overlap"].values())
    with open(os.path.join(ROOT, OUT_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:40]}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
