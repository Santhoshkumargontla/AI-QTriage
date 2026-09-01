"""Build a clean YOLO training set from existing sources.

Does not modify raw datasets. Does not remap abrasion/laceration/swelling
onto wound. Does not fabricate or oversample images.
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

from ml.models.canonical_paths import YOLO_PROCESSED_ROOT, YOLO_PROCESSED_YAML

ACTIVE = {0: "cut", 1: "bruise", 2: "wound"}
ACTIVE_NAMES = set(ACTIVE.values())
DROPPED_NAMES = {"abrasion", "laceration", "swelling", "burn", "other", "blister"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
PROVENANCE = "SYNTHETIC"


def _rel(path: str) -> str:
    return path.replace("\\", "/")


def _pixel_sha256(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None
    h, w = bgr.shape[:2]
    digest = hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()
    return digest, w, h


def _byte_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xywh_to_xyxy(xc, yc, w, h):
    return xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0


def _xyxy_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h


def _intersect(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def _clip_xyxy(x1, y1, x2, y2):
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _merge_same_class(boxes: list[dict]) -> tuple[list[dict], int]:
    """Union overlapping same-class boxes. Returns (merged, n_removed_by_merge)."""
    by_cls = defaultdict(list)
    for box in boxes:
        by_cls[box["class_id"]].append(box)
    merged = []
    removed = 0
    for cid, group in by_cls.items():
        xyxys = []
        for box in group:
            xyxy = _clip_xyxy(*_xywh_to_xyxy(*box["xywh"]))
            if xyxy is None:
                removed += 1
                continue
            xyxys.append(xyxy)
        parent = list(range(len(xyxys)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(xyxys)):
            for j in range(i + 1, len(xyxys)):
                if _intersect(xyxys[i], xyxys[j]):
                    parent[find(j)] = find(i)
        clusters = defaultdict(list)
        for i, xyxy in enumerate(xyxys):
            clusters[find(i)].append(xyxy)
        for members in clusters.values():
            x1 = min(m[0] for m in members)
            y1 = min(m[1] for m in members)
            x2 = max(m[2] for m in members)
            y2 = max(m[3] for m in members)
            clipped = _clip_xyxy(x1, y1, x2, y2)
            if clipped is None:
                removed += len(members)
                continue
            removed += len(members) - 1
            xc, yc, w, h = _xyxy_to_xywh(*clipped)
            merged.append({
                "class_id": cid,
                "class_name": ACTIVE[cid],
                "xywh": (xc, yc, w, h),
            })
    return merged, removed


def _parse_label_lines(path: str, source_names: dict[int, str], raw_class_override: str | None):
    """Keep boxes whose SOURCE class is cut/bruise/wound. Never remap dropped classes."""
    kept = []
    dropped = []
    invalid = 0
    if not os.path.exists(path):
        return kept, dropped, invalid
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = [ln.strip() for ln in handle if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            invalid += 1
            continue
        if w <= 1e-6 or h <= 1e-6:
            invalid += 1
            continue
        src_name = (raw_class_override or source_names.get(cid) or "").lower().strip()
        if src_name in DROPPED_NAMES or src_name not in ACTIVE_NAMES:
            dropped.append(src_name or f"id_{cid}")
            continue
        dest_id = {v: k for k, v in ACTIVE.items()}[src_name]
        kept.append({"class_id": dest_id, "class_name": src_name, "xywh": (xc, yc, w, h)})
    return kept, dropped, invalid


def _load_real_wound_raw_class():
    path = os.path.join("data", "datasets", "yolo_real_wound", "manifest.csv")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = (row.get("sample_id") or "").strip()
            raw = (row.get("raw_class") or "").strip().lower()
            if stem:
                mapping[stem] = raw
    return mapping


def _iter_source_images(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in {"qc_samples", "masks", "mask"}]
        low = dirpath.replace("\\", "/").lower()
        if "/qc_samples" in low or "/masks" in low or "/mask/" in low:
            continue
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXTS:
                continue
            stem = os.path.splitext(name)[0].lower()
            if stem in {"blank_skin", "dummy_test"} or name.lower().startswith("qc_"):
                continue
            yield os.path.join(dirpath, name)


def _infer_split(path: str) -> str:
    parts = path.replace("\\", "/").lower().split("/")
    if "valid" in parts:
        return "val"
    for split in ("train", "val", "test"):
        if split in parts:
            return split
    return "train"


def _label_for_image(img_path: str, dataset_root: str, split: str, stem: str) -> str | None:
    parent = os.path.dirname(img_path)
    candidates = [
        os.path.join(os.path.dirname(parent), "labels", stem + ".txt"),
        os.path.join(dataset_root, "labels", split, stem + ".txt"),
        os.path.join(parent.replace(os.path.join("images", split), os.path.join("labels", split)), stem + ".txt"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def _write_label(path: str, boxes: list[dict]):
    lines = []
    for box in boxes:
        xc, yc, w, h = box["xywh"]
        lines.append(f"{box['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def prepare_yolo_processed_dataset():
    sources = [
        {
            "id": "raw_synthetic_wound",
            "root": os.path.join("data", "datasets", "raw", "synthetic_wound"),
            "names": {0: "cut", 1: "bruise", 2: "abrasion", 3: "laceration"},
            "use_manifest_raw_class": False,
        },
        {
            "id": "yolo_injury",
            "root": os.path.join("data", "datasets", "yolo_injury"),
            "names": {0: "cut", 1: "bruise", 2: "abrasion", 3: "laceration"},
            "use_manifest_raw_class": False,
        },
        {
            "id": "yolo_real_wound",
            "root": os.path.join("data", "datasets", "yolo_real_wound"),
            "names": {0: "cut", 1: "bruise", 2: "wound"},
            "use_manifest_raw_class": True,
        },
    ]

    real_raw = _load_real_wound_raw_class()
    if os.path.exists(YOLO_PROCESSED_ROOT):
        shutil.rmtree(YOLO_PROCESSED_ROOT)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(YOLO_PROCESSED_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_PROCESSED_ROOT, "labels", split), exist_ok=True)

    seen_pixel = {}
    seen_byte = {}
    seen_stem = {}
    kept_rows = []
    exclusions = []
    merge_removed_total = 0
    dropped_class_counts = Counter()

    for source in sources:
        root = source["root"]
        if not os.path.exists(root):
            continue
        for img_path in sorted(_iter_source_images(root)):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            split = _infer_split(img_path)
            rel_src = _rel(img_path)

            pixel, width, height = _pixel_sha256(img_path)
            if pixel is None:
                exclusions.append({"source_path": rel_src, "reason": "unreadable_image"})
                continue
            byte = _byte_sha256(img_path)
            if pixel in seen_pixel:
                exclusions.append({
                    "source_path": rel_src,
                    "reason": "exact_pixel_duplicate",
                    "kept": seen_pixel[pixel],
                })
                continue
            if byte in seen_byte:
                exclusions.append({
                    "source_path": rel_src,
                    "reason": "exact_byte_duplicate",
                    "kept": seen_byte[byte],
                })
                continue
            dest_stem = f"{source['id']}__{stem}"
            if dest_stem in seen_stem:
                exclusions.append({
                    "source_path": rel_src,
                    "reason": "duplicate_stem",
                    "kept": seen_stem[dest_stem],
                })
                continue

            raw_override = None
            if source["use_manifest_raw_class"]:
                raw_override = real_raw.get(stem)
                if not raw_override:
                    exclusions.append({
                        "source_path": rel_src,
                        "reason": "missing_raw_class_in_manifest",
                    })
                    continue
                if raw_override in DROPPED_NAMES:
                    dropped_class_counts[raw_override] += 1
                    exclusions.append({
                        "source_path": rel_src,
                        "reason": "unsupported_source_class",
                        "dropped_classes": raw_override,
                    })
                    continue
                if raw_override not in ACTIVE_NAMES:
                    exclusions.append({
                        "source_path": rel_src,
                        "reason": "unsupported_source_class",
                        "dropped_classes": raw_override,
                    })
                    continue

            label_path = _label_for_image(img_path, root, split, stem)
            kept_boxes, dropped, invalid = _parse_label_lines(
                label_path or "",
                source["names"],
                raw_override,
            )
            for name in dropped:
                dropped_class_counts[name] += 1
            if invalid:
                exclusions.append({
                    "source_path": rel_src,
                    "reason": "invalid_or_zero_area_boxes_skipped",
                    "invalid_count": invalid,
                })
            if not kept_boxes:
                exclusions.append({
                    "source_path": rel_src,
                    "reason": "no_supported_class_boxes",
                    "dropped_classes": ",".join(sorted(set(dropped))) if dropped else (raw_override or ""),
                })
                continue

            merged, n_removed = _merge_same_class(kept_boxes)
            merge_removed_total += n_removed
            if not merged:
                exclusions.append({"source_path": rel_src, "reason": "boxes_collapsed_to_empty_after_merge"})
                continue

            ext = os.path.splitext(img_path)[1].lower()
            dest_img = os.path.join(YOLO_PROCESSED_ROOT, "images", split, dest_stem + ext)
            dest_lbl = os.path.join(YOLO_PROCESSED_ROOT, "labels", split, dest_stem + ".txt")
            shutil.copy2(img_path, dest_img)
            _write_label(dest_lbl, merged)

            seen_pixel[pixel] = dest_stem
            seen_byte[byte] = dest_stem
            seen_stem[dest_stem] = dest_stem
            kept_rows.append({
                "sample_id": dest_stem,
                "split": split,
                "dest_image": _rel(dest_img),
                "dest_label": _rel(dest_lbl),
                "source_dataset": source["id"],
                "source_path": rel_src,
                "source_label": _rel(label_path) if label_path else "",
                "source_stem": stem,
                "source_split": split,
                "provenance": PROVENANCE,
                "pixel_sha256": pixel,
                "byte_sha256": byte,
                "width": width,
                "height": height,
                "kept_classes": ",".join(sorted({b["class_name"] for b in merged})),
                "dropped_classes": ",".join(sorted(set(dropped))),
                "boxes_in": len(kept_boxes),
                "boxes_out": len(merged),
                "boxes_merged_away": n_removed,
            })

    # Disjoint split check (by pixel hash and dest stem)
    split_pixels = defaultdict(set)
    split_stems = defaultdict(set)
    for row in kept_rows:
        split_pixels[row["split"]].add(row["pixel_sha256"])
        split_stems[row["split"]].add(row["sample_id"])
    overlap_pixel = []
    overlap_stem = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pix = split_pixels[a] & split_pixels[b]
        st = split_stems[a] & split_stems[b]
        if pix:
            overlap_pixel.append((a, b, len(pix)))
        if st:
            overlap_stem.append((a, b, len(st)))
    if overlap_pixel or overlap_stem:
        raise RuntimeError(f"Split overlap remains: pixels={overlap_pixel} stems={overlap_stem}")

    boxes_by_split = defaultdict(Counter)
    images_by_split = defaultdict(Counter)
    for row in kept_rows:
        lbl = row["dest_label"]
        present = set()
        with open(lbl, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                name = ACTIVE[int(float(parts[0]))]
                boxes_by_split[row["split"]][name] += 1
                present.add(name)
        for name in present:
            images_by_split[row["split"]][name] += 1

    split_counts = Counter(r["split"] for r in kept_rows)
    box_totals = Counter()
    img_totals = Counter()
    for split in ("train", "val", "test"):
        box_totals.update(boxes_by_split[split])
        img_totals.update(images_by_split[split])

    ratio = None
    nonzero = [v for v in box_totals.values() if v]
    if len(box_totals) == 3 and all(box_totals.get(n, 0) > 0 for n in ACTIVE_NAMES):
        ratio = round(max(box_totals.values()) / min(box_totals.values()), 3)
    missing_classes = [n for n in ACTIVE.values() if box_totals.get(n, 0) == 0]

    yaml_body = (
        f"path: {os.path.abspath(YOLO_PROCESSED_ROOT)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: cut\n"
        "  1: bruise\n"
        "  2: wound\n"
        f"provenance: {PROVENANCE}\n"
        "notes: |\n"
        "  Honest mapping only. cut=0, bruise=1, wound=2.\n"
        "  abrasion, laceration, and swelling were dropped, not remapped.\n"
        "  Raw sources were not modified. Exact pixel/byte duplicates were removed.\n"
        "  Do not treat this set as real clinical photography.\n"
    )
    with open(YOLO_PROCESSED_YAML, "w", encoding="utf-8") as handle:
        handle.write(yaml_body)

    manifest_path = os.path.join(YOLO_PROCESSED_ROOT, "manifest.csv")
    fieldnames = list(kept_rows[0].keys()) if kept_rows else [
        "sample_id", "split", "dest_image", "source_dataset", "provenance"
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    excl_path = os.path.join(YOLO_PROCESSED_ROOT, "exclusions.csv")
    excl_fields = sorted({k for row in exclusions for k in row})
    with open(excl_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=excl_fields)
        writer.writeheader()
        writer.writerows(exclusions)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_train": True,
        "did_not_modify_raw_sources": True,
        "did_not_fabricate_images": True,
        "did_not_oversample_minority_classes": True,
        "provenance": PROVENANCE,
        "class_mapping": {
            "0": "cut",
            "1": "bruise",
            "2": "wound",
            "dropped_not_remapped": sorted(DROPPED_NAMES),
            "silent_remap": False,
        },
        "final_unique_image_count": len(kept_rows),
        "train_val_test_counts": dict(split_counts),
        "boxes_per_class": dict(box_totals),
        "images_per_class": dict(img_totals),
        "boxes_per_class_by_split": {k: dict(v) for k, v in boxes_by_split.items()},
        "images_per_class_by_split": {k: dict(v) for k, v in images_by_split.items()},
        "duplicate_removal": {
            "exact_pixel_duplicate": sum(1 for e in exclusions if e["reason"] == "exact_pixel_duplicate"),
            "exact_byte_duplicate": sum(1 for e in exclusions if e["reason"] == "exact_byte_duplicate"),
            "duplicate_stem": sum(1 for e in exclusions if e["reason"] == "duplicate_stem"),
            "overlapping_same_class_boxes_merged_away": merge_removed_total,
        },
        "exclusions": dict(Counter(e["reason"] for e in exclusions)),
        "dropped_unsupported_class_box_or_image_counts": dict(dropped_class_counts),
        "split_overlap_pixel": overlap_pixel,
        "split_overlap_stem": overlap_stem,
        "remaining_imbalance": {
            "box_ratio_max_over_min": ratio,
            "classes_with_zero_boxes": missing_classes,
            "note": (
                "Minority classes were not duplicated. wound is empty if no source "
                "image was originally labeled wound (swelling/abrasion/laceration were dropped)."
            ),
        },
        "paths": {
            "root": _rel(YOLO_PROCESSED_ROOT),
            "yaml": _rel(YOLO_PROCESSED_YAML),
            "manifest": _rel(manifest_path),
            "exclusions": _rel(excl_path),
        },
    }
    summary_path = os.path.join(YOLO_PROCESSED_ROOT, "processing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    card = [
        "# yolo_processed",
        "",
        f"Provenance: **{PROVENANCE}**",
        "Active classes: `0 cut`, `1 bruise`, `2 wound`.",
        "abrasion, laceration, and swelling were dropped. They were not remapped.",
        "Raw source datasets were left unchanged.",
        "",
        f"- Unique images: {len(kept_rows)}",
        f"- Splits: {dict(split_counts)}",
        f"- Boxes per class: {dict(box_totals)}",
        f"- Images per class: {dict(img_totals)}",
        f"- Zero-box classes: {missing_classes}",
        "",
        "See `manifest.csv` and `processing_summary.json`.",
        "Do not train until remaining imbalance / missing wound is accepted or new honest labels exist.",
        "",
    ]
    with open(os.path.join(YOLO_PROCESSED_ROOT, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(card))

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    prepare_yolo_processed_dataset()
