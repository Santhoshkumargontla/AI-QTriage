"""Independent forensic audit of the live YOLO checkpoint. Does not retrain."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ml.models.canonical_paths import (  # noqa: E402
    ROOT as CANON_ROOT,
    YOLO_BACKUP_PATHS,
    YOLO_CANONICAL,
    YOLO_SYNTHETIC_BASELINE,
    abs_path,
    exists,
    posix,
    sha256_file,
)

NAMES = {0: "cut", 1: "bruise", 2: "wound"}
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EDGE_PX = 8
TINY_AREA_FRAC = 0.01
OUT_PATH = os.path.join(ROOT, "scratch", "forensic_yolo_audit.json")


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            out.append(os.path.join(folder, name))
    return out


def _dhash(bgr: np.ndarray, size: int = 8) -> str:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return f"{int(bits, 2):016x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _read_labels(path: str) -> list[tuple[int, float, float, float, float]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                continue
            rows.append((cid, xc, yc, w, h))
    return rows


def _xywhn_to_xyxy(xc, yc, w, h, iw, ih):
    x1 = (xc - w / 2.0) * iw
    y1 = (yc - h / 2.0) * ih
    x2 = (xc + w / 2.0) * iw
    y2 = (yc + h / 2.0) * ih
    return [x1, y1, x2, y2]


def _box_flags(xyxy, iw, ih) -> dict:
    x1, y1, x2, y2 = xyxy
    touches = {
        "left": x1 <= EDGE_PX,
        "top": y1 <= EDGE_PX,
        "right": x2 >= iw - EDGE_PX,
        "bottom": y2 >= ih - EDGE_PX,
    }
    n_edges = sum(touches.values())
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return {
        "touches_border": n_edges >= 1,
        "corner_like": n_edges >= 2,
        "n_edges": n_edges,
        "area_frac": float(area / max(iw * ih, 1)),
        "tiny": (area / max(iw * ih, 1)) < TINY_AREA_FRAC,
        **touches,
    }


def audit_split(root: str, split: str) -> dict:
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    images = _list_images(img_dir)
    box_counts = Counter()
    img_class = Counter()
    hashes = {}
    dhashes = {}
    stems = []
    annot = {
        "missing_label_file": 0,
        "empty_label": 0,
        "invalid_class_id": 0,
        "out_of_range_coords": 0,
        "gt_border_boxes": 0,
        "gt_corner_boxes": 0,
        "gt_tiny_boxes": 0,
        "boxes_total": 0,
        "multi_box_images": 0,
        "max_boxes_on_image": 0,
        "examples_corner_gt": [],
        "examples_tiny_gt": [],
    }
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        stems.append(stem)
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        ih, iw = bgr.shape[:2]
        digest = hashlib.sha256(bgr.tobytes() + f"|{iw}x{ih}".encode()).hexdigest()
        hashes[digest] = hashes.get(digest, []) + [posix(os.path.relpath(img_path, ROOT))]
        dhashes[stem] = (_dhash(bgr), split, posix(os.path.relpath(img_path, ROOT)))
        lf = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.isfile(lf):
            annot["missing_label_file"] += 1
            continue
        rows = _read_labels(lf)
        if not rows:
            annot["empty_label"] += 1
        present = set()
        annot["boxes_total"] += len(rows)
        annot["max_boxes_on_image"] = max(annot["max_boxes_on_image"], len(rows))
        if len(rows) > 1:
            annot["multi_box_images"] += 1
        for cid, xc, yc, w, h in rows:
            name = NAMES.get(cid, f"id_{cid}")
            if cid not in NAMES:
                annot["invalid_class_id"] += 1
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.5 and 0.0 < h <= 1.5):
                annot["out_of_range_coords"] += 1
            box_counts[name] += 1
            present.add(name)
            xyxy = _xywhn_to_xyxy(xc, yc, w, h, iw, ih)
            flags = _box_flags(xyxy, iw, ih)
            if flags["touches_border"]:
                annot["gt_border_boxes"] += 1
            if flags["corner_like"]:
                annot["gt_corner_boxes"] += 1
                if len(annot["examples_corner_gt"]) < 8:
                    annot["examples_corner_gt"].append({
                        "image": posix(os.path.relpath(img_path, ROOT)),
                        "class": name,
                        "xyxy": [round(v, 2) for v in xyxy],
                        "wh": [iw, ih],
                    })
            if flags["tiny"]:
                annot["gt_tiny_boxes"] += 1
                if len(annot["examples_tiny_gt"]) < 8:
                    annot["examples_tiny_gt"].append({
                        "image": posix(os.path.relpath(img_path, ROOT)),
                        "class": name,
                        "area_frac": round(flags["area_frac"], 5),
                    })
        for name in present:
            img_class[name] += 1
    exact_dups = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    return {
        "images": len(images),
        "boxes": dict(box_counts),
        "images_per_class": dict(img_class),
        "exact_pixel_duplicate_groups": exact_dups,
        "annotation": annot,
        "stems": stems,
        "dhashes": dhashes,
        "pixel_hashes": hashes,
    }


def leakage_between(splits: dict) -> dict:
    pixel_to_splits = defaultdict(set)
    stem_to_splits = defaultdict(set)
    for split, data in splits.items():
        for digest, paths in data["pixel_hashes"].items():
            pixel_to_splits[digest].add(split)
        for stem in data["stems"]:
            stem_to_splits[stem].add(split)
    pixel_leak = {h: sorted(s) for h, s in pixel_to_splits.items() if len(s) > 1}
    stem_leak = {st: sorted(s) for st, s in stem_to_splits.items() if len(s) > 1}

    items = []
    for split, data in splits.items():
        for stem, (dh, sp, rel) in data["dhashes"].items():
            items.append((dh, sp, rel, stem))
    near = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[1] == b[1]:
                continue
            dist = _hamming(a[0], b[0])
            if dist <= 6:
                near.append({"a": a[2], "a_split": a[1], "b": b[2], "b_split": b[1], "hamming": dist})
    return {
        "pixel_hash_cross_split": pixel_leak,
        "stem_cross_split": stem_leak,
        "near_duplicate_cross_split_hamming_le_6": near[:50],
        "near_duplicate_cross_split_count": len(near),
        "subject_id_field_present": False,
        "subject_split_status": "NOT_AVAILABLE — filenames have no subject_id",
    }


def infer_image(model, path: str, conf: float) -> dict:
    bgr = cv2.imread(path)
    if bgr is None:
        return {"image": posix(path), "error": "unreadable"}
    ih, iw = bgr.shape[:2]
    results = model(path, conf=conf, verbose=False)
    dets = []
    if results:
        boxes = results[0].boxes
        orig = getattr(results[0], "orig_shape", (ih, iw))
        if orig is not None and len(orig) == 2:
            ih, iw = int(orig[0]), int(orig[1])
        names = results[0].names if hasattr(results[0], "names") else NAMES
        if boxes is not None:
            for box in boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                cid = int(box.cls[0])
                score = float(box.conf[0])
                if isinstance(names, dict):
                    cname = str(names.get(cid, NAMES.get(cid, cid)))
                else:
                    cname = str(names[cid]) if cid < len(names) else str(cid)
                flags = _box_flags(xyxy, iw, ih)
                dets.append({
                    "class": str(cname).lower(),
                    "confidence": round(score, 4),
                    "xyxy": [round(v, 2) for v in xyxy],
                    "touches_border": flags["touches_border"],
                    "corner_like": flags["corner_like"],
                    "tiny": flags["tiny"],
                    "area_frac": round(flags["area_frac"], 5),
                })
    return {
        "image": posix(os.path.relpath(path, ROOT)) if path.startswith(ROOT) else posix(path),
        "width": iw,
        "height": ih,
        "threshold": conf,
        "n_detections": len(dets),
        "detections": dets,
        "n_border": sum(1 for d in dets if d["touches_border"]),
        "n_corner": sum(1 for d in dets if d["corner_like"]),
    }


def summarize_sweep(rows: list[dict]) -> dict:
    out = {}
    for conf in THRESHOLDS:
        subset = [r for r in rows if r.get("threshold") == conf and "error" not in r]
        n_det = sum(r["n_detections"] for r in subset)
        out[str(conf)] = {
            "images": len(subset),
            "detections": n_det,
            "mean_dets_per_image": round(n_det / max(len(subset), 1), 3),
            "border_boxes": sum(r["n_border"] for r in subset),
            "corner_boxes": sum(r["n_corner"] for r in subset),
            "class_counts": dict(Counter(d["class"] for r in subset for d in r["detections"])),
            "max_conf": max((d["confidence"] for r in subset for d in r["detections"]), default=None),
            "min_conf": min((d["confidence"] for r in subset for d in r["detections"]), default=None),
        }
    return out


def ultralytics_val(model, yaml_path: str, split: str) -> dict:
    res = model.val(data=yaml_path, split=split, verbose=False, plots=False)
    box = res.box
    names = dict(getattr(res, "names", None) or model.names or NAMES)
    per_class = {}
    maps = getattr(box, "maps", None)
    p = getattr(box, "p", None)
    r = getattr(box, "r", None)
    ap50 = getattr(box, "ap50", None)
    maps = list(maps) if maps is not None else []
    p = list(p) if p is not None else []
    r = list(r) if r is not None else []
    ap50 = list(ap50) if ap50 is not None else []
    for idx, name in names.items():
        i = int(idx)
        entry = {"name": name}
        if i < len(p):
            entry["precision"] = float(p[i])
        if i < len(r):
            entry["recall"] = float(r[i])
        if i < len(ap50):
            entry["mAP50"] = float(ap50[i])
        if i < len(maps):
            entry["mAP50-95"] = float(maps[i])
        per_class[name] = entry
    return {
        "split": split,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": per_class,
        "nc": int(getattr(box, "nc", len(names))),
    }


def main() -> None:
    from ultralytics import YOLO

    ckpt = abs_path(YOLO_CANONICAL)
    model = YOLO(ckpt)
    names = dict(model.names) if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    live_sha = sha256_file(ckpt)

    dup_rows = []
    for rel in (YOLO_CANONICAL, YOLO_CANONICAL + ".pre_retrain_backup", YOLO_SYNTHETIC_BASELINE, *YOLO_BACKUP_PATHS):
        path = abs_path(rel)
        if os.path.isfile(path):
            digest = sha256_file(path)
            dup_rows.append({
                "path": posix(rel),
                "sha256": digest,
                "same_as_canonical": digest == live_sha,
                "size": os.path.getsize(path),
            })

    datasets = {}
    leakage = {}
    for key, root in (
        ("yolo_merged", abs_path("data/datasets/yolo_merged")),
        ("yolo_processed", abs_path("data/datasets/yolo_processed")),
    ):
        splits = {}
        for split in ("train", "val", "test"):
            splits[split] = audit_split(root, split)
        leakage[key] = leakage_between(splits)
        datasets[key] = {
            split: {
                "images": splits[split]["images"],
                "boxes": splits[split]["boxes"],
                "images_per_class": splits[split]["images_per_class"],
                "exact_pixel_duplicate_groups": splits[split]["exact_pixel_duplicate_groups"],
                "annotation": {k: v for k, v in splits[split]["annotation"].items() if k != "examples_corner_gt" and k != "examples_tiny_gt"}
                | {
                    "examples_corner_gt": splits[split]["annotation"]["examples_corner_gt"],
                    "examples_tiny_gt": splits[split]["annotation"]["examples_tiny_gt"],
                },
            }
            for split in splits
        }
        # keep dhash structures only for leakage
        _splits_keep = splits  # noqa: F841

    # Independent ultralytics.val on the training distribution (yolo_merged) and processed set.
    val_metrics = {}
    merged_yaml = abs_path("data/datasets/yolo_merged/yolo_merged.yaml")
    processed_yaml = abs_path("data/datasets/yolo_processed/data.yaml")
    for label, yaml_path, split in (
        ("yolo_merged_val", merged_yaml, "val"),
        ("yolo_merged_test", merged_yaml, "test"),
        ("yolo_processed_val", processed_yaml, "val"),
        ("yolo_processed_test", processed_yaml, "test"),
    ):
        if os.path.isfile(yaml_path):
            val_metrics[label] = ultralytics_val(model, yaml_path, split)

    extra = []
    for rel in (
        "data/sample/image/football_injury.jpg",
        "data/datasets/yolo_injury/blank_skin.jpg",
        "data/datasets/yolo_injury/dummy_test.jpg",
        "data/datasets/yolo_injury/uniform_skin.jpg",
    ):
        p = abs_path(rel)
        if os.path.isfile(p):
            extra.append(p)

    # Unrelated: a non-injury image if present; otherwise a generated gray is NOT used as "found".
    unrelated = []
    for rel in (
        "frontend/public/next.svg",
        "data/sample/image/unrelated.jpg",
    ):
        p = abs_path(rel)
        if os.path.isfile(p):
            unrelated.append(p)

    merged_val_imgs = _list_images(abs_path("data/datasets/yolo_merged/images/val"))
    merged_test_imgs = _list_images(abs_path("data/datasets/yolo_merged/images/test"))
    proc_val_imgs = _list_images(abs_path("data/datasets/yolo_processed/images/val"))
    proc_test_imgs = _list_images(abs_path("data/datasets/yolo_processed/images/test"))

    sweep = {"demo": [], "merged_val": [], "merged_test": [], "processed_val": [], "processed_test": [], "negatives": [], "unrelated": []}
    groups = [
        ("demo", extra[:1]),
        ("negatives", extra[1:]),
        ("unrelated", unrelated),
        ("merged_val", merged_val_imgs),
        ("merged_test", merged_test_imgs),
        ("processed_val", proc_val_imgs),
        ("processed_test", proc_test_imgs),
    ]
    for conf in THRESHOLDS:
        for group, paths in groups:
            for path in paths:
                sweep[group].append(infer_image(model, path, conf))

    # Demo detail at every threshold
    demo_path = abs_path("data/sample/image/football_injury.jpg")
    demo_dims = None
    if os.path.isfile(demo_path):
        bgr = cv2.imread(demo_path)
        if bgr is not None:
            demo_dims = {"width": int(bgr.shape[1]), "height": int(bgr.shape[0])}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_retrain": True,
        "checkpoint": {
            "canonical_path": posix(YOLO_CANONICAL),
            "abs_path": posix(ckpt),
            "sha256": live_sha,
            "task": model.task,
            "names": names,
            "class_list": [str(names[k]).lower() for k in sorted(names, key=lambda x: int(x))],
            "unsupported_not_in_model_names": [c for c in ("abrasion", "laceration", "swelling") if c not in {str(v).lower() for v in names.values()}],
            "file_size": os.path.getsize(ckpt),
        },
        "duplicate_artifacts": dup_rows,
        "training_set_for_this_checkpoint": "yolo_merged (metadata + yolo_baseline_vs_candidate.json). yolo_processed candidate SHA 5afe49cb was not promoted.",
        "datasets": datasets,
        "leakage": leakage,
        "independent_ultralytics_val": val_metrics,
        "threshold_sweep_summary": {g: summarize_sweep(rows) for g, rows in sweep.items()},
        "threshold_sweep_demo": [r for r in sweep["demo"]],
        "threshold_sweep_negatives": sweep["negatives"],
        "demo_image": {
            "path": "data/sample/image/football_injury.jpg",
            "exists": os.path.isfile(demo_path),
            "dims": demo_dims,
            "note": "Synthetic demonstration graphic, not a clinical photograph.",
        },
        "corner_box_examples_at_0.25": {
            group: [
                {
                    "image": r["image"],
                    "wh": [r.get("width"), r.get("height")],
                    "det": d,
                }
                for r in rows if r.get("threshold") == 0.25
                for d in r.get("detections", [])
                if d.get("corner_like")
            ][:20]
            for group, rows in sweep.items()
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", OUT_PATH)
    print("sha", live_sha)
    print("task", model.task, "names", names)
    print("val metrics keys", list(val_metrics))
    for k, v in val_metrics.items():
        print(k, {x: v[x] for x in ("precision", "recall", "mAP50", "mAP50-95")})
    print("demo 0.25", [r for r in sweep["demo"] if r.get("threshold") == 0.25])


if __name__ == "__main__":
    main()
