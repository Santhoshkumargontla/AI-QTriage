"""Prepare fracture X-ray YOLO detect set from pkdarabi (CC BY 4.0).

Converts YOLO-seg polygons → axis-aligned xywh boxes.
This is a SEPARATE modality from skin injury YOLO — never merge into skin canonical.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.models.canonical_paths import ROOT

SRC = os.path.join(
    "data", "raw", "kaggle", "pkdarabi_bone_fracture_yolo", "BoneFractureYolo8"
)
OUT_ROOT = os.path.join("data", "datasets", "yolo_fracture_xray_v1")


def _poly_to_xywh(parts: list[str]) -> str | None:
    cid = int(float(parts[0]))
    nums = [float(x) for x in parts[1:]]
    if len(nums) == 4:
        # already xywh
        xc, yc, w, h = nums
        if w <= 0 or h <= 0:
            return None
        return f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
    if len(nums) < 6 or len(nums) % 2 != 0:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    # clip
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    return f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def prepare() -> dict:
    src = os.path.join(ROOT, SRC)
    yaml_src = os.path.join(src, "data.yaml")
    with open(yaml_src, encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)
    names = meta.get("names") or []

    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    box_counts = Counter()
    split_counts = {}
    for split_src, split_dst in (("train", "train"), ("valid", "val"), ("test", "test")):
        img_src = os.path.join(src, split_src, "images")
        lab_src = os.path.join(src, split_src, "labels")
        img_dst = os.path.join(OUT_ROOT, "images", split_dst)
        lab_dst = os.path.join(OUT_ROOT, "labels", split_dst)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lab_dst, exist_ok=True)
        n = 0
        if not os.path.isdir(img_src):
            split_counts[split_dst] = 0
            continue
        for name in os.listdir(img_src):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            stem = os.path.splitext(name)[0]
            shutil.copy2(os.path.join(img_src, name), os.path.join(img_dst, name))
            lines_out = []
            lab_path = os.path.join(lab_src, stem + ".txt")
            if os.path.isfile(lab_path):
                for ln in open(lab_path, encoding="utf-8"):
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    converted = _poly_to_xywh(parts)
                    if converted:
                        lines_out.append(converted)
                        box_counts[int(float(parts[0]))] += 1
            with open(os.path.join(lab_dst, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines_out) + ("\n" if lines_out else ""))
            n += 1
        split_counts[split_dst] = n

    data_yaml = os.path.join(OUT_ROOT, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": os.path.abspath(OUT_ROOT).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(names),
                "names": names,
            },
            handle,
            sort_keys=False,
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": os.path.relpath(OUT_ROOT, ROOT).replace("\\", "/"),
        "yaml": os.path.relpath(data_yaml, ROOT).replace("\\", "/"),
        "names": names,
        "split_image_counts": split_counts,
        "box_counts_by_class_id": {str(k): v for k, v in sorted(box_counts.items())},
        "license": "CC BY 4.0 (Roboflow / pkdarabi Bone Fracture Detection)",
        "modality": "XRAY_NOT_SKIN_PHOTO",
        "notes": "Polygons converted to AABB. Do not promote into skin YOLO canonical.",
    }
    with open(os.path.join(OUT_ROOT, "PREPARE_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    prepare()
