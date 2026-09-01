import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.vision.yolo_wrapper import YOLO11Detector


THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50, 0.75]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    info.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "mime_type": mimetypes.guess_type(str(path))[0],
        }
    )
    return info


def image_info(path: Path) -> dict:
    info = file_info(path)
    with Image.open(path) as image:
        info.update(
            {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "is_corrupted": False,
            }
        )
    return info


def detection_dict(box, names: dict) -> dict:
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
    return {
        "class_id": cls_id,
        "class_name": str(names.get(cls_id, cls_id)),
        "confidence": conf,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def draw_detections(image_path: Path, detections: list, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    for det in detections:
        x1, y1, x2, y2 = [int(round(det[k])) for k in ("x1", "y1", "x2", "y2")]
        label = f"{det['class_name']} {det['confidence']:.3f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(output_path), image)


def inspect_model(detector: YOLO11Detector) -> dict:
    model = detector.model
    stride = getattr(model, "stride", None)
    if hasattr(stride, "tolist"):
        stride = stride.tolist()
    pt_model = getattr(model, "model", None)
    yaml_data = getattr(pt_model, "yaml", {}) if pt_model is not None else {}
    return {
        "wrapper_info": detector.get_info(),
        "model_path": detector.model_path,
        "model_file": file_info(Path(detector.model_path)),
        "task": getattr(model, "task", None),
        "names": {str(k): v for k, v in getattr(model, "names", {}).items()},
        "num_classes": len(getattr(model, "names", {})),
        "stride": stride,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "runtime_device": "cuda" if torch.cuda.is_available() else "cpu",
        "architecture": yaml_data.get("model") or yaml_data.get("yaml_file") or pt_model.__class__.__name__ if pt_model is not None else None,
        "yaml_nc": yaml_data.get("nc"),
    }


def threshold_sweep(model: YOLO, image_path: Path, out_dir: Path) -> list:
    rows = []
    for threshold in THRESHOLDS:
        results = model(str(image_path), conf=threshold, verbose=False)
        boxes = results[0].boxes if results else []
        detections = [detection_dict(box, model.names) for box in boxes]
        out_image = out_dir / f"yolo_conf_{int(threshold * 1000):03d}.jpg"
        draw_detections(image_path, detections, out_image)
        rows.append(
            {
                "confidence_threshold": threshold,
                "iou_threshold": None,
                "raw_detection_count": "NOT_EXPOSED_BY_ULTRALYTICS_RESULTS_API",
                "boxes_after_nms": len(detections),
                "max_confidence": max([d["confidence"] for d in detections], default=None),
                "class_ids": [d["class_id"] for d in detections],
                "class_names": [d["class_name"] for d in detections],
                "detections": detections,
                "annotated_output": str(out_image),
            }
        )
    return rows


def wrapper_stage(detector: YOLO11Detector, image_path: Path) -> dict:
    detections = detector.detect(str(image_path))
    return {
        "wrapper_conf_threshold_field": detector.conf_threshold,
        "wrapper_raw_call_conf": 0.10,
        "detections_after_supported_class_filter": len(detections),
        "detections": detections,
    }


def dataset_audit(dataset_yaml: Path) -> dict:
    data = yaml.safe_load(dataset_yaml.read_text())
    root = Path(data["path"])
    names = {int(k): v for k, v in data["names"].items()}
    report = {"yaml_path": str(dataset_yaml), "root": str(root), "names": names, "splits": {}, "errors": []}
    total_images = total_labels = empty_labels = invalid_labels = corrupt_images = 0
    class_counts = Counter()
    image_hashes = defaultdict(list)

    for split in ("train", "val", "test"):
        image_dir = root / data[split]
        label_dir = root / data[split].replace("images", "labels", 1)
        images = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        labels = sorted(label_dir.glob("*.txt"))
        split_invalid = 0
        split_empty = 0
        split_corrupt = 0
        split_class_counts = Counter()
        missing_labels = 0

        for image_path in images:
            image_hashes[sha256_file(image_path)].append(str(image_path))
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                split_corrupt += 1
                corrupt_images += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
                continue
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                split_empty += 1
                empty_labels += 1
            for line_no, line in enumerate(lines, start=1):
                parts = line.split()
                is_invalid = False
                if len(parts) != 5:
                    is_invalid = True
                else:
                    try:
                        cls_id = int(float(parts[0]))
                        coords = [float(v) for v in parts[1:]]
                        if cls_id not in names:
                            is_invalid = True
                        if any(v < 0 or v > 1 for v in coords):
                            is_invalid = True
                        if coords[2] <= 0 or coords[3] <= 0:
                            is_invalid = True
                        split_class_counts[names.get(cls_id, str(cls_id))] += 1
                        class_counts[names.get(cls_id, str(cls_id))] += 1
                    except Exception:
                        is_invalid = True
                if is_invalid:
                    split_invalid += 1
                    invalid_labels += 1
                    report["errors"].append(f"{label_path}:{line_no}: {line}")

        total_images += len(images)
        total_labels += len(labels)
        report["splits"][split] = {
            "image_count": len(images),
            "label_file_count": len(labels),
            "missing_label_files": missing_labels,
            "empty_label_files": split_empty,
            "invalid_label_rows": split_invalid,
            "corrupt_images": split_corrupt,
            "class_distribution": dict(split_class_counts),
        }

    duplicates = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}
    report["summary"] = {
        "total_images": total_images,
        "total_label_files": total_labels,
        "empty_label_files": empty_labels,
        "invalid_label_rows": invalid_labels,
        "corrupt_images": corrupt_images,
        "class_distribution": dict(class_counts),
        "duplicate_image_hashes": len(duplicates),
        "duplicate_images": duplicates,
    }
    return report


def read_training_results(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {"exists": False}
    import csv

    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    last = rows[-1] if rows else {}
    return {
        "exists": True,
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "last_epoch": last,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--out-dir", default="data/debug/yolo")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_dir / f"original{image_path.suffix.lower()}")

    detector = YOLO11Detector()
    model_report = inspect_model(detector)
    sweep = threshold_sweep(detector.model, image_path, out_dir)
    wrapper = wrapper_stage(detector, image_path)
    audit = dataset_audit(Path("data/datasets/yolo_real_wound/yolo_real.yaml"))
    training = read_training_results(Path("ml/models/yolo_real_training/run_real_wound/results.csv"))

    report = {
        "case_id": args.case_id,
        "image": image_info(image_path),
        "model": model_report,
        "threshold_sweep": sweep,
        "wrapper_stage": wrapper,
        "dataset_audit": audit,
        "training_results": training,
    }
    (out_dir / "yolo_detections.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
