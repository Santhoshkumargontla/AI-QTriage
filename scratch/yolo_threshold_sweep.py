"""YOLO threshold sweep on representative images. Evidence only — no fabricated detections."""
import os
import sys
import json
import hashlib
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from ml.models.canonical_paths import YOLO_CANONICAL

THRESHOLDS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
OUT = os.path.join("scratch", "yolo_threshold_sweep.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images():
    images = []
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        images.append(("demo_football", demo))
    val_dir = os.path.join("data", "datasets", "yolo_real_wound", "images", "val")
    if os.path.isdir(val_dir):
        for name in sorted(os.listdir(val_dir))[:5]:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append((f"val_{name}", os.path.join(val_dir, name)))
    train_dir = os.path.join("data", "datasets", "yolo_real_wound", "images", "train")
    if os.path.isdir(train_dir):
        for name in sorted(os.listdir(train_dir))[:3]:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append((f"train_{name}", os.path.join(train_dir, name)))
    return images


def main():
    os.makedirs("scratch", exist_ok=True)
    model = YOLO(YOLO_CANONICAL)
    rows = []
    for label, path in collect_images():
        for thr in THRESHOLDS:
            result = model(path, conf=thr, verbose=False)[0]
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    cid = int(box.cls[0].item())
                    boxes.append({
                        "class": str(model.names.get(cid, cid)),
                        "confidence": round(float(box.conf[0].item()), 4),
                        "bounding_box": [round(float(v), 2) for v in box.xyxy[0].cpu().numpy().tolist()],
                    })
            rows.append({
                "image": label,
                "path": path,
                "threshold": thr,
                "number_of_boxes": len(boxes),
                "detections": boxes,
            })
    report = {
        "checkpoint": YOLO_CANONICAL,
        "sha256": _sha256(YOLO_CANONICAL),
        "task": model.task,
        "names": dict(model.names),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT} rows={len(rows)}")
    for row in rows:
        if row["threshold"] in (0.10, 0.25, 0.40) and ("demo" in row["image"] or row["number_of_boxes"] > 0):
            print(f"{row['image']} thr={row['threshold']} n={row['number_of_boxes']}")


if __name__ == "__main__":
    main()
