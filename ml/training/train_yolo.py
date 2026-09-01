"""Real YOLO11 training via Ultralytics. Does not copy weights or fabricate metrics."""
import os
import sys
import json
import shutil
import glob
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    YOLO_BASELINE_COPY,
    YOLO_PROCESSED_ROOT,
    YOLO_PROCESSED_YAML,
)

CANDIDATE_PROJECT = os.path.join("ml", "models", "yolo_candidate_training")
METADATA_SAVE_PATH = os.path.join("ml", "models", "vision", "yolo11_metadata.json")
COMPARE_PATH = os.path.join("ml", "models", "vision", "yolo_baseline_vs_candidate.json")

NAMES = {0: "cut", 1: "bruise", 2: "wound"}


def prepare_training_dataset():
    """Use the clean processed set. Never remap abrasion/laceration/swelling to wound."""
    if not os.path.exists(YOLO_PROCESSED_YAML):
        raise FileNotFoundError(
            "Processed YOLO dataset missing. Run "
            "ml/training/prepare_yolo_processed_dataset.py first. "
            "Silent class remaps are disabled."
        )
    return YOLO_PROCESSED_YAML


def prepare_merged_dataset():
    """Back-compat alias. Does not rebuild yolo_merged and does not remap classes."""
    return prepare_training_dataset()


def _count_dataset():
    from collections import Counter
    stats = {}
    for split in ("train", "val", "test"):
        imgs = glob.glob(os.path.join(YOLO_PROCESSED_ROOT, "images", split, "*"))
        boxes = Counter()
        img_classes = Counter()
        for img in imgs:
            stem = os.path.splitext(os.path.basename(img))[0]
            lf = os.path.join(YOLO_PROCESSED_ROOT, "labels", split, stem + ".txt")
            present = set()
            if os.path.exists(lf):
                with open(lf, encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            cid = int(float(parts[0]))
                            boxes[NAMES.get(cid, str(cid))] += 1
                            present.add(NAMES.get(cid, str(cid)))
            for c in present:
                img_classes[c] += 1
        stats[split] = {
            "images": len(imgs),
            "boxes": dict(boxes),
            "images_per_class": dict(img_classes),
        }
    return stats


def _eval_checkpoint(weights, data_yaml):
    from ultralytics import YOLO
    model = YOLO(weights)
    res = model.val(data=data_yaml, split="val", verbose=False)
    box = res.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "names": dict(model.names),
        "task": model.task,
    }


def train_yolo(epochs: int = 8, imgsz: int = 640, batch: int = 4):
    """Train a candidate YOLO11n and promote only if val mAP50-95 improves vs baseline."""
    from ultralytics import YOLO

    data_yaml = prepare_training_dataset()
    stats = _count_dataset()
    print("Processed YOLO dataset:", json.dumps(stats, indent=2))

    baseline = YOLO_CANONICAL if os.path.exists(YOLO_CANONICAL) else YOLO_BASELINE_COPY
    if not os.path.exists(baseline):
        raise FileNotFoundError("No baseline YOLO checkpoint to compare against.")

    print("Evaluating baseline...")
    baseline_metrics = _eval_checkpoint(baseline, data_yaml)
    print("Baseline:", baseline_metrics)

    pretrained = "yolo11n.pt"
    local_pretrained = os.path.join("ml", "models", "yolo11n_pretrained.pt")
    if os.path.exists("yolo11n.pt"):
        pretrained = "yolo11n.pt"
    elif os.path.exists(local_pretrained):
        pretrained = local_pretrained
    model = YOLO(pretrained)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Training candidate YOLO11n on {device} for {epochs} epochs")
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.abspath(CANDIDATE_PROJECT),
        name="run_merged",
        exist_ok=True,
        seed=42,
        pretrained=True,
        verbose=True,
    )
    candidate = os.path.join(CANDIDATE_PROJECT, "run_merged", "weights", "best.pt")
    save_dir = getattr(train_results, "save_dir", None)
    if save_dir:
        alt = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.exists(alt):
            candidate = alt
    if not os.path.exists(candidate):
        fallback = os.path.join("C:\\Users\\santh\\runs\\detect\\ml\\models\\yolo_candidate_training\\run_merged\\weights\\best.pt")
        if os.path.exists(fallback):
            candidate = fallback
    if not os.path.exists(candidate):
        raise FileNotFoundError("Candidate best.pt missing after training.")

    candidate_metrics = _eval_checkpoint(candidate, data_yaml)
    print("Candidate:", candidate_metrics)

    improved = candidate_metrics["mAP50_95"] >= baseline_metrics["mAP50_95"]
    selected = candidate if improved else baseline
    selected_label = "candidate" if improved else "baseline"

    os.makedirs(os.path.dirname(YOLO_CANONICAL), exist_ok=True)
    if improved:
        backup = YOLO_CANONICAL + ".pre_retrain_backup"
        if os.path.exists(YOLO_CANONICAL) and not os.path.exists(backup):
            shutil.copy2(YOLO_CANONICAL, backup)
        shutil.copy2(candidate, YOLO_CANONICAL)

    comparison = {
        "baseline_path": baseline,
        "candidate_path": candidate,
        "selected": selected_label,
        "selected_path": YOLO_CANONICAL if improved else baseline,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "dataset_stats": stats,
        "epochs": epochs,
        "improved": improved,
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "note": "Promoted only if candidate val mAP50-95 >= baseline.",
    }
    with open(COMPARE_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    selected_metrics = candidate_metrics if improved else baseline_metrics
    metadata = {
        "model_name": "YOLO11 Detection",
        "version": "v1.3.0",
        "status": "TRAINED_AND_EVALUATED",
        "classes": list(NAMES.values()),
        "untrained_classes": ["abrasion", "laceration", "swelling"],
        "metrics": {
            "dataset_name": "yolo_processed (cut/bruise/wound, no silent remap)",
            "dataset_stats": stats,
            "precision": selected_metrics["precision"],
            "recall": selected_metrics["recall"],
            "mAP50": selected_metrics["mAP50"],
            "mAP50-95": selected_metrics["mAP50_95"],
            "metrics_source": "ultralytics.val on merged val split",
            "selected_checkpoint": selected_label,
        },
        "weights_path": YOLO_CANONICAL,
        "training_was_real": True,
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    from ml.models.model_registry_manager import register_model_artifact
    register_model_artifact(
        model_name="YOLO11 Detection",
        version="v1.3.0",
        artifact_path=YOLO_CANONICAL,
        training_dataset="yolo_processed",
        sample_count=stats["train"]["images"],
        classes=list(NAMES.values()),
        metrics=metadata["metrics"],
        training_command="backend\\venv\\Scripts\\python.exe ml\\training\\train_yolo.py",
        notes=f"Real Ultralytics train. Selected {selected_label}. abrasion/laceration/swelling are UNTRAINED_CLASS.",
    )
    print(f"[OK] YOLO selected={selected_label} mAP50-95={selected_metrics['mAP50_95']:.4f}")
    return comparison


if __name__ == "__main__":
    train_yolo()
