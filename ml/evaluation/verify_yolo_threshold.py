"""Verify configurable YOLO thresholds against sweep evidence."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.vision.yolo_wrapper import (
    CONSERVATIVE_YOLO_INFER_CONF,
    DEFAULT_YOLO_INFER_CONF,
    YOLO11Detector,
)

REPORT = os.path.join("ml", "models", "yolo_threshold_eval", "THRESHOLD_SWEEP_REPORT.json")


def main():
    with open(REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    report["recommendation"] = {
        "research_demo_threshold": DEFAULT_YOLO_INFER_CONF,
        "conservative_threshold": CONSERVATIVE_YOLO_INFER_CONF,
        "default_runtime_threshold": DEFAULT_YOLO_INFER_CONF,
        "do_not_use": 0.10,
        "one_global_threshold_appropriate": True,
        "class_specific_would_help": (
            "Raising wound (optional) can cut wound-only FPs. "
            "Lowering cut does not help: cut TP scores overlap cut FP scores (~0.01-0.02)."
        ),
        "selection_basis": "processed val+test IoU@0.5, not the demo image",
        "not_clinical_accuracy": True,
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    os.environ.pop("YOLO_CONF_THRESHOLD", None)
    det = YOLO11Detector()
    info = det.get_info()
    print("infer_conf", det.infer_conf)
    print("class_thresholds", det.class_thresholds)
    print("info.infer_conf", info["infer_conf"])
    assert det.infer_conf == 0.25

    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    upload = os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg")

    for label, path, conf in [
        ("blank_default", blank, None),
        ("dummy_default", dummy, None),
        ("demo_default_0.25", demo, None),
        ("demo_conservative_0.30", demo, 0.30),
        ("demo_old_0.10", demo, 0.10),
        ("upload_default", upload, None),
    ]:
        if not os.path.exists(path):
            print(label, "MISSING")
            continue
        kwargs = {} if conf is None else {"conf": conf}
        findings = det.detect(path, **kwargs)
        print(
            label,
            "n=",
            len(findings),
            [(f["finding"], f["confidence"]) for f in findings[:5]],
        )


if __name__ == "__main__":
    main()
