"""Independent E2E + YOLO box localization forensic (not demo presets)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("PYTHONPATH", str(ROOT))

OUT = ROOT / "scratch" / "independent_e2e_2026_08_30"
OUT.mkdir(parents=True, exist_ok=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_controlled_cut() -> tuple[Path, list[int]]:
    """Skin-like canvas with a dark-red cut in BOTTOM-RIGHT. Returns path + GT xyxy."""
    h, w = 720, 960
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (145, 170, 205)  # BGR skin-ish
    rng = np.random.default_rng(42)
    noise = rng.integers(-12, 12, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(80, 200, 25):
        cv2.line(img, (40, y), (300, y + 10), (120, 140, 180), 1)

    gt = [680, 480, 860, 620]
    x1, y1, x2, y2 = gt
    cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (90, 55), 20, 0, 360, (90, 70, 140), -1)
    cv2.line(img, (x1 + 10, y1 + 20), (x2 - 15, y2 - 25), (20, 25, 180), 6)
    cv2.line(img, (x1 + 20, y1 + 40), (x2 - 30, y2 - 10), (15, 20, 160), 4)

    path = OUT / "controlled_cut_bottom_right.png"
    cv2.imwrite(str(path), img)
    return path, gt


def make_bruise_center() -> tuple[Path, list[int]]:
    h, w = 640, 640
    img = np.full((h, w, 3), (160, 185, 210), np.uint8)
    rng = np.random.default_rng(7)
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 10, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    gt = [220, 220, 420, 420]
    cx, cy = (gt[0] + gt[2]) // 2, (gt[1] + gt[3]) // 2
    cv2.ellipse(img, (cx, cy), (95, 80), 0, 0, 360, (100, 60, 120), -1)
    cv2.ellipse(img, (cx - 20, cy - 10), (40, 30), 15, 0, 360, (70, 40, 90), -1)
    path = OUT / "controlled_bruise_center.png"
    cv2.imwrite(str(path), img)
    return path, gt


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def probe_yolo(path: Path, gt: list[int], tag: str) -> dict:
    from ml.vision.yolo_wrapper import YOLO11Detector

    det = YOLO11Detector()
    findings = det.detect(str(path))
    img = cv2.imread(str(path))
    vis = img.copy()
    cv2.rectangle(vis, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
    cv2.putText(vis, "GT", (gt[0], gt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    rows = []
    for f in findings:
        b = [float(x) for x in f["bounding_box"]]
        score = iou(gt, b)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"{f['finding']} {f['confidence']:.2f} iou={score:.2f}",
            (int(b[0]), max(20, int(b[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        rows.append({**f, "iou_vs_gt": round(score, 4)})
    overlay = OUT / f"{tag}_yolo_overlay.jpg"
    cv2.imwrite(str(overlay), vis)
    return {
        "tag": tag,
        "path": str(path),
        "sha256": _sha(path),
        "gt": gt,
        "image_shape_hw": list(img.shape[:2]),
        "findings": rows,
        "overlay": str(overlay),
        "best_iou": max((r["iou_vs_gt"] for r in rows), default=0.0),
        "coord_system_note": "Ultralytics xyxy vs GT in original pixel space",
    }


def simulate_frontend_box(box, ow, oh, disp_w, disp_h, object_fit: str) -> dict:
    """Reproduce FE percentage mapping + optional CSS object-fit mismatch."""
    left_pct = box[0] / ow
    top_pct = box[1] / oh
    width_pct = (box[2] - box[0]) / ow
    height_pct = (box[3] - box[1]) / oh
    # With object-fill and matching aspect ratio, percentages map 1:1 to display pixels.
    if object_fit == "fill" and abs((disp_w / disp_h) - (ow / oh)) < 1e-6:
        return {
            "fit": object_fit,
            "display_box": [
                left_pct * disp_w,
                top_pct * disp_h,
                (left_pct + width_pct) * disp_w,
                (top_pct + height_pct) * disp_h,
            ],
            "aligned": True,
        }
    # object-contain letterbox simulation when aspect differs
    scale = min(disp_w / ow, disp_h / oh)
    content_w, content_h = ow * scale, oh * scale
    pad_x = (disp_w - content_w) / 2
    pad_y = (disp_h - content_h) / 2
    return {
        "fit": object_fit,
        "display_box": [
            pad_x + box[0] * scale,
            pad_y + box[1] * scale,
            pad_x + box[2] * scale,
            pad_y + box[3] * scale,
        ],
        "aligned": abs((disp_w / disp_h) - (ow / oh)) < 1e-6 and object_fit in {"fill", "contain"},
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale": scale,
    }


def e2e_case(client: TestClient, image_path: Path, tag: str) -> dict:
    create = client.post("/api/cases", json={"notes": f"Independent non-demo {tag}"})
    case_id = create.json()["case_id"]
    with open(image_path, "rb") as handle:
        up = client.post(
            f"/api/cases/{case_id}/image",
            files={"file": (image_path.name, handle, "image/png")},
        )
    q = client.post(
        f"/api/cases/{case_id}/questionnaire",
        json={
            "answers": {
                "pain_level": 6,
                "location": "Arm",
                "open_wound": "yes",
                "bruising_discoloration": "yes",
                "mechanism": "sharp",
            },
            "answer_source": "typed",
        },
    )
    sensor = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": "sudden_impact"})
    if sensor.status_code >= 400:
        sensor = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": "sudden_fall"})
    if sensor.status_code >= 400:
        sensor = client.post(f"/api/cases/{case_id}/sensor/skip")
    analyze = client.post(f"/api/cases/{case_id}/analyze")
    got = client.get(f"/api/cases/{case_id}")
    body = got.json() if got.status_code == 200 else {"error": got.text}
    vi = (body or {}).get("visible_injury") or {}
    return {
        "tag": tag,
        "case_id": case_id,
        "file_sha": _sha(image_path),
        "upload_status": up.status_code,
        "upload_detail": up.json() if up.headers.get("content-type", "").startswith("application/json") else up.text[:300],
        "questionnaire_status": q.status_code,
        "sensor_status": sensor.status_code,
        "analyze_status": analyze.status_code,
        "analyze_detail": (
            analyze.json()
            if analyze.status_code < 500 and analyze.headers.get("content-type", "").startswith("application/json")
            else analyze.text[:500]
        ),
        "get_status": got.status_code,
        "api": {
            "image_sha256": vi.get("image_sha256"),
            "yolo_finding": vi.get("yolo_finding"),
            "yolo_confidence": vi.get("yolo_confidence"),
            "yolo_bounding_box": vi.get("yolo_bounding_box"),
            "bounding_box": vi.get("bounding_box"),
            "original_width": vi.get("original_width"),
            "original_height": vi.get("original_height"),
            "overlay_width": vi.get("overlay_width"),
            "overlay_height": vi.get("overlay_height"),
            "classifier_finding": vi.get("classifier_finding"),
            "classifier_is_confident": vi.get("classifier_is_confident"),
            "classifier_status": vi.get("classifier_status"),
            "source_type": vi.get("source_type"),
            "data_provenance": vi.get("data_provenance"),
            "segmentation_reliable": vi.get("segmentation_reliable"),
        },
        "xgb": (body.get("xgboost_prediction") or {}).get("class"),
        "vqc_used": (body.get("quantum_prediction") or {}).get("used_in_main_decision"),
        "clinical_claim": body.get("clinical_claim"),
        "modalities": body.get("modalities_used"),
    }


def main():
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "probes": {}, "e2e": [], "fe_sim": {}}

    cut_path, cut_gt = make_controlled_cut()
    bruise_path, bruise_gt = make_bruise_center()
    report["probes"]["cut"] = probe_yolo(cut_path, cut_gt, "controlled_cut")
    report["probes"]["bruise"] = probe_yolo(bruise_path, bruise_gt, "controlled_bruise")

    # FE coordinate simulation for best bruise detection if any
    bruise_findings = report["probes"]["bruise"]["findings"]
    if bruise_findings:
        best = max(bruise_findings, key=lambda r: r["confidence"])
        ow = report["probes"]["bruise"]["image_shape_hw"][1]
        oh = report["probes"]["bruise"]["image_shape_hw"][0]
        report["fe_sim"] = {
            "object_fill_matching_aspect": simulate_frontend_box(best["bounding_box"], ow, oh, 480, 480, "fill"),
            "object_contain_mismatched_aspect": simulate_frontend_box(best["bounding_box"], ow, oh, 480, 300, "contain"),
        }

    from backend.main import app

    client = TestClient(app)
    for path, tag in [(cut_path, "controlled_cut"), (bruise_path, "controlled_bruise")]:
        try:
            report["e2e"].append(e2e_case(client, path, tag))
        except Exception as exc:
            report["e2e"].append({"tag": tag, "error": str(exc)})

    # Compare direct YOLO vs API box for bruise case
    for row in report["e2e"]:
        tag = row.get("tag")
        probe = report["probes"].get("cut" if "cut" in tag else "bruise", {})
        direct = (probe.get("findings") or [None])[0]
        api_box = (row.get("api") or {}).get("yolo_bounding_box")
        if direct and api_box:
            row["direct_vs_api_box_match"] = (
                [round(x, 2) for x in direct["bounding_box"]] == [round(float(x), 2) for x in api_box]
            )
            row["direct_box"] = direct["bounding_box"]
            row["api_box"] = api_box

    out_json = OUT / "independent_e2e_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WROTE", out_json)


if __name__ == "__main__":
    main()
