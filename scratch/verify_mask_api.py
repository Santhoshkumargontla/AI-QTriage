"""API TestClient: mask PNG must match original image size after analyze."""
import os, sys, json
sys.path.insert(0, os.path.abspath("."))
from fastapi.testclient import TestClient
from backend.main import app
import cv2

client = TestClient(app)
img_path = os.path.join("data", "sample", "image", "football_injury.jpg")
orig = cv2.imread(img_path)
oh, ow = orig.shape[:2]
r = client.post("/api/cases", json={})
case_id = r.json()["case_id"]
with open(img_path, "rb") as f:
    up = client.post(f"/api/cases/{case_id}/image", files={"file": ("demo.jpg", f, "image/jpeg")})
print("upload", up.status_code)
an = client.post(f"/api/cases/{case_id}/analyze")
print("analyze", an.status_code)
case = client.get(f"/api/cases/{case_id}").json()
vi = case.get("visible_injury") or {}
print("orig", ow, oh)
print("api orig", vi.get("original_width"), vi.get("original_height"), "overlay", vi.get("overlay_width"), vi.get("overlay_height"))
print("yolo", vi.get("yolo_finding_detected"), vi.get("yolo_finding"), vi.get("yolo_confidence"), vi.get("bounding_box"))
print("coverage", vi.get("classifier_yolo_coverage"), "clf", vi.get("classifier_finding"), vi.get("classifier_status"))
print("is_demo", case.get("is_demo"), "filename_in_ref", "football_injury" in str(case.get("image_reference")))
mask_path = vi.get("mask_path")
print("mask_path", mask_path)
if mask_path and os.path.exists(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print("mask_shape", None if m is None else m.shape, "matches_orig", m is not None and m.shape[0] == oh and m.shape[1] == ow)
    if m is not None and vi.get("bounding_box"):
        x1,y1,x2,y2 = [int(v) for v in vi["bounding_box"]]
        outside = m.copy()
        outside[y1:y2, x1:x2] = 0
        print("positive_outside_bbox", int((outside > 0).sum()), "positive_inside", int((m[y1:y2, x1:x2] > 0).sum()))
print("xgb", (case.get("xgboost_prediction") or {}).get("class"))
print("vqc_status", (case.get("quantum_prediction") or {}).get("status") or case.get("quantum_prediction"))
