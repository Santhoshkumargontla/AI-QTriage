"""One-shot create-case flow (no dashboard demo) + Twilio SOS send."""
import json
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
IMG = Path(
    r"data\raw\roboflow\injury_detection_v2\test\images\bruises-10-_jpg.rf.576c19d548a47ae327530ebb44b18697.jpg"
)


def req(method, path, data=None, files=None):
    url = BASE + path
    headers = {}
    body = None
    if files is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for name, (filename, content, ctype) in files.items():
            parts.append(f"--{boundary}".encode())
            parts.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
            )
            parts.append(f"Content-Type: {ctype}".encode())
            parts.append(b"")
            parts.append(content)
        parts.append(f"--{boundary}--".encode())
        parts.append(b"")
        body = b"\r\n".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        print("HTTP", exc.code, path, raw[:1000])
        raise


def main():
    code, cfg = req("GET", "/api/sos/config")
    safe = {
        k: cfg.get(k)
        for k in (
            "configured",
            "enabled",
            "twilio_enabled",
            "status_message",
            "missing_fields",
            "status",
            "disclaimer",
        )
        if k in cfg
    }
    print("SOS_CONFIG", json.dumps(safe, default=str))

    code, case = req("POST", "/api/cases", data={})
    case_id = case.get("case_id") or case.get("id")
    print("CASE", case_id)

    img_bytes = IMG.read_bytes()
    code, up = req(
        "POST",
        f"/api/cases/{case_id}/image",
        files={"file": (IMG.name, img_bytes, "image/jpeg")},
    )
    print("IMAGE_UPLOAD", code)

    print("ANALYZE1...")
    code, a1 = req("POST", f"/api/cases/{case_id}/analyze", data={})
    vi = a1.get("visible_injury") or {}
    print(
        "YOLO",
        vi.get("yolo_finding"),
        vi.get("yolo_confidence"),
        "detected",
        vi.get("yolo_finding_detected"),
    )
    print("EFFNET", vi.get("classifier_finding"), vi.get("classifier_status"))

    # Sample answers consistent with a bruise photo (no open wound).
    answers = {
        "location": "Left lower leg / shin area",
        "pain_level": "5",
        "cause": "sports",
        "onset_hours": "1-3 hours ago",
        "movement_limitation": "mild",
        "weight_bearing": "full",
        "swelling": "yes",
        "bruising_discoloration": "yes",
        "redness": "no",
        "warmth": "no",
        "open_wound": "no",
        "bleeding": "none",
        "crack_pop": "no",
        "deformity": "no",
        "numbness_tingling": "no",
        "previous_injury": "no",
        "symptom_progression": "stable",
        "direct_impact": "yes",
    }
    code, q = req("POST", f"/api/cases/{case_id}/questionnaire", data={"answers": answers})
    print("QUESTIONNAIRE", code)

    code, s = req("POST", f"/api/cases/{case_id}/sensor/demo", data={})
    print(
        "SENSOR_DEMO",
        code,
        s.get("sensor_source_type") or s.get("source_type") or "loaded",
    )

    print("ANALYZE2...")
    code, a2 = req("POST", f"/api/cases/{case_id}/analyze", data={})
    print(
        "XGB",
        (a2.get("xgboost_prediction") or {}).get("class"),
        (a2.get("xgboost_prediction") or {}).get("probability"),
    )
    print(
        "VQC",
        (a2.get("quantum_prediction") or {}).get("class"),
        (a2.get("quantum_prediction") or {}).get("status"),
    )

    code, trig = req(
        "POST",
        f"/api/cases/{case_id}/sos/demo/trigger",
        data={"mode": "twilio_test"},
    )
    print("SOS_TRIGGER", json.dumps(trig, default=str)[:800])

    print("Waiting 11s for countdown expiry path...")
    time.sleep(11)
    code, resp = req(
        "POST",
        f"/api/cases/{case_id}/sos/demo/respond",
        data={"user_response": "no_response", "mode": "twilio_test"},
    )
    print("SOS_RESPOND", json.dumps(resp, default=str)[:1200])

    code, st = req("GET", f"/api/cases/{case_id}/sos/status")
    print("SOS_STATUS", json.dumps(st, default=str)[:1200])
    print("CASE_URL", f"http://localhost:3000/cases/{case_id}")
    print("DONE")


if __name__ == "__main__":
    main()
