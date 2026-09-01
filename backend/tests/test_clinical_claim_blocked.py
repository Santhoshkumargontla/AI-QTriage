"""Fusion clinical claims stay blocked until real paired labels exist."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_analyze_response_blocks_clinical_claim():
    resp = client.post("/api/cases", json={"notes": "clinical claim gate"})
    assert resp.status_code == 201
    case_id = resp.json()["case_id"]

    q = client.post(
        f"/api/cases/{case_id}/questionnaire",
        json={
            "answers": {
                "pain_level": "4",
                "cause": "fall",
                "crack_pop": "no",
            },
            "answer_source": "typed",
        },
    )
    assert q.status_code == 200, q.text
    client.post(f"/api/cases/{case_id}/sensor/skip")

    analyzed = client.post(f"/api/cases/{case_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    data = analyzed.json()

    assert data.get("clinical_claim_blocked") is True
    assert data.get("fusion_label_source") == "SYNTHETIC_RULE_LABELS"
    assert int(data.get("paired_clinical_samples", 0)) == 0
    assert "BLOCKED" in str(data.get("clinical_claim", "")).upper()

    xgb = data.get("xgboost") or {}
    assert xgb.get("clinical_claim_blocked") is True
    assert xgb.get("label_source") == "SYNTHETIC_RULE_LABELS"
    assert int(xgb.get("paired_clinical_samples", 0)) == 0
    assert "BLOCKED" in str(xgb.get("clinical_claim", "")).upper()

    stored = client.get(f"/api/cases/{case_id}")
    assert stored.status_code == 200
    case = stored.json()
    assert case.get("clinical_claim_blocked") is True
    assert case.get("fusion_label_source") == "SYNTHETIC_RULE_LABELS"
    assert int(case.get("paired_clinical_samples", 0)) == 0
