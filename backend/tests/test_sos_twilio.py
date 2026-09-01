"""SOS / Twilio workflow: canonical env, outcomes, no SMS_SENT without a SID."""
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import get_database
from backend.services.sos_delivery import (
    LOCAL_SIMULATION,
    TWILIO_FAILED,
    TWILIO_NOT_CONFIGURED,
    TWILIO_REQUEST_QUEUED,
)
from backend.services.twilio_service import CANONICAL_ENV, LEGACY_ALIASES, TwilioService, twilio_service

client = TestClient(app)


def _case(notes="sos twilio"):
    resp = client.post("/api/cases", json={"notes": notes})
    assert resp.status_code == 201
    return resp.json()["case_id"]


def _cleanup(case_id):
    db = get_database()
    db.cases.delete_one({"case_id": case_id})
    db.sos_events.delete_many({"case_id": case_id})


def _mock_svc(**overrides):
    svc = TwilioService.__new__(TwilioService)
    svc._manual_mock = True
    svc.enabled = True
    svc.account_sid = "ACtest12345678901234567890123456"
    svc.auth_token = "test_auth_token"
    svc.from_number = "+15550001111"
    svc.to_number = "+15550002222"
    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


def test_config_endpoint_canonical_and_no_secrets():
    r = client.get("/api/sos/config")
    assert r.status_code == 200
    data = r.json()
    assert data["canonical_env"] == list(CANONICAL_ENV)
    assert data["legacy_aliases"] == LEGACY_ALIASES
    assert set(data["delivery_outcomes"]) == {
        LOCAL_SIMULATION,
        TWILIO_NOT_CONFIGURED,
        TWILIO_REQUEST_QUEUED,
        TWILIO_FAILED,
    }
    blob = str(data)
    assert data.get("auth_token") is None
    assert data.get("TWILIO_AUTH_TOKEN") is None
    assert "tok" not in blob.lower() or data["configured"] is False
    # Env var NAMES are listed; secret VALUES must not appear.
    assert twilio_service.auth_token not in blob if twilio_service.auth_token else True
    assert "real_sms_tested" in data
    if not data["configured"]:
        assert data["real_sms_tested"] is False


def test_local_simulation_api():
    case_id = _case()
    try:
        trig = client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "local_demo"})
        assert trig.status_code == 200
        assert trig.json()["event"]["sos_status"] == "countdown"
        resp = client.post(
            f"/api/cases/{case_id}/sos/demo/respond",
            json={"user_response": "no_response", "mode": "local_demo"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sos_status"] == LOCAL_SIMULATION
        assert body.get("twilio_message_sid") in (None, "")
        stored = get_database().cases.find_one({"case_id": case_id})
        assert stored["sos_status"] == LOCAL_SIMULATION
        assert stored.get("sos_twilio_sid") in (None, "")
        assert stored.get("sos_send_timestamp")
        assert "SMS_SENT" not in str(stored.get("sos_status"))
        st = client.get(f"/api/cases/{case_id}/sos/status").json()
        assert st["status"] == LOCAL_SIMULATION
        assert st.get("sms_sent") is False
    finally:
        _cleanup(case_id)


def test_disabled_twilio_is_not_configured():
    svc = _mock_svc(enabled=False)
    configured, msg = svc.is_configured()
    assert configured is False
    assert "TWILIO_ENABLED=false" in msg
    out = svc.send_test_sos_message(case_id="c1", sos_event_id="e1")
    assert out["success"] is False
    assert out["status"] == TWILIO_NOT_CONFIGURED
    assert out["twilio_message_sid"] is None


def test_missing_credentials_lists_field_names_only():
    svc = _mock_svc(account_sid="", auth_token="", from_number="", to_number="")
    configured, msg = svc.is_configured()
    assert configured is False
    assert "TWILIO_ACCOUNT_SID" in msg
    assert "TWILIO_AUTH_TOKEN" in msg
    assert "TWILIO_FROM_NUMBER" in msg
    assert "TWILIO_TO_NUMBER" in msg
    assert "test_auth_token" not in msg
    out = svc.send_test_sos_message(case_id="c1", sos_event_id="e1")
    assert out["status"] == TWILIO_NOT_CONFIGURED
    assert out["twilio_message_sid"] is None


def test_legacy_from_alias_is_accepted():
    svc = TwilioService.__new__(TwilioService)
    with patch.dict(os.environ, {
        "TWILIO_ENABLED": "true",
        "TWILIO_ACCOUNT_SID": "ACtest12345678901234567890123456",
        "TWILIO_AUTH_TOKEN": "tok",
        "TWILIO_FROM_NUMBER": "",
        "TWILIO_TO_NUMBER": "+15550002222",
        "TWILIO_PHONE_NUMBER": "+15550001111",
        "EMERGENCY_CONTACT_PHONE": "",
        "EMERGENCY_PHONE_NUMBER": "",
    }, clear=False):
        with patch("backend.services.twilio_service.load_dotenv"):
            svc.reload_config()
    svc._manual_mock = True
    assert svc.from_number == "+15550001111"
    configured, _ = svc.is_configured()
    assert configured is True


def test_mocked_twilio_success_requires_sid():
    svc = _mock_svc()
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"sid": "SMabc123queued", "status": "queued"}
        mock_post.return_value = mock_resp
        out = svc.send_test_sos_message(case_id="c-success", sos_event_id="e-success")
    assert out["success"] is True
    assert out["status"] == TWILIO_REQUEST_QUEUED
    assert out["twilio_message_sid"] == "SMabc123queued"
    assert out["provider_status"] == "queued"
    assert out.get("timestamp")
    assert out.get("failure_reason") is None
    assert "SMS_SENT" not in str(out["status"])


def test_http_success_without_sid_is_failed_not_sent():
    svc = _mock_svc()
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"status": "queued"}
        mock_post.return_value = mock_resp
        out = svc.send_test_sos_message(case_id="c-nosid", sos_event_id="e-nosid")
    assert out["success"] is False
    assert out["status"] == TWILIO_FAILED
    assert out["twilio_message_sid"] is None
    assert "SID" in (out.get("failure_reason") or "")


def test_mocked_twilio_failure():
    svc = _mock_svc()
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"code": 21211, "message": "The 'To' number is not a valid phone number."}
        mock_post.return_value = mock_resp
        out = svc.send_test_sos_message(case_id="c-fail", sos_event_id="e-fail")
    assert out["success"] is False
    assert out["status"] == TWILIO_FAILED
    assert out["twilio_message_sid"] is None
    assert out["error_code"] == "21211"
    assert out["failure_reason"]


def test_respond_twilio_test_when_disabled_stores_not_configured():
    case_id = _case()
    try:
        client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "twilio_test"})
        with patch("backend.services.twilio_service.twilio_service.send_test_sos_message") as mock_send:
            mock_send.return_value = {
                "success": False,
                "status": TWILIO_NOT_CONFIGURED,
                "configured": False,
                "message": "Twilio integration disabled (TWILIO_ENABLED=false).",
                "failure_reason": "Twilio integration disabled (TWILIO_ENABLED=false).",
                "twilio_message_sid": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            resp = client.post(
                f"/api/cases/{case_id}/sos/demo/respond",
                json={"user_response": "no_response", "mode": "twilio_test"},
            )
        assert resp.status_code == 200
        assert resp.json()["sos_status"] == TWILIO_NOT_CONFIGURED
        stored = get_database().cases.find_one({"case_id": case_id})
        assert stored["sos_status"] == TWILIO_NOT_CONFIGURED
        assert stored.get("sos_twilio_sid") in (None, "")
        assert stored.get("sos_failure_reason")
        assert stored.get("sos_send_timestamp")
    finally:
        _cleanup(case_id)


def test_respond_mocked_success_persists_sid():
    case_id = _case()
    try:
        client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "twilio_test"})
        sid = "SM" + uuid.uuid4().hex[:30]
        ts = datetime.now(timezone.utc).isoformat()
        with patch("backend.services.twilio_service.twilio_service.send_test_sos_message") as mock_send:
            mock_send.return_value = {
                "success": True,
                "status": TWILIO_REQUEST_QUEUED,
                "twilio_message_sid": sid,
                "provider_status": "queued",
                "timestamp": ts,
                "failure_reason": None,
            }
            resp = client.post(
                f"/api/cases/{case_id}/sos/demo/respond",
                json={"user_response": "no_response", "mode": "twilio_test"},
            )
        assert resp.status_code == 200
        assert resp.json()["sos_status"] == TWILIO_REQUEST_QUEUED
        stored = get_database().cases.find_one({"case_id": case_id})
        assert stored["sos_twilio_sid"] == sid
        assert stored["sos_provider_status"] == "queued"
        assert stored["sos_send_timestamp"]
        event = get_database().sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
        assert event["twilio_message_sid"] == sid
        assert event.get("provider_status") == "queued"
        got = client.get(f"/api/cases/{case_id}").json()
        assert got["sos_twilio_sid"] == sid
        assert got["sos_status"] == TWILIO_REQUEST_QUEUED
    finally:
        _cleanup(case_id)


def test_respond_mocked_failure_persists_reason():
    case_id = _case()
    try:
        client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "twilio_test"})
        with patch("backend.services.twilio_service.twilio_service.send_test_sos_message") as mock_send:
            mock_send.return_value = {
                "success": False,
                "status": TWILIO_FAILED,
                "error_code": "21608",
                "message": "The number is unverified.",
                "failure_reason": "The number is unverified.",
                "twilio_message_sid": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            resp = client.post(
                f"/api/cases/{case_id}/sos/demo/respond",
                json={"user_response": "no_response", "mode": "twilio_test"},
            )
        assert resp.json()["sos_status"] == TWILIO_FAILED
        stored = get_database().cases.find_one({"case_id": case_id})
        assert stored["sos_status"] == TWILIO_FAILED
        assert stored.get("sos_twilio_sid") in (None, "")
        assert "unverified" in stored.get("sos_failure_reason", "").lower()
        assert stored.get("sos_send_timestamp")
    finally:
        _cleanup(case_id)


def test_sms_sent_never_written_as_application_status():
    from backend.services.sos_delivery import persist_outcome
    db = get_database()
    case_id = "test-sms-sent-forbidden"
    db.cases.delete_one({"case_id": case_id})
    db.cases.insert_one({"case_id": case_id, "created_at": datetime.now(timezone.utc)})
    try:
        raised = False
        try:
            persist_outcome(db, case_id, None, "SMS_SENT", twilio_message_sid="SMfake")
        except ValueError:
            raised = True
        assert raised
        stored = db.cases.find_one({"case_id": case_id})
        assert stored.get("sos_status") != "SMS_SENT"
    finally:
        db.cases.delete_one({"case_id": case_id})
