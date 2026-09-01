"""Canonical SOS delivery outcomes. These are application statuses, not Twilio's raw status field."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOCAL_SIMULATION = "LOCAL_SIMULATION"
TWILIO_NOT_CONFIGURED = "TWILIO_NOT_CONFIGURED"
TWILIO_REQUEST_QUEUED = "TWILIO_REQUEST_QUEUED"
TWILIO_FAILED = "TWILIO_FAILED"

DELIVERY_OUTCOMES = (
    LOCAL_SIMULATION,
    TWILIO_NOT_CONFIGURED,
    TWILIO_REQUEST_QUEUED,
    TWILIO_FAILED,
)

# Workflow states that are not delivery claims.
WORKFLOW_COUNTDOWN = "countdown"
WORKFLOW_TRIGGERED = "triggered"
WORKFLOW_SENDING = "sending"
WORKFLOW_CANCELLED = "cancelled"
WORKFLOW_ABORTED = "aborted"

# Documented legacy names still accepted when reading stored documents.
_LEGACY_OUTCOME = {
    "demo_triggered": LOCAL_SIMULATION,
    "dispatched": LOCAL_SIMULATION,
    "twilio_accepted": TWILIO_REQUEST_QUEUED,
    "twilio_failed": TWILIO_FAILED,
    "not_configured": TWILIO_NOT_CONFIGURED,
    "missing_phone_numbers": TWILIO_NOT_CONFIGURED,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_outcome(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    if status in DELIVERY_OUTCOMES:
        return status
    return _LEGACY_OUTCOME.get(status, status)


def interpret_twilio_result(twilio_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a TwilioService return dict to a canonical outcome. SID required for queued."""
    res = twilio_res or {}
    sid = res.get("twilio_message_sid") or res.get("sid")
    if sid:
        sid = str(sid).strip() or None
    timestamp = res.get("timestamp") or utc_now_iso()
    provider = res.get("provider_status") or res.get("delivery_status")
    raw_status = res.get("status") or res.get("delivery_outcome")

    if res.get("success") and sid:
        return {
            "outcome": TWILIO_REQUEST_QUEUED,
            "twilio_message_sid": sid,
            "provider_status": provider or (raw_status if raw_status not in DELIVERY_OUTCOMES else "queued"),
            "timestamp": timestamp,
            "failure_reason": None,
            "error_code": None,
        }

    if raw_status in (TWILIO_NOT_CONFIGURED, "not_configured", "missing_phone_numbers") or res.get("configured") is False:
        outcome = TWILIO_NOT_CONFIGURED
    else:
        outcome = TWILIO_FAILED

    return {
        "outcome": outcome,
        "twilio_message_sid": None,
        "provider_status": provider,
        "timestamp": timestamp,
        "failure_reason": res.get("failure_reason") or res.get("message") or "Twilio request did not return a message SID.",
        "error_code": res.get("error_code"),
    }


def persist_outcome(
    db,
    case_id: str,
    event_id: Optional[str],
    outcome: str,
    *,
    twilio_message_sid: Optional[str] = None,
    provider_status: Optional[str] = None,
    timestamp: Optional[str] = None,
    failure_reason: Optional[str] = None,
    error_code: Any = None,
    extra_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write canonical outcome + SID/provider/timestamp/failure to case and event. Never writes SMS_SENT."""
    if outcome == "SMS_SENT":
        raise ValueError("SMS_SENT is not an application status. Use TWILIO_REQUEST_QUEUED when a SID is present.")
    ts = timestamp or utc_now_iso()
    event_set: Dict[str, Any] = {
        "sos_status": outcome,
        "delivery_outcome": outcome,
        "twilio_message_sid": twilio_message_sid,
        "provider_status": provider_status,
        "delivery_status": provider_status,
        "send_timestamp": ts,
        "failure_reason": failure_reason,
        "twilio_error_message": failure_reason,
        "twilio_error_code": error_code,
        "resolved_at": ts,
    }
    if extra_event:
        event_set.update(extra_event)
    if event_id:
        db.sos_events.update_one({"event_id": event_id}, {"$set": event_set})
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sos_status": outcome,
            "sos_delivery_outcome": outcome,
            "sos_twilio_sid": twilio_message_sid,
            "sos_provider_status": provider_status,
            "sos_delivery_status": provider_status,
            "sos_send_timestamp": ts,
            "sos_failure_reason": failure_reason,
            "sos_twilio_error": failure_reason,
            "sos_twilio_error_code": error_code,
        }},
    )
    return {
        "status": outcome,
        "delivery_outcome": outcome,
        "twilio_message_sid": twilio_message_sid,
        "provider_status": provider_status,
        "timestamp": ts,
        "failure_reason": failure_reason,
        "twilio_error_code": error_code,
        "event_id": event_id,
    }


def persist_local_simulation(db, case_id: str, event_id: Optional[str], extra_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extra = {"user_response": "no_response"}
    if extra_event:
        extra.update(extra_event)
    return persist_outcome(
        db,
        case_id,
        event_id,
        LOCAL_SIMULATION,
        extra_event=extra,
    )


def persist_twilio_result(db, case_id: str, event_id: Optional[str], twilio_res: Optional[Dict[str, Any]], extra_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    interpreted = interpret_twilio_result(twilio_res)
    extra = {"twilio_provider": "twilio"}
    if extra_event:
        extra.update(extra_event)
    return persist_outcome(
        db,
        case_id,
        event_id,
        interpreted["outcome"],
        twilio_message_sid=interpreted["twilio_message_sid"],
        provider_status=interpreted["provider_status"],
        timestamp=interpreted["timestamp"],
        failure_reason=interpreted["failure_reason"],
        error_code=interpreted["error_code"],
        extra_event=extra,
    )


def status_response_fields(case: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ev = event or {}
    raw = ev.get("sos_status") or case.get("sos_status")
    outcome = normalize_outcome(ev.get("delivery_outcome") or case.get("sos_delivery_outcome") or raw)
    sid = ev.get("twilio_message_sid") or case.get("sos_twilio_sid")
    provider = ev.get("provider_status") or ev.get("delivery_status") or case.get("sos_provider_status") or case.get("sos_delivery_status")
    failure = ev.get("failure_reason") or ev.get("twilio_error_message") or case.get("sos_failure_reason") or case.get("sos_twilio_error")
    ts = ev.get("send_timestamp") or case.get("sos_send_timestamp")
    user_location = ev.get("user_location") or case.get("sos_user_location")
    return {
        "status": raw,
        "delivery_outcome": outcome if outcome in DELIVERY_OUTCOMES else None,
        "event_id": ev.get("event_id") or case.get("active_sos_event_id") or case.get("sos_event_id"),
        "twilio_message_sid": sid,
        "provider_status": provider,
        "delivery_status": provider,
        "timestamp": ts,
        "failure_reason": failure,
        "twilio_error_message": failure,
        "twilio_error_code": ev.get("twilio_error_code") or case.get("sos_twilio_error_code"),
        "user_location": user_location,
        "sms_sent": False,
    }
