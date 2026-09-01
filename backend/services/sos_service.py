from datetime import datetime, timezone
import requests
from backend.database.connection import get_database
from backend.services.sos_delivery import (
    LOCAL_SIMULATION,
    TWILIO_FAILED,
    TWILIO_NOT_CONFIGURED,
    TWILIO_REQUEST_QUEUED,
    persist_local_simulation,
    persist_outcome,
    persist_twilio_result,
    status_response_fields,
    utc_now_iso,
)

class SOSCountdownService:
    """Manages independent sensor-based emergency support alerts and warnings countdown states."""
    
    def __init__(self):
        pass

    def check_and_trigger(self, case_id: str, peak_g_force: float, stabilization_time: float) -> dict:
        """
        Evaluates kinetics to determine if an emergency alert countdown should trigger.
        Updates MongoDB case document with SOS parameters.
        Missing kinetics are FEATURE_MISSING — never treated as 0.0g / 0.0s.
        """
        db = get_database()
        try:
            g_val = float(peak_g_force) if peak_g_force is not None else None
            t_val = float(stabilization_time) if stabilization_time is not None else None
        except (TypeError, ValueError):
            g_val, t_val = None, None
        if g_val is None or t_val is None:
            return {
                "sos_triggered": False,
                "countdown_seconds": 0,
                "reason": "FEATURE_MISSING: peak_g_force or post_impact_stabilization_seconds unavailable.",
            }

        # Threshold check: peak g-force >= 4.0g AND stabilization time >= 1.5s
        if g_val >= 4.0 and t_val >= 1.5:
            now_iso = datetime.now(timezone.utc).isoformat()
            trigger_reason = f"Severe kinetic impact ({g_val:.1f}g) with prolonged stabilization time ({t_val:.1f}s) detected."
            
            db.cases.update_one(
                {"case_id": case_id},
                {"$set": {
                    "sos_status": "triggered",
                    "sos_trigger_time": now_iso,
                    "sos_countdown_seconds": 30,
                    "sos_trigger_reason": trigger_reason
                }}
            )
            return {
                "sos_triggered": True,
                "countdown_seconds": 30,
                "reason": trigger_reason
            }
            
        return {
            "sos_triggered": False,
            "countdown_seconds": 0,
            "reason": "Sensor logs do not satisfy the emergency alert countdown thresholds."
        }

    def abort_sos(self, case_id: str) -> dict:
        """Aborts the active SOS countdown."""
        db = get_database()
        case = db.cases.find_one({"case_id": case_id})
        if not case:
            raise ValueError("Case not found")
            
        db.cases.update_one(
            {"case_id": case_id},
            {"$set": {
                "sos_status": "aborted"
            }}
        )
        return {
            "message": "SOS countdown aborted successfully",
            "status": "aborted"
        }

    def get_sos_status(self, case_id: str) -> dict:
        """
        Computes remaining countdown time and updates the case state if expired.
        """
        db = get_database()
        case = db.cases.find_one({"case_id": case_id})
        if not case:
            raise ValueError("Case not found")
            
        status = case.get("sos_status", "inactive")
        
        # Poll Twilio only when a SID exists. Provider status is stored separately from the app outcome.
        if status in (TWILIO_REQUEST_QUEUED, "twilio_accepted"):
            event = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
            if event and event.get("twilio_message_sid"):
                msg_sid = event["twilio_message_sid"]
                curr_delivery = event.get("provider_status") or event.get("delivery_status")
                if curr_delivery not in ("delivered", "failed", "undelivered"):
                    from backend.services.twilio_service import twilio_service
                    if twilio_service.account_sid and twilio_service.auth_token:
                        try:
                            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_service.account_sid}/Messages/{msg_sid}.json"
                            r = requests.get(
                                url,
                                auth=(twilio_service.account_sid, twilio_service.auth_token),
                                timeout=5
                            )
                            if r.status_code == 200:
                                resp = r.json()
                                new_delivery = resp.get("status", curr_delivery)
                                err_code = resp.get("error_code")
                                err_msg = resp.get("error_message")
                                if new_delivery in ("failed", "undelivered"):
                                    persist_outcome(
                                        db,
                                        case_id,
                                        event.get("event_id"),
                                        TWILIO_FAILED,
                                        twilio_message_sid=msg_sid,
                                        provider_status=new_delivery,
                                        failure_reason=err_msg or f"Twilio reported message status '{new_delivery}'",
                                        error_code=err_code,
                                    )
                                    status = TWILIO_FAILED
                                else:
                                    persist_outcome(
                                        db,
                                        case_id,
                                        event.get("event_id"),
                                        TWILIO_REQUEST_QUEUED,
                                        twilio_message_sid=msg_sid,
                                        provider_status=new_delivery,
                                        failure_reason=err_msg,
                                        error_code=err_code,
                                    )
                                    status = TWILIO_REQUEST_QUEUED
                        except (requests.RequestException, ValueError, KeyError, OSError) as e:
                            print(f"Error checking Twilio message status: {str(e)[:120]}")

        event_doc = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
        case_refreshed = db.cases.find_one({"case_id": case_id}) or case
        status = case_refreshed.get("sos_status", status)

        if status not in ("triggered", "countdown"):
            payload = status_response_fields(case_refreshed, event_doc)
            payload["status"] = status
            payload["remaining_seconds"] = 0.0
            payload["reason"] = case_refreshed.get("sos_trigger_reason", "")
            return payload
            
        # Parse trigger timestamp and calculate elapsed time
        trigger_time_str = case.get("sos_trigger_time")
        if not trigger_time_str:
            return {"status": "inactive", "remaining_seconds": 0.0, "reason": ""}
            
        trigger_time = datetime.fromisoformat(trigger_time_str)
        if trigger_time.tzinfo is not None:
            trigger_time = trigger_time.replace(tzinfo=None)
        now = datetime.utcnow()
        
        elapsed = (now - trigger_time).total_seconds()
        countdown_total = float(case.get("sos_countdown_seconds", 30))
        
        remaining = max(0.0, countdown_total - elapsed)
        
        if remaining <= 0.0:
            mode = case.get("sos_delivery_mode", "local_demo").lower()

            if mode == "twilio_test":
                claimed_event = db.sos_events.find_one_and_update(
                    {"case_id": case_id, "sos_status": "countdown"},
                    {"$set": {
                        "sos_status": "sending",
                        "sending_started_at": utc_now_iso()
                    }},
                    return_document=True
                )
                if claimed_event:
                    db.cases.update_one(
                        {"case_id": case_id},
                        {"$set": {"sos_status": "sending"}}
                    )
                    from backend.services.twilio_service import twilio_service
                    vi = case.get("visible_injury") or {}
                    q_data = (case.get("questionnaire") or {}).get("answers") or {}
                    geo = case.get("sos_user_location") or claimed_event.get("user_location") or {}
                    location_fallback = (
                        geo.get("location_label")
                        or q_data.get("location")
                        or "GPS unavailable"
                    )
                    if geo.get("latitude") is None or geo.get("longitude") is None:
                        reason = "Twilio SOS blocked: user GPS location missing."
                        applied = persist_twilio_result(
                            db,
                            case_id,
                            claimed_event["event_id"],
                            {
                                "success": False,
                                "status": TWILIO_FAILED,
                                "delivery_outcome": TWILIO_FAILED,
                                "failure_reason": reason,
                                "message": reason,
                                "twilio_message_sid": None,
                                "provider_status": None,
                                "timestamp": utc_now_iso(),
                            },
                            extra_event={"user_response": "no_response", "user_location": geo},
                        )
                        status = applied["status"]
                    else:
                        if vi.get("yolo_finding_detected") and vi.get("yolo_finding"):
                            yolo_finding = vi["yolo_finding"]
                            classifier_category = vi.get("classifier_finding")
                        else:
                            yolo_finding = None
                            classifier_category = vi.get("classifier_finding") or vi.get("finding")

                        twilio_res = twilio_service.send_test_sos_message(
                            case_id=case_id,
                            yolo_finding=yolo_finding,
                            classifier_category=classifier_category,
                            user_location=location_fallback,
                            latitude=geo.get("latitude"),
                            longitude=geo.get("longitude"),
                            maps_url=geo.get("maps_url"),
                            sos_event_id=claimed_event["event_id"],
                            trigger_time=claimed_event.get("created_at", "")[:19]
                        )
                        applied = persist_twilio_result(
                            db, case_id, claimed_event["event_id"], twilio_res,
                            extra_event={"user_response": "no_response", "user_location": geo},
                        )
                        status = applied["status"]
                else:
                    latest_event = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
                    status = (latest_event.get("sos_status") if latest_event else None) or case.get("sos_status") or TWILIO_FAILED
            else:
                claimed_event = db.sos_events.find_one_and_update(
                    {"case_id": case_id, "sos_status": "countdown"},
                    {"$set": {"sos_status": LOCAL_SIMULATION, "resolved_at": utc_now_iso()}},
                    return_document=True,
                )
                persist_local_simulation(
                    db,
                    case_id,
                    (claimed_event or event_doc or {}).get("event_id"),
                )
                status = LOCAL_SIMULATION

            remaining = 0.0

        event_latest = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
        case_latest = db.cases.find_one({"case_id": case_id}) or case
        payload = status_response_fields(case_latest, event_latest)
        payload["status"] = status
        payload["remaining_seconds"] = round(remaining, 1)
        payload["reason"] = case_latest.get("sos_trigger_reason", "")
        return payload

