"""
Twilio Integration Service for AI-QTriage Research Prototype.
Provides safe test/sandbox SOS alert notifications.
Strictly non-emergency research prototype simulation.
DO NOT use for contacting real emergency services or 911/112.
"""

import os
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from backend.config import settings
from backend.services.sos_delivery import (
    TWILIO_FAILED,
    TWILIO_NOT_CONFIGURED,
    TWILIO_REQUEST_QUEUED,
    utc_now_iso,
)

from dotenv import load_dotenv

# Ensure backend/.env is loaded into os.environ
env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_file_path)

def format_compact_sms_timestamp(utc_iso_or_dt=None, tz_name: str = "Asia/Kolkata") -> str:
    """
    Formats a UTC timestamp or datetime object into a concise display string
    e.g. '15 Aug 2026 15:55 IST'.
    """
    from zoneinfo import ZoneInfo
    if utc_iso_or_dt is None:
        dt_utc = datetime.now(timezone.utc)
    elif isinstance(utc_iso_or_dt, datetime):
        dt_utc = utc_iso_or_dt if utc_iso_or_dt.tzinfo is not None else utc_iso_or_dt.replace(tzinfo=timezone.utc)
    elif isinstance(utc_iso_or_dt, str):
        try:
            dt_parse = datetime.fromisoformat(utc_iso_or_dt.replace("Z", "+00:00"))
            dt_utc = dt_parse if dt_parse.tzinfo is not None else dt_parse.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            dt_utc = datetime.now(timezone.utc)
    else:
        dt_utc = datetime.now(timezone.utc)

    try:
        local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
        tz_abbr = "IST" if tz_name == "Asia/Kolkata" else local_dt.strftime("%Z")
        return local_dt.strftime(f"%d %b %Y %H:%M {tz_abbr}")
    except (ValueError, OSError, KeyError):
        return dt_utc.strftime("%d %b %Y %H:%M UTC")

format_sms_timestamp = format_compact_sms_timestamp


def format_gps_location_line(
    latitude: float | None = None,
    longitude: float | None = None,
    maps_url: str | None = None,
    fallback: str | None = None,
) -> str:
    """Prefer GPS coordinates (+ short maps URL) over injury body-part text."""
    try:
        if latitude is not None and longitude is not None:
            lat = float(latitude)
            lng = float(longitude)
            if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
                coord = f"{lat:.5f},{lng:.5f}"
                url = (maps_url or f"https://maps.google.com/?q={coord}").strip()
                url = "".join(c for c in url if ord(c) < 128)
                # Keep maps URL short enough for trial SMS limits.
                if len(url) > 48:
                    url = f"https://maps.google.com/?q={coord}"
                return f"GPS {coord} {url}"
    except (TypeError, ValueError):
        pass
    fb = str(fallback or "").strip() or "GPS unavailable"
    fb = "".join(c for c in fb if ord(c) < 128)
    if fb.lower() in ("unspecified", "not_provided", "unknown", "unknown location"):
        return "GPS unavailable"
    return f"Site:{fb}"


def build_compact_sos_message(
    case_id: str,
    sos_event_id: str,
    trigger_time: str = None,
    yolo_finding: str = None,
    classifier_category: str = None,
    user_location: str = None,
    latitude: float = None,
    longitude: float = None,
    maps_url: str = None,
    max_length: int = 320,
) -> str:
    """
    Builds a concise GSM-7/ASCII safe SMS body for Twilio trial/test SOS.
    Includes user GPS when available so responders can open a maps link.
    """
    short_case = (case_id[-8:] if case_id else "unknown")
    short_event = (sos_event_id[-8:] if sos_event_id else "unknown")
    tz_name = settings.sos_display_timezone
    time_str = format_compact_sms_timestamp(trigger_time, tz_name)

    if yolo_finding and str(yolo_finding).strip() and str(yolo_finding).strip() != "None":
        finding_str = str(yolo_finding).strip()
    elif classifier_category and str(classifier_category).strip() and str(classifier_category).strip() != "None":
        finding_str = str(classifier_category).strip()
    else:
        finding_str = "None detected"

    finding_str = "".join(c for c in finding_str if ord(c) < 128)
    if len(finding_str) > 18:
        finding_str = finding_str[:15] + "..."

    loc_str = format_gps_location_line(
        latitude=latitude,
        longitude=longitude,
        maps_url=maps_url,
        fallback=user_location,
    )
    if len(loc_str) > 72:
        loc_str = loc_str[:69] + "..."

    lines = [
        "AI-QTriage DEMO SOS",
        f"Case: {short_case}",
        f"Event: {short_event}",
        f"Time: {time_str}",
        f"Finding: {finding_str}",
        f"Location: {loc_str}",
        "SIMULATED TEST ONLY.",
        "NO EMERGENCY SERVICES CONTACTED.",
    ]
    message_body = "\n".join(lines)

    # Prefer keeping GPS/location over long finding text when length-capped.
    if len(message_body) > max_length:
        lines = [
            "AI-QTriage DEMO SOS",
            f"Case: {short_case} Ev:{short_event}",
            f"Time: {time_str}",
            f"Find: {finding_str[:12]}",
            f"Loc: {loc_str[:60]}",
            "TEST ONLY. No emergency svc.",
        ]
        message_body = "\n".join(lines)

    if len(message_body) > max_length:
        message_body = message_body[: max_length - 3] + "..."

    return message_body


CANONICAL_ENV = (
    "TWILIO_ENABLED",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM_NUMBER",
    "TWILIO_TO_NUMBER",
)

# Documented legacy aliases only (forensic report B08). Canonical name wins.
LEGACY_ALIASES = {
    "TWILIO_PHONE_NUMBER": "TWILIO_FROM_NUMBER",
    "EMERGENCY_CONTACT_PHONE": "TWILIO_TO_NUMBER",
    "EMERGENCY_PHONE_NUMBER": "TWILIO_TO_NUMBER",
}


def _env_first(*names: str) -> str:
    for name in names:
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


class TwilioService:
    def __init__(self):
        self.reload_config()

    def reload_config(self):
        load_dotenv(env_file_path, override=True)
        self.enabled = os.environ.get("TWILIO_ENABLED", "false").lower() in ("true", "1", "yes")
        self.account_sid = _env_first("TWILIO_ACCOUNT_SID")
        self.auth_token = _env_first("TWILIO_AUTH_TOKEN")
        self.from_number = _env_first("TWILIO_FROM_NUMBER", "TWILIO_PHONE_NUMBER")
        self.to_number = _env_first("TWILIO_TO_NUMBER", "EMERGENCY_CONTACT_PHONE", "EMERGENCY_PHONE_NUMBER")

    def missing_fields(self) -> list:
        missing = []
        if not self.account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not self.auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not self.from_number:
            missing.append("TWILIO_FROM_NUMBER")
        if not self.to_number:
            missing.append("TWILIO_TO_NUMBER")
        return missing

    def is_configured(self) -> Tuple[bool, str]:
        """
        Checks if Twilio test integration is configured and enabled.
        Returns (is_configured: bool, status_message: str)
        NEVER returns auth_token, phone numbers, or other secrets.
        """
        if not getattr(self, "_manual_mock", False):
            self.reload_config()
        if not self.enabled:
            return False, "Twilio integration disabled (TWILIO_ENABLED=false)."
        missing = self.missing_fields()
        if missing:
            return False, "Twilio is enabled but missing: " + ", ".join(missing) + "."
        return True, "Twilio configured for test alerts only. No real emergency services will be contacted."

    def get_status_info(self) -> Dict[str, Any]:
        """Returns safe status metadata for frontend and report generation."""
        configured, msg = self.is_configured()
        sid = self.account_sid
        return {
            "enabled": self.enabled,
            "configured": configured,
            "status_message": msg,
            "canonical_env": list(CANONICAL_ENV),
            "legacy_aliases": dict(LEGACY_ALIASES),
            "missing_fields": [] if configured else (["TWILIO_ENABLED"] if not self.enabled else self.missing_fields()),
            "delivery_outcomes": [
                "LOCAL_SIMULATION",
                "TWILIO_NOT_CONFIGURED",
                "TWILIO_REQUEST_QUEUED",
                "TWILIO_FAILED",
            ],
            "account_sid_suffix": f"...{sid[-4:]}" if len(sid) >= 4 else None,
            "from_number_set": bool(self.from_number),
            "to_number_set": bool(self.to_number),
            "real_sms_tested": False,
            "disclaimer": "RESEARCH PROTOTYPE ONLY: No real emergency services (911/112) are contacted. TWILIO_REQUEST_QUEUED means the API returned a SID, not that a handset received SMS.",
        }

    def send_test_sos_message(
        self,
        case_id: str,
        user_location: str = "Unknown location",
        is_whatsapp: bool = False,
        sos_event_id: str = None,
        trigger_time: str = None,
        yolo_finding: str = None,
        classifier_category: str = None,
        finding_name: str = None,
        latitude: float = None,
        longitude: float = None,
        maps_url: str = None,
    ) -> Dict[str, Any]:
        """
        POST to Twilio Messages.json. Success requires HTTP 200/201 AND a message SID.
        Application status is TWILIO_REQUEST_QUEUED, never SMS_SENT.
        """
        del is_whatsapp  # SMS path only; WhatsApp is not a documented SOS transport.
        configured, msg = self.is_configured()
        ts = utc_now_iso()
        if not configured:
            return {
                "success": False,
                "status": TWILIO_NOT_CONFIGURED,
                "delivery_outcome": TWILIO_NOT_CONFIGURED,
                "configured": False,
                "message": msg,
                "failure_reason": msg,
                "twilio_message_sid": None,
                "provider_status": None,
                "timestamp": ts,
                "disclaimer": "No real emergency services were contacted. No SMS was sent.",
            }

        safe_body = build_compact_sos_message(
            case_id=case_id,
            sos_event_id=sos_event_id,
            trigger_time=trigger_time,
            yolo_finding=yolo_finding or finding_name,
            classifier_category=classifier_category,
            user_location=user_location,
            latitude=latitude,
            longitude=longitude,
            maps_url=maps_url,
            max_length=320,
        )

        from_num = self.from_number
        to_num = self.to_number
        if not from_num or not to_num:
            reason = "Twilio is enabled but TWILIO_FROM_NUMBER or TWILIO_TO_NUMBER is missing."
            return {
                "success": False,
                "status": TWILIO_NOT_CONFIGURED,
                "delivery_outcome": TWILIO_NOT_CONFIGURED,
                "configured": False,
                "message": reason,
                "failure_reason": reason,
                "twilio_message_sid": None,
                "provider_status": None,
                "timestamp": ts,
            }

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {
            "From": from_num,
            "To": to_num,
            "Body": safe_body
        }

        try:
            r = requests.post(
                url,
                data=data,
                auth=(self.account_sid, self.auth_token),
                timeout=10
            )

            if r.status_code in (200, 201):
                try:
                    resp_json = r.json()
                except ValueError:
                    resp_json = {}
                msg_sid = (resp_json.get("sid") or "").strip() or None
                provider_status = resp_json.get("status")
                if not msg_sid:
                    reason = "Twilio API returned HTTP success without a message SID. SMS_SENT was not recorded."
                    return {
                        "success": False,
                        "status": TWILIO_FAILED,
                        "delivery_outcome": TWILIO_FAILED,
                        "error_code": "MISSING_SID",
                        "message": reason,
                        "failure_reason": reason,
                        "body_length": len(safe_body),
                        "twilio_message_sid": None,
                        "provider_status": provider_status,
                        "timestamp": ts,
                    }
                return {
                    "success": True,
                    "status": TWILIO_REQUEST_QUEUED,
                    "delivery_outcome": TWILIO_REQUEST_QUEUED,
                    "twilio_message_sid": msg_sid,
                    "provider_status": provider_status or "queued",
                    "delivery_status": provider_status or "queued",
                    "timestamp": ts,
                    "failure_reason": None,
                    "body_length": len(safe_body),
                    "message": "Twilio API accepted the request and returned a message SID. This is not proof of handset delivery.",
                    "disclaimer": "RESEARCH PROTOTYPE — Twilio test request only. NO real emergency services were contacted.",
                }

            try:
                err_json = r.json()
                err_code = err_json.get("code", r.status_code)
                err_detail = err_json.get("message", r.text)
            except ValueError:
                err_code = r.status_code
                err_detail = (r.text or "")[:200]

            if str(err_code) == "30044":
                err_detail = "Trial SMS message exceeded Twilio's allowed message length."

            return {
                "success": False,
                "status": TWILIO_FAILED,
                "delivery_outcome": TWILIO_FAILED,
                "error_code": str(err_code),
                "message": err_detail,
                "failure_reason": err_detail,
                "body_length": len(safe_body),
                "twilio_message_sid": None,
                "provider_status": None,
                "timestamp": ts,
            }
        except requests.RequestException as e:
            reason = f"Failed to connect to Twilio API: {str(e)[:150]}"
            return {
                "success": False,
                "status": TWILIO_FAILED,
                "delivery_outcome": TWILIO_FAILED,
                "message": reason,
                "failure_reason": reason,
                "twilio_message_sid": None,
                "provider_status": None,
                "timestamp": ts,
                "disclaimer": "NO real emergency services were contacted.",
            }

twilio_service = TwilioService()
