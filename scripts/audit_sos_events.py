"""
SOS Event MongoDB Audit Script
Inspect all sos_events for a given case and check the latest Twilio message status.
"""
import os
import sys
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from backend.database.connection import get_database
from backend.services.twilio_service import twilio_service

def audit_sos_events(case_id: str = None):
    db = get_database()

    query = {}
    if case_id:
        query["case_id"] = case_id

    events = list(db.sos_events.find(query).sort("created_at", -1).limit(20))

    print(f"\n{'='*70}")
    print(f"  SOS EVENTS AUDIT  (latest 20 events{f' for case {case_id}' if case_id else ''})")
    print(f"{'='*70}")

    if not events:
        print("  No SOS events found.")
        return

    for i, ev in enumerate(events):
        ev.pop("_id", None)
        print(f"\n--- Event #{i+1} ---")
        fields = [
            "event_id", "case_id", "sos_status", "delivery_mode",
            "twilio_message_sid", "delivery_status",
            "created_at", "sending_started_at", "resolved_at",
            "twilio_send_timestamp", "twilio_provider",
            "twilio_error_code", "twilio_error_message"
        ]
        for f in fields:
            val = ev.get(f)
            if val is not None:
                print(f"  {f}: {val}")

        # Live Twilio status poll
        msg_sid = ev.get("twilio_message_sid")
        if msg_sid and twilio_service.account_sid and twilio_service.auth_token:
            print(f"\n  [Polling Twilio for SID {msg_sid[:8]}...]")
            try:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_service.account_sid}/Messages/{msg_sid}.json"
                r = requests.get(
                    url,
                    auth=(twilio_service.account_sid, twilio_service.auth_token),
                    timeout=5
                )
                if r.status_code == 200:
                    resp = r.json()
                    print(f"  twilio.status:         {resp.get('status')}")
                    print(f"  twilio.date_created:   {resp.get('date_created')}")
                    print(f"  twilio.date_sent:      {resp.get('date_sent')}")
                    print(f"  twilio.date_updated:   {resp.get('date_updated')}")
                    print(f"  twilio.error_code:     {resp.get('error_code')}")
                    print(f"  twilio.error_message:  {resp.get('error_message')}")
                    body = resp.get("body", "")
                    print(f"  twilio.body (first 200 chars):")
                    print(f"    {body[:200]}")
                else:
                    print(f"  Twilio API responded with HTTP {r.status_code}")
            except Exception as e:
                print(f"  Error polling Twilio: {e}")

    print(f"\n{'='*70}")
    print("  CASE DOCUMENT SOS FIELDS")
    print(f"{'='*70}")
    if case_id:
        case = db.cases.find_one({"case_id": case_id})
    else:
        # Get the case for the most recent event
        case = db.cases.find_one({"case_id": events[0]["case_id"]})
    if case:
        sos_fields = {k: v for k, v in case.items() if "sos" in k.lower() or k == "case_id"}
        for k, v in sos_fields.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else None
    audit_sos_events(cid)
