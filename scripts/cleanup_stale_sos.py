"""
Cleanup stale SOS countdown events in MongoDB.
Marks any sos_events with status="countdown" that are older than grace period as "expired_stale".
Run this once to clean up historical test events.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from backend.database.connection import get_database

def cleanup_stale_events():
    db = get_database()
    now = datetime.utcnow()
    grace_seconds = 60.0  # mark as stale if countdown + 60s have elapsed

    stale_events = list(db.sos_events.find({"sos_status": "countdown"}))
    updated = 0
    kept = 0

    for ev in stale_events:
        created_str = ev.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_str) if created_str else None
            countdown_secs = float(ev.get("countdown_seconds", 10))
            threshold = countdown_secs + grace_seconds
            if created_at and (now - created_at).total_seconds() > threshold:
                db.sos_events.update_one(
                    {"event_id": ev["event_id"]},
                    {"$set": {
                        "sos_status": "expired_stale",
                        "expired_at": now.isoformat()
                    }}
                )
                print(f"  [EXPIRED] event_id={ev['event_id']}  case_id={ev['case_id']}  created={created_str[:19]}")
                updated += 1
            else:
                print(f"  [KEPT ACTIVE] event_id={ev['event_id']}  case_id={ev['case_id']}  created={created_str[:19]}")
                kept += 1
        except Exception as e:
            print(f"  [ERROR] event_id={ev.get('event_id')} — {e}")

    print(f"\nDone. Expired: {updated} events. Kept active: {kept} events.")

if __name__ == "__main__":
    print("Cleaning up stale SOS countdown events...")
    cleanup_stale_events()
