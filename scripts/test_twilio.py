import os
import sys
import time
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.twilio_service import twilio_service

def test_direct_twilio():
    print("====================================================")
    print("TWILIO DIRECT DIAGNOSTIC TEST")
    print("====================================================")
    
    # Load config
    twilio_service.reload_config()
    
    print(f"Twilio Enabled: {twilio_service.enabled}")
    print(f"Account SID: {twilio_service.account_sid[:6]}...{twilio_service.account_sid[-4:] if len(twilio_service.account_sid) >= 4 else ''}")
    print(f"From Number: {twilio_service.from_number}")
    print(f"To Number: {twilio_service.to_number}")
    
    configured, msg = twilio_service.is_configured()
    print(f"Configured: {configured} ({msg})")
    
    if not configured:
        print("Twilio is not configured properly. Exiting.")
        return
        
    print("\nSending test message...")
    res = twilio_service.send_test_sos_message(
        case_id="direct_diag_test_123",
        finding_name="Cut (Forensic Test)",
        user_location="Wrist"
    )
    
    print("\nTwilio Response:")
    for k, v in res.items():
        if k == "auth_token" or k == "password":
            continue
        print(f"  {k}: {v}")
        
    msg_sid = res.get("twilio_message_sid")
    if msg_sid:
        print("\nWaiting 5 seconds to query delivery status...")
        time.sleep(5)
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_service.account_sid}/Messages/{msg_sid}.json"
        try:
            r = requests.get(
                url,
                auth=(twilio_service.account_sid, twilio_service.auth_token),
                timeout=5
            )
            if r.status_code == 200:
                resp = r.json()
                print("Polled Twilio Message Info:")
                print(f"  SID: {resp.get('sid')}")
                print(f"  Status: {resp.get('status')}")
                print(f"  Error Code: {resp.get('error_code')}")
                print(f"  Error Message: {resp.get('error_message')}")
            else:
                print(f"Failed to query status, API responded with status {r.status_code}")
        except Exception as e:
            print(f"Error querying Twilio status: {e}")

if __name__ == "__main__":
    test_direct_twilio()
