import os
import sys
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.twilio_service import twilio_service

def check_status():
    twilio_service.reload_config()
    msg_sid = "SM9a84e13dcf41fa97e8b668996ba56e56"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_service.account_sid}/Messages/{msg_sid}.json"
    
    r = requests.get(
        url,
        auth=(twilio_service.account_sid, twilio_service.auth_token)
    )
    print("Status code:", r.status_code)
    print("Response JSON:")
    print(r.json())

if __name__ == "__main__":
    check_status()
