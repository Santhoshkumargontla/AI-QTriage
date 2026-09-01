"""Roboflow connectivity check. Uses ROBOFLOW_API_KEY from the environment only."""
import os

import requests

key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not key:
    print("ROBOFLOW_API_KEY is not set; skip.")
    raise SystemExit(0)

url = f"https://api.roboflow.com/?api_key={key}"
response = requests.get(url, timeout=30)
print("status", response.status_code)
print("ok" if response.status_code == 200 else response.text[:300])
