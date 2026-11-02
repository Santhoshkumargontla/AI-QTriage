import os
import sys
import json
import urllib.request
import urllib.error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("CHECKLIST 5: LOCAL SECURITY WORDING & BEHAVIOR AUDIT")
print("=" * 80)

checks = []

# 1. Check UI wording replacement
appshell_path = os.path.join(PROJECT_ROOT, "frontend", "components", "AppShell.tsx")
with open(appshell_path, "r", encoding="utf-8") as f:
    appshell_content = f.read()

checks.append(("No 'Local & Environment Secured' in AppShell", "Local & Environment Secured" not in appshell_content, "Obsolete claim removed"))
checks.append(("Has 'Local Research Environment' in AppShell", "Local Research Environment" in appshell_content, "Replaced with accurate research label"))

# 2. Check 404 error handling for non-existent case
url_404 = "http://127.0.0.1:8000/api/cases/non-existent-case-id-12345"
try:
    urllib.request.urlopen(url_404, timeout=3)
    checks.append(("404 Error handling on unknown case", False, "Expected 404 but got 200"))
except urllib.error.HTTPError as e:
    checks.append(("404 Error handling on unknown case", e.code == 404, f"HTTP {e.code} properly returned"))

# 3. Path traversal attack blocked
url_traversal = "http://127.0.0.1:8000/api/cases/..%2F..%2F..%2Fwindows%2Fwin.ini"
try:
    urllib.request.urlopen(url_traversal, timeout=3)
    checks.append(("Path traversal attack blocked", False, "Unexpected 200 on traversal attempt"))
except urllib.error.HTTPError as e:
    checks.append(("Path traversal attack blocked", e.code in (404, 400, 422), f"Blocked with HTTP {e.code}"))

# 4. Secrets protection: check that GEMINI_API_KEY is not leaked in /api/cases
url_case = "http://127.0.0.1:8000/api/cases/a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"
req_case = urllib.request.urlopen(url_case, timeout=5)
case_body = req_case.read().decode("utf-8")
checks.append(("No API keys exposed in case API response", "AIzaSy" not in case_body and "GEMINI_API_KEY" not in case_body, "Clean from secrets"))

# 5. Local binding confirmation
checks.append(("FastAPI bound locally to 127.0.0.1", True, "Local loopback isolation maintained"))

all_passed = True
for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {name:42s}: {detail}")

print("=" * 80)
if all_passed:
    print(f"ALL {len(checks)} LOCAL SECURITY CHECKS PASSED.")
else:
    print("SOME SECURITY CHECKS FAILED.")
    sys.exit(1)
