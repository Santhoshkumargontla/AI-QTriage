import urllib.request
import json
import sys

case_id = sys.argv[1] if len(sys.argv) > 1 else "e684a322-bb77-4a00-8431-4903cd07612e"
url = f"http://127.0.0.1:8000/api/cases/{case_id}"
res = urllib.request.urlopen(url)
data = json.loads(res.read().decode("utf-8"))
shaps = data.get("shap_explanations", [])

print(f"Total features in shap_explanations: {len(shaps)}")
print("-" * 50)
for i, s in enumerate(shaps):
    feat = s.get("feature", "unknown")
    val = float(s.get("shap_value", 0.0))
    print(f"{i+1:2d}. {feat:<30} {val:+.6f}")

print("-" * 50)
top6 = sum(float(s["shap_value"]) for s in shaps[:6])
total = sum(float(s["shap_value"]) for s in shaps)
tail17 = sum(float(s["shap_value"]) for s in shaps[6:])

print(f"Top 6 sum       : {top6:+.6f}")
print(f"Remaining 17 sum: {tail17:+.6f}")
print(f"Total (all 23)  : {total:+.6f}")
print(f"Base E[f(x)]    : -0.082254")
print(f"Margin z_HIGH   : {-0.082254 + total:+.6f}")
