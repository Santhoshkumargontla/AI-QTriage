import urllib.request
import json

url = "http://127.0.0.1:8000/api/cases/bd890e11-530d-4d0e-8c4f-4bda296a8600"
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
