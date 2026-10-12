import os
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

for root, dirs, files in os.walk("audit"):
    for f in files:
        if f.endswith((".json", ".md")):
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                for kw in ["reviewer", "comment 1", "comment 2", "review 1", "review 2"]:
                    if kw in content.lower():
                        print(f"Match '{kw}' in: {p}")
                        idx = content.lower().find(kw)
                        print("  ", content[max(0, idx-50):min(len(content), idx+200)].replace("\n", " "))
                        print("="*60)
                        break
            except:
                pass
