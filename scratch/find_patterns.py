import os
import glob
import sys

sys.stdout.reconfigure(encoding="utf-8")

patterns = ["PAGE 1", "PAGE 2", "PAGE 4", "PAGE 6", "PAGE 7", "PAGE 8", "Remaining scientific weakness", "Needs improvement", "Reviewer 1", "Reviewer 2"]

def check_file(p):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for pat in patterns:
            if pat.lower() in txt.lower():
                print(f"Match '{pat}' in: {p}")
                idx = txt.lower().find(pat.lower())
                print(txt[max(0, idx-100):min(len(txt), idx+300)].replace("\n", " "))
                print("="*60)
                break
    except:
        pass

for root in [r"C:\Users\santh\Capstone Project Code", r"C:\Users\santh\Capstone Project"]:
    for dirpath, dirnames, filenames in os.walk(root):
        if any(x in dirpath for x in ["venv", "node_modules", ".git", ".next"]):
            continue
        for f in filenames:
            if f.endswith((".md", ".txt", ".json", ".py", ".html")):
                check_file(os.path.join(dirpath, f))

# Also check git log!
print("Done checking files.")
