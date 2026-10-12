import os
import glob
import sys

sys.stdout.reconfigure(encoding="utf-8")

search_dirs = [
    r"C:\Users\santh\Capstone Project",
    r"C:\Users\santh\Downloads",
    r"C:\Users\santh\Documents",
    r"C:\Users\santh\Desktop",
    r"C:\Users\santh\others"
]

target_str = "PAGE 3 — Dataset"
alt_str = "PAGE 3"

for sdir in search_dirs:
    if not os.path.exists(sdir):
        continue
    for root, dirs, files in os.walk(sdir):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(('.txt', '.md', '.docx', '.json', '.pdf')):
                try:
                    with open(p, 'rb') as handle:
                        raw = handle.read()
                    if b"PAGE 3" in raw or b"PAGE 5" in raw or b"Dataset and preprocessing" in raw:
                        print(f"MATCH in {p} (size={len(raw)})")
                except Exception as e:
                    pass
