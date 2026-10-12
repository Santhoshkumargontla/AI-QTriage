import sys

sys.stdout.reconfigure(encoding="utf-8")
with open(r"scratch\extracted_paper.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(230, 275):
    print(f"L{i}: {lines[i]}", end="")
