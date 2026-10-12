import sys

sys.stdout.reconfigure(encoding="utf-8")

with open(r"scratch\extracted_paper.md", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of paper:", len(text))
print("First 2000 chars:")
print(text[:2000])

print("\n" + "="*50 + "\nHeadings and Tables:")
lines = text.split("\n")
for idx, l in enumerate(lines):
    s = l.strip()
    if s.startswith(("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "TABLE", "Table", "Fig", "FIG", "Abstract", "REFERENCES", "References")):
        print(f"L{idx}: {s[:100]}")
