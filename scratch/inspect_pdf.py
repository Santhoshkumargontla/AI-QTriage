import re
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

def extract_strings_from_pdf(p):
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        data = f.read()
    # Find text between BT and ET, or plain text streams
    strings = []
    # simple extraction of ascii strings of length >= 4
    for match in re.finditer(b"[A-Za-z0-9 ,.:;?!()'\"]{6,}", data):
        s = match.group().decode("latin1", errors="ignore")
        if any(w in s.lower() for w in ["reviewer", "review", "comment", "yolo", "quantum", "xgboost", "dice", "slide"]):
            strings.append(s)
    return "\n".join(strings)

pdfs = [
    r"C:\Users\santh\Downloads\Final Paper Submission Report.pdf",
    r"C:\Users\santh\Downloads\Capstone_PDF_Matter_PPT_11_Slides.pdf",
    r"C:\Users\santh\Downloads\previewFormOrdinaryDetail.pdf",
    r"C:\Users\santh\Downloads\pad report .pdf",
    r"C:\Users\santh\Capstone Project\Final Report Paper Submission.pdf"
]

for pdf in pdfs:
    print(f"=== {os.path.basename(pdf)} ===")
    out = extract_strings_from_pdf(pdf)
    print(out[:1500])
    print("="*60)
