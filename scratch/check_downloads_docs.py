import zipfile
import xml.etree.ElementTree as ET
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

def extract_docx(p):
    try:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml")
        tree = ET.fromstring(xml)
        lines = []
        for elem in tree.iter():
            if elem.tag.endswith("}t"):
                lines.append(elem.text or "")
            elif elem.tag.endswith("}p"):
                lines.append("\n")
        return "".join(lines)
    except Exception as e:
        return ""

for fname in ["pad 1.docx", "Final PD.docx", "FINAL.docx", "college.docx", "FIE REPORT.docx"]:
    p = os.path.join(r"C:\Users\santh\Downloads", fname)
    if os.path.exists(p):
        txt = extract_docx(p)
        print(f"=== {fname} === (len={len(txt)})")
        for line in txt.split("\n"):
            if any(k in line.lower() for k in ["page 1", "page 2", "page 3", "page 4", "page 5", "page 6", "page 7", "page 8", "reviewer", "weakness"]):
                print("  ", line.strip()[:120])
        print("="*60)
