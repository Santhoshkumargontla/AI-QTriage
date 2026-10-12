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

def search_in_pdf(p):
    try:
        # crude text extraction from uncompressed streams in pdf
        with open(p, "rb") as f:
            content = f.read().decode("latin1", errors="ignore")
        return content
    except Exception as e:
        return ""

candidates = glob.glob(r"C:\Users\santh\Capstone Project\*.docx") + \
             glob.glob(r"C:\Users\santh\Downloads\*.docx")

print("Checking docx files:")
for c in candidates:
    txt = extract_docx(c)
    print(f"Docx: {os.path.basename(c)}, len={len(txt)}")
    for word in ["reviewer 1", "reviewer 2", "review comment", "11 comment", "comment 1", "comment 2", "weakness"]:
        if word in txt.lower():
            print(f"  --> Found '{word}' in {os.path.basename(c)}")
            idx = txt.lower().find(word)
            print("     Snippet:", txt[max(0, idx-50):min(len(txt), idx+200)].replace("\n", " "))

