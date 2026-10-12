import zipfile
import xml.etree.ElementTree as ET
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

def extract_pptx(p):
    try:
        with zipfile.ZipFile(p) as z:
            slides = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            # sort slides numerically
            slides.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
            text = []
            for s in slides:
                xml = z.read(s)
                tree = ET.fromstring(xml)
                s_text = []
                for elem in tree.iter():
                    if elem.tag.endswith("}t"):
                        s_text.append(elem.text or "")
                text.append(f"--- Slide {s} ---\n" + " ".join(s_text))
            return "\n".join(text)
    except Exception as e:
        return f"Error: {e}"

pptx_files = glob.glob(r"C:\Users\santh\Downloads\*.pptx") + glob.glob(r"C:\Users\santh\Capstone Project\*.pptx")
for p in pptx_files:
    txt = extract_pptx(p)
    print(f"=== {os.path.basename(p)} === (len={len(txt)})")
    for word in ["reviewer", "comment", "page 1", "page 2", "page 3", "page 4", "page 5", "weakness"]:
        if word in txt.lower():
            print(f"  Found '{word}' in {os.path.basename(p)}")
    # Print slide titles
    for line in txt.split("\n"):
        if line.startswith("--- Slide"):
            print("  ", line)
    print("="*60)
