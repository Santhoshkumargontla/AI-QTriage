import zlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

def parse_pdf(path):
    with open(path, "rb") as f:
        data = f.read()
    
    stream_re = re.compile(b"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
    texts = []
    for m in stream_re.finditer(data):
        raw = m.group(1)
        decomp = None
        try:
            decomp = zlib.decompress(raw)
        except:
            decomp = raw
        
        # extract text inside parentheses ( ... ) Tj or [ ... ] TJ
        tj_re = re.compile(b"\((.*?)\)")
        for t in tj_re.finditer(decomp):
            txt = t.group(1).decode("latin1", errors="ignore")
            if len(txt.strip()) > 1:
                texts.append(txt)
    return " ".join(texts)

res = parse_pdf(r"C:\Users\santh\Downloads\Capstone_PDF_Matter_PPT_11_Slides.pdf")
print("Extracted from Capstone_PDF_Matter_PPT_11_Slides.pdf (len=" + str(len(res)) + "):")
print(res[:3000])
