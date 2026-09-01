from pathlib import Path
from collections import Counter
import json

root = Path("data/raw/kaggle")
report = {}
for ds in sorted(root.iterdir()):
    if not ds.is_dir():
        continue
    files = [p for p in ds.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    yolo_txt = [
        p
        for p in files
        if p.suffix.lower() == ".txt" and "classes" not in p.name.lower()
    ]
    class_counts = Counter()
    for img in imgs:
        parent = img.parent.name.lower()
        if parent in {"images", "image", "img", "train", "val", "test", "valid", "labels"}:
            class_counts[img.parent.parent.name] += 1
        else:
            class_counts[img.parent.name] += 1
    sample_lines = []
    for p in yolo_txt[:3]:
        try:
            sample_lines.append(p.read_text(encoding="utf-8", errors="ignore")[:160])
        except Exception:
            pass
    yaml_files = [str(p.relative_to(ds)).replace("\\", "/") for p in ds.rglob("*.yaml")][:15]
    report[ds.name] = {
        "n_files": len(files),
        "n_images": len(imgs),
        "n_yolo_txt": len(yolo_txt),
        "top_folders": class_counts.most_common(25),
        "yaml": yaml_files,
        "sample_yolo": sample_lines,
    }

out = Path("data/raw/kaggle/AUDIT_SUMMARY.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("WROTE", out)
