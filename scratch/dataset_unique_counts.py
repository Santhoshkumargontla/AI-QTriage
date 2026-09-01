"""Dataset unique-stem counts + label quality for YOLO datasets."""
import os, glob, json
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

def unique_stems(img_dir):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    stems = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        stems.setdefault(stem, []).append(os.path.basename(f))
    return stems, files

def analyze(base, names):
    out = {"base": base, "names": names, "splits": {}}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(base, "images", split)
        lbl_dir = os.path.join(base, "labels", split)
        stems, files = unique_stems(img_dir) if os.path.exists(img_dir) else ({}, [])
        class_counts = Counter()
        boxes_per_image = []
        empty = 0
        missing = 0
        invalid = 0
        unmatched_labels = 0
        label_files = glob.glob(os.path.join(lbl_dir, "*.txt")) if os.path.exists(lbl_dir) else []
        label_stems = {os.path.splitext(os.path.basename(p))[0] for p in label_files}
        for stem in stems:
            lf = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lf):
                missing += 1
                continue
            with open(lf, encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                empty += 1
                continue
            nbox = 0
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    invalid += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    x,y,w,h = map(float, parts[1:5])
                except ValueError:
                    invalid += 1
                    continue
                if min(x,y,w,h) < 0 or x > 1 or y > 1 or w > 1 or h > 1:
                    invalid += 1
                class_counts[cid] += 1
                nbox += 1
            boxes_per_image.append(nbox)
        for ls in label_stems:
            if ls not in stems:
                unmatched_labels += 1
        named = {names.get(i, str(i)): class_counts.get(i, 0) for i in sorted(set(list(names) + list(class_counts)))}
        out["splits"][split] = {
            "image_files": len(files),
            "unique_stems": len(stems),
            "duplicate_stems": sum(1 for v in stems.values() if len(v) > 1),
            "label_files": len(label_files),
            "missing_label_for_image": missing,
            "empty_label_files_for_images": empty,
            "invalid_coord_lines": invalid,
            "unmatched_label_files": unmatched_labels,
            "total_boxes": int(sum(boxes_per_image)),
            "class_box_counts": named,
            "images_with_boxes": len(boxes_per_image),
        }
        print(f"{os.path.basename(base)}/{split}: unique_images={len(stems)} files={len(files)} labels={len(label_files)} missing={missing} empty={empty} boxes={sum(boxes_per_image)} classes={dict(named)}")
    return out

real_names = {0: "cut", 1: "bruise", 2: "wound"}
inj_names = {0: "cut", 1: "bruise", 2: "abrasion", 3: "laceration"}
results = {
    "yolo_real_wound": analyze(os.path.join(ROOT, "data", "datasets", "yolo_real_wound"), real_names),
    "yolo_injury": analyze(os.path.join(ROOT, "data", "datasets", "yolo_injury"), inj_names),
}

# public / injury dataset file counts ignoring gitignore
for d in ["public_wound_dataset", "injury_dataset", "raw"]:
    p = os.path.join(ROOT, "data", "datasets", d)
    n = 0
    exts = Counter()
    if os.path.exists(p):
        for dp, dn, fns in os.walk(p):
            n += len(fns)
            for fn in fns:
                exts[os.path.splitext(fn)[1].lower()] += 1
    print(f"data/datasets/{d}: files={n} exts={dict(exts)}")
    results[d + "_files"] = {"files": n, "exts": dict(exts)}

dest = os.path.join(ROOT, "scratch", "dataset_unique_counts.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("wrote", dest)
