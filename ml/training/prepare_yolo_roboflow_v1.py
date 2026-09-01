"""Build YOLO Roboflow unified candidate dataset: cut / bruise / abrasion.

Honest mapping only. Excludes Burn/scarring/Injury-detection/Stab unless mapped.
Adds confirmed negatives. Excludes forensic hand-case hash from all splits.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "datasets" / "yolo_roboflow_v1"
RAW = ROOT / "data" / "raw" / "roboflow"
MANIFESTS = ROOT / "data" / "manifests"
NEG_SOURCES = [
    ROOT / "data" / "datasets" / "yolo_real_skin_v2" / "images" / "train",
    ROOT / "data" / "datasets" / "yolo_injury",
]

# Production-compatible primary classes + abrasion (honest boxes available)
CLASS_MAP = {
    "cut": 0,
    "cuts": 0,
    "bruise": 1,
    "bruises": 1,
    "bruising": 1,
    "abrasion": 2,
    "abrasions": 2,
    "abration": 2,
    "abrations": 2,
}
NAMES = ["cut", "bruise", "abrasion"]
SKIP = {
    "burn", "burns", "scarring", "injury-detection", "injury_detection",
    "stab", "stab_wound", "blister", "no_abnormality",
}
FORENSIC = ROOT / "data" / "uploads" / "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"
SEED = 42


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _norm_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _line_to_xywh(parts: list[str]) -> tuple[float, float, float, float] | None:
    """Accept YOLO box (5 tokens) or polygon (>=7 tokens, odd length) → normalized xywh."""
    try:
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        xc, yc, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs = vals[0::2]
        ys = vals[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
    else:
        return None
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    # clamp box inside image
    w = min(w, 2 * min(xc, 1 - xc))
    h = min(h, 2 * min(yc, 1 - yc))
    if w <= 0 or h <= 0:
        return None
    return xc, yc, w, h


def _parse_boxes(lab: Path, src_names: dict[int, str]) -> list[str]:
    if not lab.is_file():
        return []
    out = []
    for ln in lab.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        raw = src_names.get(int(float(parts[0])), "")
        key = _norm_key(raw)
        if key in SKIP or any(s in key for s in SKIP):
            continue
        if key not in CLASS_MAP:
            continue
        tid = CLASS_MAP[key]
        xywh = _line_to_xywh(parts)
        if not xywh:
            continue
        xc, yc, w, h = xywh
        out.append(f"{tid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


def _split_for(sha: str) -> str:
    b = int(sha[:8], 16) % 100
    if b < 70:
        return "train"
    if b < 85:
        return "val"
    return "test"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    forensic_sha = _sha(FORENSIC) if FORENSIC.is_file() else None
    seen: set[str] = set()
    if forensic_sha:
        seen.add(forensic_sha)

    box_counts: Counter = Counter()
    img_counts: Counter = Counter()
    neg_counts: Counter = Counter()
    source_counts: Counter = Counter()
    provenance = []
    skipped_classes: Counter = Counter()

    sources = []
    for child in sorted(RAW.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        y = next(child.rglob("data.yaml"), None)
        if y:
            sources.append((child.name, y.parent))

    for src_id, root in sources:
        src_names = _names(root / "data.yaml")
        # count skipped for report
        for lab in root.rglob("*.txt"):
            if "labels" not in lab.as_posix().replace("\\", "/"):
                continue
            for ln in lab.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                key = _norm_key(src_names.get(int(float(parts[0])), ""))
                if key in SKIP or key not in CLASS_MAP:
                    skipped_classes[key or "unknown"] += 1

        for img in root.rglob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            if "images" not in img.as_posix().replace("\\", "/"):
                continue
            sha = _sha(img)
            if sha in seen:
                continue
            seen.add(sha)
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            if not lab.is_file():
                lab = img.with_suffix(".txt")
            lines = _parse_boxes(lab, src_names)
            # Keep empties only as negatives later — skip empty injury images with no mapped boxes
            # Actually keep images that had labels but all skipped as potential hard negatives? Skip for purity.
            if not lines:
                continue
            split = _split_for(sha)
            stem = f"{src_id}_{img.stem}"
            dest_img = OUT / "images" / split / f"{stem}{img.suffix.lower()}"
            dest_lab = OUT / "labels" / split / f"{stem}.txt"
            shutil.copy2(img, dest_img)
            dest_lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
            img_counts[split] += 1
            source_counts[src_id] += 1
            for ln in lines:
                box_counts[NAMES[int(ln.split()[0])]] += 1
            provenance.append({"sha256": sha, "source": src_id, "split": split, "n_boxes": len(lines), "kind": "positive"})

    # Confirmed negatives: empty labels from yolo_real_skin_v2 empties + blank_skin/dummy
    neg_added = 0
    max_neg = 250
    for neg_root in NEG_SOURCES:
        if not neg_root.is_dir():
            continue
        candidates = list(neg_root.rglob("*")) if neg_root.name != "yolo_injury" else list(neg_root.glob("*"))
        for img in candidates:
            if neg_added >= max_neg:
                break
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # Prefer known empty-label images from real_skin: check sibling labels
            if "yolo_real_skin_v2" in img.as_posix():
                lab = Path(str(img).replace(f"{Path('images')}{img.anchor}", "").replace("images", "labels")).with_suffix(".txt")
                # simpler: train/val/test empty labels
                parts = img.parts
                try:
                    idx = parts.index("images")
                    split_name = parts[idx + 1]
                    lab = ROOT / "data" / "datasets" / "yolo_real_skin_v2" / "labels" / split_name / (img.stem + ".txt")
                except ValueError:
                    continue
                if not lab.is_file():
                    continue
                if lab.read_text(encoding="utf-8").strip():
                    continue  # only empty labels
            elif img.name not in {"blank_skin.jpg", "dummy_test.jpg"}:
                continue
            sha = _sha(img)
            if sha in seen:
                continue
            seen.add(sha)
            split = _split_for(sha)
            stem = f"neg_{img.stem}_{sha[:8]}"
            dest_img = OUT / "images" / split / f"{stem}{img.suffix.lower()}"
            dest_lab = OUT / "labels" / split / f"{stem}.txt"
            shutil.copy2(img, dest_img)
            dest_lab.write_text("", encoding="utf-8")
            img_counts[split] += 1
            neg_counts[split] += 1
            neg_added += 1
            provenance.append({"sha256": sha, "source": "confirmed_negative", "split": split, "n_boxes": 0, "kind": "negative"})

    yaml_out = {
        "path": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(NAMES)},
        "nc": len(NAMES),
    }
    (OUT / "data.yaml").write_text(yaml.dump(yaml_out, sort_keys=False), encoding="utf-8")

    # leakage: rebuild sha sets per split
    split_sha = {"train": set(), "val": set(), "test": set()}
    for p in provenance:
        split_sha[p["split"]].add(p["sha256"])
    leakage = {
        "train_val": len(split_sha["train"] & split_sha["val"]),
        "train_test": len(split_sha["train"] & split_sha["test"]),
        "val_test": len(split_sha["val"] & split_sha["test"]),
    }

    report = {
        "created_utc": _utc(),
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "names": NAMES,
        "image_counts": dict(img_counts),
        "box_counts": dict(box_counts),
        "negative_counts": dict(neg_counts),
        "source_counts": dict(source_counts),
        "skipped_class_boxes": dict(skipped_classes),
        "unique_images": len(provenance),
        "forensic_excluded": bool(forensic_sha),
        "leakage_exact_hash": leakage,
        "leakage_free": all(v == 0 for v in leakage.values()),
        "class_support_honest": {n: {"supported": box_counts[n] > 0, "total_boxes": box_counts[n]} for n in NAMES},
        "known_limitations": [
            "Burn/scarring/Injury-detection/Stab excluded — not remapped to cut/bruise/abrasion.",
            "wound class NOT included — insufficient honest wound boxes in downloaded Roboflow sets.",
            "Negatives include empty-label real_skin images + blank_skin/dummy_test.",
            "CC BY 4.0 Roboflow sources; research use only.",
        ],
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (OUT / "PREPARE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MANIFESTS / "yolo_roboflow_v1_split_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MANIFESTS / "yolo_roboflow_v1_class_distribution.json").write_text(
        json.dumps({"box_counts": dict(box_counts), "image_counts": dict(img_counts)}, indent=2), encoding="utf-8"
    )
    (MANIFESTS / "yolo_roboflow_v1_leakage.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
