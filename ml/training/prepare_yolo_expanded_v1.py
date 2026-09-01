"""Build YOLO expanded skin dataset: cut/bruise/abrasion/burn/wound/laceration.

Normal and OOD_Reject are empty-label negatives (not box classes).
Fracture X-rays are NOT merged here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "datasets" / "yolo_expanded_v1"
RAW_RF = ROOT / "data" / "raw" / "roboflow"
BURN_DIR = ROOT / "data" / "raw" / "kaggle" / "shubhambaid_skin_burn"
WSEG_ROOT = ROOT / "data" / "datasets" / "external" / "hf_wseg_dataset" / "extracted" / "wseg_dataset"
NORMAL_DIR = ROOT / "data" / "raw" / "kaggle" / "ibrahimfateen_wound_classification" / "Wound_dataset copy" / "Normal"
FORENSIC = ROOT / "data" / "uploads" / "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"
MANIFESTS = ROOT / "data" / "manifests"

NAMES = ["cut", "bruise", "abrasion", "burn", "wound", "laceration"]
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
    "burn": 3,
    "burns": 3,
    "wound": 4,
    "injury_detection": 4,
    "injury-detection": 4,
    "laceration": 5,
    "lacerations": 5,
    "stab": 5,
    "stab_wound": 5,
}
SKIP = {"scarring", "blister", "no_abnormality", "ingrown_nails", "ingrown_nail"}
WSEG_CAP = 300
NORMAL_CAP = 180
OOD_CAP = 40
SEED = 42
MIN_MASK_AREA = 0.005
MAX_MASK_AREA = 0.85


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _line_to_xywh(parts: list[str]) -> tuple[float, float, float, float] | None:
    try:
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        xc, yc, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
    else:
        return None
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    w = min(w, 2 * min(xc, 1 - xc))
    h = min(h, 2 * min(yc, 1 - yc))
    if w <= 0 or h <= 0:
        return None
    return xc, yc, w, h


def _parse_boxes(lab: Path, src_names: dict[int, str] | None = None, force_class: int | None = None) -> list[str]:
    if not lab.is_file():
        return []
    out = []
    for ln in lab.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        if force_class is not None:
            tid = force_class
        else:
            raw = src_names.get(int(float(parts[0])), "") if src_names else ""
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


def _mask_to_line(mask: np.ndarray, class_id: int = 4) -> str | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.mean())
    if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
        return None
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"


def _add_image(
    img: Path,
    lines: list[str],
    stem: str,
    kind: str,
    source: str,
    seen: set[str],
    provenance: list,
    img_counts: Counter,
    box_counts: Counter,
    source_counts: Counter,
    neg_counts: Counter | None = None,
) -> bool:
    sha = _sha(img)
    if sha in seen:
        return False
    seen.add(sha)
    split = _split_for(sha)
    dest_img = OUT / "images" / split / f"{stem}{img.suffix.lower()}"
    dest_lab = OUT / "labels" / split / f"{stem}.txt"
    shutil.copy2(img, dest_img)
    dest_lab.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    img_counts[split] += 1
    source_counts[source] += 1
    for ln in lines:
        box_counts[NAMES[int(ln.split()[0])]] += 1
    if not lines and neg_counts is not None:
        neg_counts[split] += 1
    provenance.append(
        {"sha256": sha, "source": source, "split": split, "n_boxes": len(lines), "kind": kind}
    )
    return True


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    forensic_sha = _sha(FORENSIC) if FORENSIC.is_file() else None
    if forensic_sha:
        seen.add(forensic_sha)

    box_counts: Counter = Counter()
    img_counts: Counter = Counter()
    neg_counts: Counter = Counter()
    source_counts: Counter = Counter()
    skipped: Counter = Counter()
    provenance: list = []

    # 1) Roboflow YOLO sets
    if RAW_RF.is_dir():
        for child in sorted(RAW_RF.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            yml = next(child.rglob("data.yaml"), None)
            if not yml:
                continue
            src_names = _names(yml)
            root = yml.parent
            for img in root.rglob("*"):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                if "images" not in img.as_posix().replace("\\", "/"):
                    continue
                lab = img.parent.parent / "labels" / (img.stem + ".txt")
                if not lab.is_file():
                    lab = img.with_suffix(".txt")
                # tally skips
                if lab.is_file():
                    for ln in lab.read_text(encoding="utf-8").splitlines():
                        parts = ln.split()
                        if len(parts) < 5:
                            continue
                        key = _norm_key(src_names.get(int(float(parts[0])), ""))
                        if key not in CLASS_MAP:
                            skipped[key or "unknown"] += 1
                lines = _parse_boxes(lab, src_names)
                if not lines:
                    continue
                _add_image(
                    img,
                    lines,
                    f"rf_{child.name}_{img.stem}",
                    "positive",
                    f"roboflow_{child.name}",
                    seen,
                    provenance,
                    img_counts,
                    box_counts,
                    source_counts,
                )

    # 2) shubhambaid burns (all class ids → burn)
    if BURN_DIR.is_dir():
        for img in sorted(BURN_DIR.glob("*.jpg")) + sorted(BURN_DIR.glob("*.png")):
            lab = img.with_suffix(".txt")
            lines = _parse_boxes(lab, force_class=3)
            if not lines:
                continue
            _add_image(
                img,
                lines,
                f"burn_{img.stem}",
                "positive",
                "shubhambaid_skin_burn",
                seen,
                provenance,
                img_counts,
                box_counts,
                source_counts,
            )

    # 3) WSEG masks → wound boxes
    sample_dir = WSEG_ROOT / "sample"
    mask_dir = WSEG_ROOT / "mask"
    wseg_added = 0
    if sample_dir.is_dir() and mask_dir.is_dir():
        samples = sorted(
            [
                p
                for p in sample_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        rng = np.random.RandomState(SEED)
        if len(samples) > WSEG_CAP:
            idx = rng.choice(len(samples), size=WSEG_CAP, replace=False)
            samples = [samples[i] for i in sorted(idx)]
        for img in samples:
            mask_path = mask_dir / img.name
            if not mask_path.is_file():
                mask_path = mask_dir / (img.stem + ".png")
            if not mask_path.is_file():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            line = _mask_to_line(mask, class_id=4)
            if not line:
                continue
            if _add_image(
                img,
                [line],
                f"wseg_{img.stem}",
                "positive",
                "hf_wseg",
                seen,
                provenance,
                img_counts,
                box_counts,
                source_counts,
            ):
                wseg_added += 1

    # 4) Normal empty labels
    normal_added = 0
    if NORMAL_DIR.is_dir():
        normals = sorted(
            list(NORMAL_DIR.glob("*.jpg"))
            + list(NORMAL_DIR.glob("*.jpeg"))
            + list(NORMAL_DIR.glob("*.png"))
        )[:NORMAL_CAP]
        for img in normals:
            if _add_image(
                img,
                [],
                f"normal_{img.stem}",
                "negative_normal",
                "ibrahimfateen_normal",
                seen,
                provenance,
                img_counts,
                box_counts,
                source_counts,
                neg_counts,
            ):
                normal_added += 1

    # 5) OOD hard negatives
    ood_added = 0
    ood_candidates = [
        ROOT / "data" / "datasets" / "yolo_injury" / "blank_skin.jpg",
        ROOT / "data" / "datasets" / "yolo_injury" / "dummy_test.jpg",
    ]
    # synthetic solid colors
    ood_dir = OUT / "_ood_gen"
    ood_dir.mkdir(parents=True, exist_ok=True)
    for name, color in (("black.png", (0, 0, 0)), ("white.png", (255, 255, 255)), ("gray.png", (128, 128, 128))):
        p = ood_dir / name
        if not p.exists():
            arr = np.full((256, 256, 3), color, dtype=np.uint8)
            cv2.imwrite(str(p), arr)
        ood_candidates.append(p)
    for img in ood_candidates:
        if ood_added >= OOD_CAP:
            break
        if not img.is_file():
            continue
        if _add_image(
            img,
            [],
            f"ood_{img.stem}",
            "negative_ood",
            "ood_hard_negative",
            seen,
            provenance,
            img_counts,
            box_counts,
            source_counts,
            neg_counts,
        ):
            ood_added += 1

    yaml_out = {
        "path": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(NAMES)},
        "nc": len(NAMES),
    }
    (OUT / "data.yaml").write_text(yaml.dump(yaml_out, sort_keys=False), encoding="utf-8")

    split_sha = {"train": set(), "val": set(), "test": set()}
    for p in provenance:
        split_sha[p["split"]].add(p["sha256"])
    leakage = {
        "train_val": len(split_sha["train"] & split_sha["val"]),
        "train_test": len(split_sha["train"] & split_sha["test"]),
        "val_test": len(split_sha["val"] & split_sha["test"]),
    }

    # per-split box counts for class support
    split_boxes = {n: Counter() for n in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        for lab in (OUT / "labels" / split).glob("*.txt"):
            for ln in lab.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if parts:
                    split_boxes[split][NAMES[int(parts[0])]] += 1

    class_support = {}
    for n in NAMES:
        train_n = int(split_boxes["train"][n])
        class_support[n] = {
            "status": "SUPPORTED" if train_n > 0 else "UNSUPPORTED",
            "train_boxes": train_n,
            "val_boxes": int(split_boxes["val"][n]),
            "test_boxes": int(split_boxes["test"][n]),
            "total_boxes": int(box_counts[n]),
        }

    report = {
        "created_utc": _utc(),
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "names": NAMES,
        "image_counts": dict(img_counts),
        "box_counts": dict(box_counts),
        "negative_counts": dict(neg_counts),
        "source_counts": dict(source_counts),
        "skipped_class_boxes": dict(skipped),
        "wseg_added": wseg_added,
        "normal_added": normal_added,
        "ood_added": ood_added,
        "unique_images": len(provenance),
        "forensic_excluded": bool(forensic_sha),
        "leakage_exact_hash": leakage,
        "leakage_free": all(v == 0 for v in leakage.values()),
        "class_support_honest": class_support,
        "not_box_classes": {
            "Normal": "empty-label negatives from ibrahimfateen Normal",
            "OOD_Reject": "empty-label hard negatives; inference rejects when no box",
            "Fracture": "excluded — X-ray modality; see yolo_fracture_xray_v1",
        },
        "known_limitations": [
            "Laceration mostly from Stab remaps + any Roboflow laceration if present; Yasin folder labels not used as invented boxes.",
            "Wound boxes include WSEG mask-derived bboxes (approximate).",
            "Fracture not in this head.",
        ],
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (OUT / "PREPARE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "class_support.json").write_text(json.dumps({"classes": class_support}, indent=2), encoding="utf-8")
    (MANIFESTS / "yolo_expanded_v1_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # update catalog sample count
    try:
        cat_path = ROOT / "data" / "dataset_catalog.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for ds in cat.get("datasets", []):
            if ds.get("dataset_name") == "YOLO expanded skin v1 taxonomy":
                ds["num_samples"] = len(provenance)
                ds["reason"] = (
                    f"Built by prepare_yolo_expanded_v1.py ({len(provenance)} images). "
                    "Normal/OOD empty labels. Fracture excluded."
                )
        cat_path.write_text(json.dumps(cat, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
