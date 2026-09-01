"""Download Roboflow YOLO box datasets into data/raw/roboflow/ and verify files exist.

Loads ROBOFLOW_API_KEY from process env or backend/.env (never prints the key).
Uses yolov8 export format (widely supported); labels are YOLO txt compatible with YOLO11.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "data" / "raw" / "roboflow"
MANIFESTS = ROOT / "data" / "manifests"

DATASETS = [
    ("bryans-workspace-rrftd", "injury-detection-4vjih", "injury_detection_v2"),
    ("dongdong-d6lo7", "wound2", "wound2"),
    ("tingting-rph02", "aid-lvngz", "aid"),
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    env_path = ROOT / "backend" / ".env"
    if env_path.is_file():
        text = env_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^ROBOFLOW_API_KEY=(.+)$", text, re.M)
        if m:
            key = m.group(1).strip().strip('"').strip("'")
            if key and "YOUR_" not in key:
                os.environ["ROBOFLOW_API_KEY"] = key
                return key
    return ""


def _count_files(path: Path) -> dict:
    imgs = labs = yaml_n = 0
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            imgs += 1
        elif suf == ".txt" and "labels" in p.as_posix():
            labs += 1
        elif p.name in {"data.yaml", "dataset.yaml"}:
            yaml_n += 1
    return {"images": imgs, "label_txt": labs, "yaml": yaml_n, "total_files": sum(1 for _ in path.rglob("*") if _.is_file())}


def download_one(rf, workspace: str, project: str, dest_name: str) -> dict:
    dest = OUT / dest_name
    # wipe empty prior attempts
    if dest.exists() and _count_files(dest)["images"] == 0:
        shutil.rmtree(dest, ignore_errors=True)
    if dest.exists() and _count_files(dest)["images"] > 0:
        return {"status": "ALREADY_PRESENT", "path": str(dest.relative_to(ROOT)).replace("\\", "/"), **_count_files(dest)}

    proj = rf.workspace(workspace).project(project)
    versions = proj.versions()
    if not versions:
        return {"status": "FAILED", "reason": "no versions", "workspace": workspace, "project": project}
    ver = versions[0].version
    dest.mkdir(parents=True, exist_ok=True)
    # Prefer yolov8 format — Roboflow export is reliable and YOLO11-compatible
    try:
        ds = proj.version(ver).download("yolov8", location=str(dest))
        loc = Path(getattr(ds, "location", dest))
    except Exception as exc:
        return {"status": "FAILED", "error": str(exc), "workspace": workspace, "project": project, "version": ver}

    counts = _count_files(dest)
    if counts["images"] == 0 and loc.exists() and loc != dest:
        # library may have nested the download
        nested = _count_files(loc)
        if nested["images"] > 0:
            # move contents up if needed
            for child in loc.iterdir():
                target = dest / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
            counts = _count_files(dest)

    if counts["images"] == 0:
        # search one level for data.yaml parent
        for y in dest.rglob("data.yaml"):
            counts = _count_files(y.parent)
            if counts["images"] > 0:
                return {
                    "status": "DOWNLOADED",
                    "workspace": workspace,
                    "project": project,
                    "version": ver,
                    "path": str(y.parent.relative_to(ROOT)).replace("\\", "/"),
                    **counts,
                }
        return {"status": "FAILED", "reason": "download returned empty directory", "version": ver, "dest": str(dest)}

    return {
        "status": "DOWNLOADED",
        "workspace": workspace,
        "project": project,
        "version": ver,
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        **counts,
    }


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    key = _load_key()
    if not key:
        report = {"created_utc": _utc(), "status": "BLOCKED_FOR_YOLO_DATA_ACQUISITION", "reason": "ROBOFLOW_API_KEY missing"}
        (MANIFESTS / "roboflow_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    results = {}
    for ws, proj, name in DATASETS:
        print(f"Downloading {name}...", flush=True)
        results[name] = download_one(rf, ws, proj, name)
        print(json.dumps({name: {k: results[name].get(k) for k in ("status", "images", "label_txt", "yaml", "version", "reason", "error") if k in results[name] or results[name].get("status")}}, indent=2), flush=True)

    report = {"created_utc": _utc(), "status": "OK", "downloads": results}
    (MANIFESTS / "roboflow_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
