"""Phase 2–3: acquire public datasets into data/raw/ without overwriting existing trees."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
RAW.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_download(url: str, dest: Path, timeout: int = 300) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        return {"status": "ALREADY_PRESENT", "path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size}
    try:
        req = Request(url, headers={"User-Agent": "AI-QTriage-research/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return {"status": "DOWNLOADED", "path": str(dest.relative_to(ROOT)), "bytes": len(data), "url": url}
    except Exception as exc:
        return {"status": "FAILED", "url": url, "error": f"{type(exc).__name__}: {exc}"}


def _try_kaggle(slug: str, dest_name: str) -> dict:
    dest = RAW / "kaggle" / dest_name
    if dest.exists() and any(dest.iterdir()):
        return {"status": "ALREADY_PRESENT", "path": str(dest.relative_to(ROOT))}
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return {"status": "BLOCKED_NO_CREDENTIALS", "slug": slug, "note": "kaggle.json missing"}
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "--unzip",
        "-p",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        if proc.returncode != 0:
            return {"status": "FAILED", "slug": slug, "stderr": proc.stderr[-500:]}
        return {"status": "DOWNLOADED", "slug": slug, "path": str(dest.relative_to(ROOT))}
    except Exception as exc:
        return {"status": "FAILED", "slug": slug, "error": str(exc)}


def _try_roboflow(workspace: str, project: str, dest_name: str) -> dict:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        return {"status": "BLOCKED_NO_API_KEY", "workspace": workspace, "project": project}
    dest = RAW / "roboflow" / dest_name
    if dest.exists() and any(dest.rglob("data.yaml")):
        return {"status": "ALREADY_PRESENT", "path": str(dest.relative_to(ROOT))}
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=key)
        proj = rf.workspace(workspace).project(project)
        versions = proj.versions()
        if not versions:
            return {"status": "FAILED", "reason": "no versions"}
        ver = versions[0].version
        dest.mkdir(parents=True, exist_ok=True)
        ds = proj.version(ver).download("yolov11", location=str(dest))
        return {
            "status": "DOWNLOADED",
            "workspace": workspace,
            "project": project,
            "version": ver,
            "path": str(Path(ds.location).relative_to(ROOT)) if hasattr(ds, "location") else str(dest.relative_to(ROOT)),
        }
    except Exception as exc:
        return {"status": "FAILED", "workspace": workspace, "project": project, "error": str(exc)}


ROBOFLOW_PRIORITY = [
    ("bryans-workspace-rrftd", "injury-detection-4vjih", "injury_detection_v2"),
    ("dongdong-d6lo7", "wound2", "wound2"),
    ("raghav-bharathi-vywgh", "wound-detection-and-segmentation", "wound_detection_segmentation"),
    ("tingting-rph02", "aid-lvngz", "aid"),
]

KAGGLE_PRIORITY = [
    ("yasinpratomo/wound-dataset", "yasinpratomo_wound_dataset"),
    ("leoscode/wound-segmentation-images", "leoscode_wound_segmentation_2760"),
    ("uciml/human-activity-recognition-with-smartphones", "uci_har_smartphones"),
]


def main() -> None:
    report = {"created_utc": _utc(), "downloads": {}}

    # UCI HAR direct zip (also on Kaggle mirror)
    har_zip = RAW / "uci" / "har_smartphones.zip"
    report["downloads"]["uci_har"] = _http_download(
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        har_zip,
    )
    if report["downloads"]["uci_har"]["status"] in ("DOWNLOADED", "ALREADY_PRESENT"):
        extract_dir = RAW / "uci" / "har_smartphones"
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(har_zip, "r") as zf:
                zf.extractall(extract_dir)
            report["downloads"]["uci_har"]["extracted_to"] = str(extract_dir.relative_to(ROOT))
        inner = extract_dir / "UCI HAR Dataset.zip"
        inner_dir = extract_dir / "UCI HAR Dataset"
        if inner.exists() and not inner_dir.exists():
            with zipfile.ZipFile(inner, "r") as zf:
                zf.extractall(extract_dir)
            report["downloads"]["uci_har"]["inner_extracted_to"] = str(inner_dir.relative_to(ROOT))

    report["downloads"]["roboflow"] = {}
    for ws, proj, name in ROBOFLOW_PRIORITY:
        report["downloads"]["roboflow"][name] = _try_roboflow(ws, proj, name)

    report["downloads"]["kaggle"] = {}
    for slug, folder in KAGGLE_PRIORITY:
        if (RAW / "kaggle" / folder).exists():
            report["downloads"]["kaggle"][folder] = {"status": "ALREADY_PRESENT", "path": f"data/raw/kaggle/{folder}"}
        else:
            report["downloads"]["kaggle"][folder] = _try_kaggle(slug, folder)

    provenance = {
        "created_utc": _utc(),
        "sources": [],
    }
    for group, items in report["downloads"].items():
        if isinstance(items, dict) and "status" in items:
            provenance["sources"].append({"id": group, **items})
        elif isinstance(items, dict):
            for sid, res in items.items():
                provenance["sources"].append({"id": f"{group}/{sid}", **res})

    out_report = MANIFESTS / "acquisition_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MANIFESTS / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
