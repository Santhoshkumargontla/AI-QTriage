"""Download licensed public datasets. Does not remap ulcer/healthy feet to swelling.

Records a comparison table, then downloads only candidates that pass license +
domain notes. Never overwrites existing processed manifests.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXT = os.path.join(ROOT, "data", "datasets", "external")
os.makedirs(EXT, exist_ok=True)
COMPARISON = os.path.join(EXT, "DATASET_CANDIDATE_COMPARISON.json")

CANDIDATES = [
    {
        "id": "mendeley_hsj38fwnvr_v2",
        "name": "Lower limb and feet wound image dataset",
        "url": "https://data.mendeley.com/datasets/hsj38fwnvr/2",
        "license": "CC BY 4.0",
        "size_claimed": "8129 images (Normal 2575, Wound 2686, Mask 2686)",
        "classes": ["normal_foot", "wound_unspecified"],
        "label_format": "folders + paired masks",
        "domain": "chronic/acute lower-limb and foot wounds; healthy feet",
        "matches_app_classes": False,
        "real_or_synthetic": "PUBLIC_REAL_PHOTOS",
        "suitable_for": ["unet_binary_segmentation", "efficientnet_normal_reject"],
        "do_not": "Do not remap wound->swelling or foot ulcers to cut/bruise.",
        "download": True,
        "priority": 1,
    },
    {
        "id": "hf_wseg_dataset",
        "name": "subbareddyoota/wseg_dataset (WSNet / WACV 2023)",
        "url": "https://huggingface.co/datasets/subbareddyoota/wseg_dataset",
        "license": "CC-BY-NC-4.0",
        "size_claimed": "2686 wound + 2686 masks (HF viewer also lists ~5370 rows)",
        "classes": ["wound_binary"],
        "label_format": "image/mask pairs",
        "domain": "clinical wound photographs, not cut/bruise/swelling taxonomy",
        "matches_app_classes": False,
        "real_or_synthetic": "PUBLIC_REAL_PHOTOS",
        "suitable_for": ["unet_binary_segmentation"],
        "do_not": "Academic/non-commercial only. Do not remap to swelling.",
        "download": True,
        "priority": 2,
    },
    {
        "id": "github_medetec_224",
        "name": "UWM Medetec foot ulcer 224 (wound-segmentation repo)",
        "url": "https://github.com/uwm-bigdata/wound-segmentation",
        "license": "Medetec stock terms + UWM public redistribution of annotated 224px set",
        "size_claimed": "train/test 224px image-mask pairs (order of 10^2)",
        "classes": ["foot_ulcer"],
        "label_format": "images/ + labels/",
        "domain": "foot ulcers, not sports-injury cut/bruise",
        "matches_app_classes": False,
        "real_or_synthetic": "PUBLIC_REAL_PHOTOS",
        "suitable_for": ["unet_binary_segmentation"],
        "do_not": "Do not advertise Medetec classes as YOLO cut/bruise/wound.",
        "download": True,
        "priority": 3,
    },
    {
        "id": "github_fuseg",
        "name": "FUSeg Foot Ulcer Segmentation Challenge",
        "url": "https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge",
        "license": "AZH clinic permission / MICCAI challenge data-use (not a blanket CC BY grant for the images)",
        "size_claimed": "~1210 annotated foot ulcer images",
        "classes": ["foot_ulcer"],
        "label_format": "train/validation images+labels",
        "domain": "chronic foot ulcers",
        "matches_app_classes": False,
        "real_or_synthetic": "PUBLIC_REAL_PHOTOS",
        "suitable_for": ["unet_binary_segmentation"],
        "do_not": "Skip if Medetec/wseg already provide enough pairs; large download on CPU.",
        "download": False,
        "priority": 4,
    },
    {
        "id": "roboflow_self_harm_detection",
        "name": "Roboflow Self Harm Detection (Bruises/Burns/Cuts)",
        "url": "https://universe.roboflow.com/testlayoutfortext/self-harm-detection-nuvf7",
        "license": "CC BY 4.0",
        "size_claimed": "293 images",
        "classes": ["Bruises", "Burns", "Cuts"],
        "label_format": "YOLO boxes",
        "domain": "closest public match to cut/bruise photos",
        "matches_app_classes": "partial (cut/bruise; burn kept separate, not remapped)",
        "real_or_synthetic": "PUBLIC_PHOTOS_UNVERIFIED_UNTIL_DOWNLOAD",
        "suitable_for": ["yolo_detection", "efficientnet_cut_bruise"],
        "do_not": "Do not remap Burns to swelling. Content may depict self-harm; research-only.",
        "download": True,
        "priority": 2,
    },
    {
        "id": "roboflow_injury_qmyyc",
        "name": "Roboflow myyolov5datasetforinjuries",
        "url": "https://universe.roboflow.com/injury-segmentation/myyolov5datasetforinjuries-qmyyc",
        "license": "listed CC BY 4.0 on Universe (verify on download)",
        "size_claimed": "unknown until download",
        "classes": ["injury boxes; verify names after download"],
        "label_format": "YOLO",
        "domain": "injury detection",
        "matches_app_classes": "unknown_until_download",
        "real_or_synthetic": "PUBLIC_PHOTOS_UNVERIFIED_UNTIL_DOWNLOAD",
        "suitable_for": ["yolo_detection"],
        "do_not": "Do not merge until class names verified.",
        "download": True,
        "priority": 3,
    },
    {
        "id": "willie_benchmark",
        "name": "QianGroup/willie-benchmark",
        "url": "https://huggingface.co/datasets/QianGroup/willie-benchmark",
        "license": "mixed-dataset-terms (index only; images not granted by the archive)",
        "size_claimed": "split index, not the pixels",
        "classes": ["5-class chronic wound taxonomy + no_wound"],
        "label_format": "index/splits",
        "domain": "FUSeg/AZH/Medetec chronic wounds",
        "matches_app_classes": False,
        "real_or_synthetic": "INDEX_ONLY",
        "suitable_for": [],
        "do_not": "Do not treat the HF archive as a license to the source images.",
        "download": False,
        "priority": 9,
    },
    {
        "id": "azh_wound_classification",
        "name": "AZH wound type classification (venous/diabetic/pressure/surgical)",
        "url": "https://sites.uwm.edu/bigdata/datasets/",
        "license": "UWM/AZH share; request/email for some releases",
        "size_claimed": "400-730 images",
        "classes": ["venous", "diabetic", "pressure", "surgical"],
        "label_format": "jpg folders",
        "domain": "chronic wound types, not cut/bruise/swelling",
        "matches_app_classes": False,
        "real_or_synthetic": "PUBLIC_REAL_PHOTOS",
        "suitable_for": [],
        "do_not": "Do not remap surgical/pressure to swelling.",
        "download": False,
        "priority": 8,
    },
]


def _roboflow_key() -> str | None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    path = os.path.join(ROOT, "scripts", "prepare_yolo_dataset.py")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    match = re.search(r'ROBOFLOW_API_KEY",\s*"([^"]+)"', text)
    if match:
        print("Using Roboflow key from existing script default (do not log the value).")
        return match.group(1)
    return None


def _http_download(url: str, dest: str, timeout: int = 120) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"GET {url}")
    try:
        req = Request(url, headers={"User-Agent": "AI-QTriage-research/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
        if len(data) < 1000:
            print(f"  too small ({len(data)} bytes) type={ctype}")
            return False
        with open(dest, "wb") as handle:
            handle.write(data)
        print(f"  wrote {dest} ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"  failed: {type(exc).__name__}: {exc}")
        return False


def try_mendeley() -> dict:
    dest_dir = os.path.join(EXT, "mendeley_hsj38fwnvr")
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "dataset.zip")
    urls = [
        "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hsj38fwnvr-2.zip",
        "https://data.mendeley.com/public-files/datasets/hsj38fwnvr/files/hsj38fwnvr-2.zip",
        "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hsj38fwnvr-2.zip",
    ]
    api_urls = [
        "https://data.mendeley.com/public-api/datasets/hsj38fwnvr/files?version=2",
        "https://api.elsevier.com/content/dataset/doi/10.17632/hsj38fwnvr.2",
    ]
    result = {"status": "FAILED", "paths": [], "notes": []}
    for api in api_urls:
        try:
            req = Request(api, headers={"User-Agent": "AI-QTriage-research/1.0", "Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                body = resp.read()[:4000]
            result["notes"].append(f"api {api} -> {body[:200]!r}")
            print("Mendeley API", api, body[:180])
        except Exception as exc:
            result["notes"].append(f"api {api} {type(exc).__name__}: {exc}")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 10_000_000:
        result["status"] = "ALREADY_PRESENT"
        result["paths"] = [zip_path]
        return result
    for url in urls:
        if _http_download(url, zip_path, timeout=180):
            result["status"] = "DOWNLOADED"
            result["paths"] = [zip_path]
            return result
    result["status"] = "BLOCKED_NO_DIRECT_URL"
    return result


def try_hf_wseg() -> dict:
    dest = os.path.join(EXT, "hf_wseg_dataset")
    os.makedirs(dest, exist_ok=True)
    result = {"status": "FAILED", "paths": [], "notes": []}
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    except Exception as exc:
        result["notes"].append(str(exc))
        return result
    try:
        files = list_repo_files("subbareddyoota/wseg_dataset", repo_type="dataset")
        result["notes"].append({"n_files": len(files), "sample": files[:30]})
        print("HF wseg files", len(files), files[:20])
        zip_names = [f for f in files if f.lower().endswith(".zip")]
        if zip_names:
            path = hf_hub_download(
                "subbareddyoota/wseg_dataset",
                zip_names[0],
                repo_type="dataset",
                local_dir=dest,
            )
            result["status"] = "DOWNLOADED"
            result["paths"] = [path]
            return result
        path = snapshot_download(
            "subbareddyoota/wseg_dataset",
            repo_type="dataset",
            local_dir=dest,
            allow_patterns=["*.jpg", "*.png", "*.zip", "*.json", "*.md", "*.parquet"],
        )
        result["status"] = "DOWNLOADED"
        result["paths"] = [path]
        return result
    except Exception as exc:
        result["notes"].append(f"{type(exc).__name__}: {exc}")
        print("HF wseg failed", exc)
        return result


def try_medetec_git() -> dict:
    dest = os.path.join(EXT, "wound-segmentation")
    result = {"status": "FAILED", "paths": [], "notes": []}
    med = os.path.join(dest, "data", "Medetec_foot_ulcer_224")
    if os.path.isdir(med) and any(os.walk(med)):
        n_img = 0
        for _root, _dirs, files in os.walk(med):
            n_img += sum(1 for f in files if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if n_img > 0:
            result["status"] = "ALREADY_PRESENT"
            result["paths"] = [med]
            result["notes"].append({"images": n_img})
            return result
    if not os.path.isdir(os.path.join(dest, ".git")):
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/uwm-bigdata/wound-segmentation.git",
            dest,
        ]
        print(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        err = proc.stderr or ""
        result["notes"].append({"clone_rc": proc.returncode, "stderr": err[-800:] if err else ""})
        if proc.returncode != 0:
            return result
    print("sparse-checkout Medetec + wound_dataset")
    sparse = subprocess.run(
        ["git", "-C", dest, "sparse-checkout", "set", "data/Medetec_foot_ulcer_224", "data/wound_dataset"],
        capture_output=True,
        text=True,
    )
    serr = sparse.stderr or ""
    result["notes"].append({"sparse_rc": sparse.returncode, "stderr": serr[-400:] if serr else ""})
    if os.path.isdir(med):
        result["status"] = "DOWNLOADED"
        result["paths"] = [med]
    return result


def try_roboflow() -> dict:
    result = {"status": "FAILED", "paths": [], "notes": []}
    key = _roboflow_key()
    if not key:
        result["status"] = "SKIPPED_NO_API_KEY"
        return result
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key)
    except Exception as exc:
        result["notes"].append(f"{type(exc).__name__}: {exc}")
        return result
    targets = [
        ("testlayoutfortext", "self-harm-detection-nuvf7", "roboflow_self_harm"),
        ("injury-segmentation", "myyolov5datasetforinjuries-qmyyc", "roboflow_injury_qmyyc"),
    ]
    for workspace, project, folder in targets:
        dest = os.path.join(EXT, folder)
        if os.path.isdir(dest) and any(os.scandir(dest)):
            result["paths"].append(dest)
            result["notes"].append(f"{folder} already present")
            continue
        try:
            print(f"Roboflow {workspace}/{project}")
            proj = rf.workspace(workspace).project(project)
            versions = proj.versions()
            latest = versions[0] if versions else None
            ver_num = getattr(latest, "version", None) or 1
            ds = proj.version(int(str(ver_num).split(".")[0]) if ver_num else 1).download(
                "yolov8", location=dest
            )
            result["paths"].append(dest)
            result["notes"].append({"project": project, "location": dest, "dataset": str(ds)})
        except Exception as exc:
            result["notes"].append({"project": project, "error": f"{type(exc).__name__}: {exc}"})
            print("Roboflow failed", project, exc)
    result["status"] = "DOWNLOADED" if result["paths"] else "FAILED"
    return result


def main():
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": CANDIDATES,
        "downloads": {},
    }
    print("=== Mendeley (skip if previously 403/404) ===")
    report["downloads"]["mendeley"] = {
        "status": "BLOCKED_NO_DIRECT_URL",
        "notes": ["S3 403 and public-files 404 on 2026-08-29; requires browser Download All"],
    }
    print("=== HuggingFace wseg ===")
    report["downloads"]["hf_wseg"] = try_hf_wseg()
    print("=== GitHub Medetec ===")
    report["downloads"]["medetec_git"] = try_medetec_git()
    print("=== Roboflow ===")
    report["downloads"]["roboflow"] = try_roboflow()
    with open(COMPARISON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("Wrote", COMPARISON)
    print(json.dumps({k: v.get("status") for k, v in report["downloads"].items()}, indent=2))


if __name__ == "__main__":
    main()
