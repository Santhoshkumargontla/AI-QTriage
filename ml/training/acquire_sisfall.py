"""Download SisFall from Hugging Face mirror into data/raw/sisfall/ with provenance report."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "sisfall"
ZIP_PATH = OUT_DIR / "SisFall_dataset.zip"
EXTRACT_DIR = OUT_DIR / "SisFall_dataset"
REPORT_PATH = ROOT / "data" / "manifests" / "sisfall_acquire_report.json"

HF_URL = "https://huggingface.co/datasets/Algo-rythmic/Sisfall_Dataset/resolve/main/SisFall_dataset.zip"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_utc": _utc(),
        "source": "Hugging Face Algo-rythmic/Sisfall_Dataset (CC BY 4.0 mirror of SisFall)",
        "url": HF_URL,
        "original_paper": "https://doi.org/10.3390/s17010198",
    }

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000_000:
        report["download"] = {
            "status": "ALREADY_PRESENT",
            "path": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": ZIP_PATH.stat().st_size,
            "sha256": _sha256(ZIP_PATH),
        }
    else:
        req = Request(HF_URL, headers={"User-Agent": "AI-QTriage-research/1.0"})
        with urlopen(req, timeout=600) as resp:
            data = resp.read()
        ZIP_PATH.write_bytes(data)
        report["download"] = {
            "status": "DOWNLOADED",
            "path": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    # Extract if needed
    txt_count = len(list(EXTRACT_DIR.rglob("*.txt"))) if EXTRACT_DIR.exists() else 0
    if txt_count < 100:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        txt_count = len(list(EXTRACT_DIR.rglob("*.txt")))

    report["extract"] = {
        "path": str(EXTRACT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "txt_files": txt_count,
        "status": "OK" if txt_count > 0 else "EMPTY",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
