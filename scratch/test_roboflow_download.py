import os
import sys
from roboflow import Roboflow
from backend.config import settings

# Load API Key from environment or settings
api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

def test_download():
    # Print info about api_key
    print(f"API Key: {api_key[:5]}... (length={len(api_key)})")
    rf = Roboflow(api_key=api_key)
    
    datasets_to_check = [
        {"workspace": "w-afwxp", "project": "wound-ebsdw"},
        {"workspace": "w-afwxp", "project": "new-wound-model"},
        {"workspace": "injury-segmentation", "project": "myyolov5datasetforinjuries-qmyyc"}
    ]
    
    for ds in datasets_to_check:
        print(f"\nChecking project: {ds['workspace']}/{ds['project']}...")
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            versions = project.versions()
            print(f"  Available versions: {[v.version for v in versions]}")
            if versions:
                latest_v = versions[0].version
                print(f"  Latest version: {latest_v}")
                # Print some metadata
                metadata = versions[0]
                print(f"  Images count: {getattr(metadata, 'images', 'unknown')}")
                print(f"  Classes: {getattr(metadata, 'classes', 'unknown')}")
        except Exception as e:
            print(f"  Error checking project: {e}")

if __name__ == "__main__":
    test_download()
