"""Upload DICOM files (or a folder) to Orthanc over REST.
    python scripts/upload_to_orthanc.py data/synthetic/stage_2_train_images --limit 5
"""
import argparse
import os
from pathlib import Path

from src.service.orthanc_client import OrthancClient

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--url", default=os.getenv("ORTHANC_URL", "http://localhost:8042"))
a = ap.parse_args()
c = OrthancClient(a.url, os.getenv("ORTHANC_USER", "orthanc"), os.getenv("ORTHANC_PASSWORD", "change-me-orthanc"))
p = Path(a.path)
files = [p] if p.is_file() else sorted(p.glob("*.dcm"))
for f in files[: a.limit or None]:
    print(f.name, c.upload(f.read_bytes()).get("Status"))
