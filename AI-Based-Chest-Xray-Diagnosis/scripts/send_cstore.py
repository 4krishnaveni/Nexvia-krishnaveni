"""Alternative to REST: send via the DICOM protocol (C-STORE) to Orthanc on port 4242.
    python scripts/send_cstore.py data/synthetic/stage_2_train_images --limit 3
"""
import argparse
from pathlib import Path

import pydicom
from pynetdicom import AE, StoragePresentationContexts

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=4242)
ap.add_argument("--aet", default="ORTHANC")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ae = AE(ae_title="CXR_SENDER")
ae.requested_contexts = StoragePresentationContexts
assoc = ae.associate(a.host, a.port, ae_title=a.aet)
assert assoc.is_established, "could not associate with Orthanc"
files = sorted(Path(a.path).glob("*.dcm"))
for f in files[: a.limit or None]:
    print(f.name, assoc.send_c_store(pydicom.dcmread(f)))
assoc.release()
