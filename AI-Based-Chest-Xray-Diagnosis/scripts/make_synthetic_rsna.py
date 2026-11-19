"""Create a tiny fake RSNA-format dataset (real DICOM files + CSVs) so you can test the WHOLE
pipeline in minutes without Kaggle. The 'lung opacity' is a bright blob inside the box, so
models can actually learn something. NOT real medical data.

    python scripts/make_synthetic_rsna.py --out data/synthetic --n 120
"""
import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SOP_DX = "1.2.840.10008.5.1.4.1.1.1.1"


def make_dicom(path: Path, arr: np.ndarray, pid: str, sex: str, age: int, mono1: bool) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SOP_DX
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = SOP_DX, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID, ds.PatientName = pid, f"SYNTH^{pid[:6]}"
    ds.PatientSex, ds.PatientAge, ds.Modality, ds.ViewPosition = sex, f"{age:03d}Y", "DX", "PA"
    ds.StudyDescription = "SYNTHETIC - NOT REAL"
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 1, 16, 12, 11, 0
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    px = (arr * 4095).astype(np.uint16)
    ds.PixelData = (4095 - px if mono1 else px).tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.save_as(str(path), write_like_original=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    (out / "stage_2_train_images").mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:a.size, 0:a.size] / a.size
    rows, cls_rows = [], []
    for _ in range(a.n):
        pid = str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))))
        # two dark "lungs" on a mid-grey body plus noise
        img = 0.55 + 0.05 * rng.standard_normal((a.size, a.size))
        for cx in (0.3, 0.7):
            img -= 0.30 * np.exp(-(((xx - cx) / 0.13) ** 2 + ((yy - 0.5) / 0.28) ** 2))
        pos = rng.random() < 0.3
        if pos:
            bw, bh = int(rng.uniform(0.1, 0.2) * a.size), int(rng.uniform(0.15, 0.3) * a.size)
            x = int(rng.uniform(0.15, 0.75) * a.size)
            y = int(rng.uniform(0.3, 0.55) * a.size)
            img[y:y + bh, x:x + bw] += 0.30
            rows.append({"patientId": pid, "x": x, "y": y, "width": bw, "height": bh, "Target": 1})
            cls_rows.append({"patientId": pid, "class": "Lung Opacity"})
        else:
            rows.append({"patientId": pid, "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0})
            cls_rows.append({"patientId": pid, "class": "Normal"})
        img = np.clip(img, 0, 1)
        make_dicom(out / "stage_2_train_images" / f"{pid}.dcm", img, pid,
                   str(rng.choice(["M", "F"])), int(rng.integers(20, 85)), mono1=bool(rng.random() < 0.2))
    pd.DataFrame(rows).to_csv(out / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out / "stage_2_detailed_class_info.csv", index=False)
    print(f"wrote {a.n} synthetic DICOMs to {out}")


if __name__ == "__main__":
    main()
