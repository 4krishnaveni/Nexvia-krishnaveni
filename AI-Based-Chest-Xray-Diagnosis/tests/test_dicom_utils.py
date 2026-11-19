import numpy as np

from scripts.make_synthetic_rsna import make_dicom
from src.data.dicom_utils import dicom_to_array, extract_metadata, parse_age, read_dicom
from src.data.make_masks import boxes_to_mask


def _img():
    rng = np.random.default_rng(1)
    return rng.random((64, 64)).astype(np.float32)


def test_monochrome1_is_inverted_to_match_monochrome2(tmp_path):
    img = _img()
    make_dicom(tmp_path / "a.dcm", img, "p1", "M", 40, mono1=False)
    make_dicom(tmp_path / "b.dcm", img, "p2", "F", 40, mono1=True)
    a = dicom_to_array(read_dicom(tmp_path / "a.dcm"))
    b = dicom_to_array(read_dicom(tmp_path / "b.dcm"))
    assert a.shape == (64, 64) and 0 <= a.min() and a.max() <= 1
    assert np.allclose(a, b, atol=1e-3)


def test_read_from_bytes_and_metadata(tmp_path):
    make_dicom(tmp_path / "a.dcm", _img(), "p1", "F", 58, mono1=False)
    ds = read_dicom((tmp_path / "a.dcm").read_bytes())
    meta = extract_metadata(ds)
    assert meta["PatientSex"] == "F" and meta["PatientAge"] == 58.0 and meta["Modality"] == "DX"


def test_parse_age():
    assert parse_age("058Y") == 58.0 and parse_age("") is None and parse_age("xx") is None


def test_boxes_to_mask():
    m = boxes_to_mask([(10, 10, 20, 30)], 64)
    assert m.shape == (64, 64) and m.sum() == 10 * 20
    assert boxes_to_mask([], 64).sum() == 0
    assert boxes_to_mask([(60, 60, 100, 100)], 64).sum() == 16  # clipped to the image
