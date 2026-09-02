"""Adapter join/split logic (previously only defer-tested).

Covers the NIH/VinDr validation-split carve, on synthetic raw inputs shaped
like the real file formats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess import common, nih_cxr14, vindr_cxr
from src.data.label_map import load_equivalence


def _jpg(p, sz=(90, 90)):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(0).integers(0, 255, sz + (3,), np.uint8), "RGB").save(p)


def test_nih_carves_patient_level_val(tmp_path):
    ent = []
    for i in range(10):
        nm = f"{i:08d}_000.png"; _jpg(tmp_path / "images" / nm)
        ent.append({"Image Index": nm, "Patient ID": i,
                    "Finding Labels": "Effusion" if i % 2 else "No Finding", "View Position": "PA"})
    pd.DataFrame(ent).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    (tmp_path / "test_list.txt").write_text("00000000_000.png\n00000001_000.png\n")
    df = pd.read_csv(nih_cxr14.preprocess_nih(tmp_path, tmp_path / "pre"))
    assert {"train", "val", "test"} <= set(df["split"])
    df["patient"] = df["image_id"].str.split("_").str[0]
    assert (df.groupby("patient")["split"].nunique() == 1).all()  # no patient in 2 splits


def _dicom(path, mono="MONOCHROME2"):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset(); fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.Rows = 70; ds.Columns = 80; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
    ds.SamplesPerPixel = 1; ds.PixelRepresentation = 0; ds.PhotometricInterpretation = mono
    ds.RescaleSlope = 1; ds.RescaleIntercept = 0
    ds.PixelData = np.random.default_rng(0).integers(0, 4000, (70, 80), np.uint16).tobytes()
    ds.is_little_endian = True; ds.is_implicit_VR = False; ds.save_as(str(path))


def test_vindr_train_val_test_and_majority_vote(tmp_path):
    vmap = load_equivalence().dataset_to_canonical("vindr-cxr")
    cols = list(vmap.keys())
    ann = tmp_path / "annotations"; ann.mkdir(parents=True)
    def rows(ids, eff_positive):
        rs = []
        for iid in ids:
            for _ in range(2):  # 2 radiologists -> majority vote
                r = {"image_id": iid, "rad_id": "R", **{c: 0 for c in cols}}
                if eff_positive:
                    r["Pleural effusion"] = 1
                rs.append(r)
        return rs
    train_ids = [f"tr{i}" for i in range(6)]; test_ids = [f"te{i}" for i in range(2)]
    pd.DataFrame(rows(train_ids, True)).to_csv(ann / "image_labels_train.csv", index=False)
    pd.DataFrame(rows(test_ids, False)).to_csv(ann / "image_labels_test.csv", index=False)
    for iid in train_ids:
        _dicom(tmp_path / "train" / f"{iid}.dicom")
    for iid in test_ids:
        _dicom(tmp_path / "test" / f"{iid}.dicom")
    df = pd.read_csv(vindr_cxr.preprocess_vindr(tmp_path, tmp_path / "pre"))
    assert {"train", "val", "test"} <= set(df["split"])
    assert (df[df["split"] != "test"]["Pleural_Effusion"] == 1.0).all()  # majority vote -> 1
    assert df[df["split"] == "test"]["Pleural_Effusion"].eq(0.0).all()
