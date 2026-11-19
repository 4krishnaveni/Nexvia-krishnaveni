"""Test-set evaluation: classification, calibration, subgroup AUROC, Dice, Grad-CAM pointing game.
    python -m src.eval.evaluate --data data/processed --cls models/classifier.pt --seg models/unet.pt
Writes reports/metrics.json and reports/figures/*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset
from src.explain.gradcam import blend_on_image, compute_cam, pointing_game_hit
from src.models.classifier import load_classifier
from src.models.segmenter import load_unet
from src.train.train_seg import dice_np


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else None


def expected_calibration_error(y, p, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def subgroup_auc(df: pd.DataFrame) -> dict:
    out = {}
    if "PatientSex" in df:
        for s, g in df.groupby("PatientSex"):
            if s in ("M", "F"):
                out[f"sex={s}"] = {"n": len(g), "auroc": safe_auc(g["y"], g["p"])}
    if "PatientAge" in df:
        bins = pd.cut(df["PatientAge"], [0, 40, 60, 200], labels=["<40", "40-59", "60+"])
        for b, g in df.groupby(bins, observed=True):
            out[f"age={b}"] = {"n": len(g), "auroc": safe_auc(g["y"], g["p"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--cls", default="models/classifier.pt")
    ap.add_argument("--seg", default="models/unet.pt")
    ap.add_argument("--out", default="reports/metrics.json")
    ap.add_argument("--n-gradcam-figs", type=int, default=8)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.data)

    cls, ck = load_classifier(a.cls, device)
    T, thr = ck["temperature"], ck["threshold"]
    ds = CXRDataset(root, "test", "cls")
    ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 32):
            ps.append(torch.sigmoid(cls(x.to(device)).squeeze(1) / T).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ece = expected_calibration_error(y, p)
    metrics = {
        "n_test": int(len(y)), "threshold_from_val": thr,
        "auroc": safe_auc(y, p), "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred)), "ece": ece,
    }
    meta = ds.df.copy()
    meta["y"], meta["p"] = y, p
    metrics["subgroups"] = subgroup_auc(meta)

    # Grad-CAM pointing game on true positives + Dice for the U-Net
    boxes_df = pd.read_csv(root / "boxes.csv")
    seg, _ = load_unet(a.seg, device)
    seg_ds = CXRDataset(root, "test", "seg")
    hits, dices, figs = [], [], 0
    fig_dir = Path(a.out).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for i, row in ds.df.iterrows():
        if row["Target"] != 1:
            continue
        x, _ = ds[i]
        cam = compute_cam(cls, x[None].to(device))[0]
        boxes = boxes_df[boxes_df["patientId"] == row["patientId"]][["x1", "y1", "x2", "y2"]].to_numpy()
        hits.append(pointing_game_hit(cam, boxes))
        xs, m = seg_ds[i]
        with torch.no_grad():
            pm = (torch.sigmoid(seg(xs[None].to(device))) > 0.5).float().cpu().numpy()[0]
        dices.append(dice_np(pm, m.numpy()))
        if figs < a.n_gradcam_figs:
            img = xs[0].numpy()
            cv2.imwrite(str(fig_dir / f"gradcam_{row['patientId'][:8]}_{'hit' if hits[-1] else 'miss'}.png"),
                        cv2.cvtColor(blend_on_image(img, cam), cv2.COLOR_RGB2BGR))
            figs += 1
    metrics["pointing_game_accuracy"] = float(np.mean(hits)) if hits else None
    metrics["dice_positives_box_masks"] = float(np.mean(dices)) if dices else None
    metrics["n_test_positives"] = len(hits)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
