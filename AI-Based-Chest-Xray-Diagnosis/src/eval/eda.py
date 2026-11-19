"""Quick EDA figures for the report: python -m src.eval.eda --data data/processed"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--out", default="reports/figures")
a = ap.parse_args()
root, out = Path(a.data), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(root / "meta.csv")
boxes = pd.read_csv(root / "boxes.csv")
print(meta.groupby("split")["Target"].agg(["count", "mean"]))
print(meta["class"].value_counts())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
meta["class"].value_counts().plot.bar(ax=ax[0], title="Class balance")
meta["PatientAge"].dropna().plot.hist(ax=ax[1], bins=20, title="Age")
meta["PatientSex"].value_counts().plot.bar(ax=ax[2], title="Sex")
plt.tight_layout(); plt.savefig(out / "eda_overview.png", dpi=120)

pos = meta[meta["Target"] == 1].head(6)
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
for axi, pid in zip(ax.ravel(), pos["patientId"]):
    axi.imshow(cv2.imread(str(root / "images" / f"{pid}.png"), 0), cmap="gray"); axi.axis("off")
    for _, b in boxes[boxes["patientId"] == pid].iterrows():
        axi.add_patch(plt.Rectangle((b.x1, b.y1), b.x2 - b.x1, b.y2 - b.y1, fill=False, ec="red", lw=2))
plt.tight_layout(); plt.savefig(out / "eda_boxes.png", dpi=120)
print("saved to", out)
