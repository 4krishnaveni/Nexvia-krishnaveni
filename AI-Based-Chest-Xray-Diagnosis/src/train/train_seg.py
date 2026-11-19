"""Train the MONAI U-Net on box-derived masks.
    python -m src.train.train_seg --data data/processed --epochs 25 --out models/unet.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset
from src.models.segmenter import build_unet


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred * gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


@torch.no_grad()
def eval_dice_positives(model, loader, device) -> float:
    """Mean Dice over images that actually contain a mask."""
    model.eval()
    scores = []
    for x, m in loader:
        p = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu().numpy()
        m = m.numpy()
        for pi, mi in zip(p, m):
            if mi.sum() > 0:
                scores.append(dice_np(pi, mi))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/unet.pt")
    ap.add_argument("--report", default="reports/seg_train.json")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = CXRDataset(a.data, "train", "seg", augment=True)
    va = CXRDataset(a.data, "val", "seg")
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers)
    vl = DataLoader(va, a.batch_size, num_workers=a.workers)

    model = build_unet().to(device)
    dice_loss, bce = DiceLoss(sigmoid=True), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, history = -1.0, None, []
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, m in tl:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            out = model(x)
            loss = dice_loss(out, m) + bce(out, m)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
        sched.step()
        d = eval_dice_positives(model, vl, device)
        history.append({"epoch": ep + 1, "train_loss": run / len(tr), "val_dice_pos": d, "sec": time.time() - t0})
        print(history[-1])
        if d == d and d > best:  # d == d filters NaN
            best = d
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "version": f"unet2d-{time.strftime('%Y%m%d')}"}, a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({"best_val_dice_positives": best, "history": history}, indent=2))
    print(f"saved {a.out} best val dice (positives) = {best:.4f}")


if __name__ == "__main__":
    main()
