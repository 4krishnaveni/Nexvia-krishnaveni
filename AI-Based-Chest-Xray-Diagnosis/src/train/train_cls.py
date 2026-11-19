"""Train the classifier.
    python -m src.train.train_cls --data data/processed --epochs 20 --out models/classifier.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import CXRDataset
from src.models.classifier import CXRClassifier


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).squeeze(1).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on the VALIDATION set so displayed confidence is calibrated."""
    t = torch.zeros(1, requires_grad=True)  # log-temperature
    lg, yy = torch.tensor(logits), torch.tensor(y)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(lg / t.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp().item())


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--backbone", default="densenet121")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline/tests)")
    ap.add_argument("--out", default="models/classifier.pt")
    ap.add_argument("--report", default="reports/cls_train.json")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = CXRDataset(a.data, "train", "cls", augment=True)
    va = CXRDataset(a.data, "val", "cls")
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch_size)
    vl = DataLoader(va, a.batch_size, num_workers=a.workers)

    model = CXRClassifier(a.backbone, pretrained=not a.no_pretrained).to(device)
    pos = float(tr.df["Target"].sum())
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_state, bad, history = -1.0, None, 0, []
    for ep in range(a.epochs):
        model.set_backbone_trainable(ep >= a.freeze_epochs)  # head-only warm-up
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
        sched.step()
        logits, yv = predict_logits(model, vl, device)
        auc = roc_auc_score(yv, logits) if len(set(yv)) > 1 else float("nan")
        history.append({"epoch": ep + 1, "train_loss": run / len(tr), "val_auc": auc, "sec": time.time() - t0})
        print(history[-1])
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    if best_state is None:  # val AUC undefined (single-class val set); keep last weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    logits, yv = predict_logits(model, vl, device)
    temperature = fit_temperature(logits, yv)
    probs = 1 / (1 + np.exp(-logits / temperature))
    thr = youden_threshold(yv, probs)  # chosen on VAL only, never on test
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": a.backbone, "temperature": temperature,
                "threshold": thr, "version": f"{a.backbone}-{time.strftime('%Y%m%d')}"}, a.out)
    report = {"best_val_auc": best_auc, "val_auprc": float(average_precision_score(yv, probs)),
              "temperature": temperature, "threshold": thr, "history": history}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2))
    print(f"saved {a.out}  val_auc={best_auc:.4f} T={temperature:.3f} thr={thr:.3f}")


if __name__ == "__main__":
    main()
