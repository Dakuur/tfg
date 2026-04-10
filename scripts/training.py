"""
training.py — Training loop utilities.

Provides: fix_seeds, EarlyStopping, train_epoch, val_epoch, save_checkpoint.
"""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch_geometric.loader import DataLoader


# ── reproducibility ────────────────────────────────────────────────────────────

def fix_seeds(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val_loss has not improved for `patience` epochs
    (skipped during the first `warm_up` epochs)."""

    def __init__(self, warm_up: int = 10, patience: int = 20, delta: float = 0.0):
        self.warm_up    = warm_up
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_score = float("inf")
        self.best_model = None
        self.epoch      = 0
        self.early_stop = False

    def __call__(self, epoch: int, val_loss: float, model: torch.nn.Module) -> bool:
        score = round(val_loss, 4)
        if epoch > self.warm_up:
            if score < self.best_score - self.delta:
                self.best_score = score
                self.best_model = copy.deepcopy(model)
                self.epoch      = epoch
                self.counter    = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop


# ── train / val loops ──────────────────────────────────────────────────────────

def train_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler:    torch.amp.GradScaler,
    device:    torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits    = model(batch.x, batch.edge_index, batch.batch)
            main_loss = criterion(logits, batch.y)
            aux_loss  = model.aux_loss.to(device)
            loss      = main_loss + aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_true.extend(batch.y.cpu().tolist())
        all_pred.extend(F.softmax(logits.detach(), dim=1).argmax(dim=1).cpu().tolist())

    n = len(loader)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_true, all_pred),
    }


@torch.no_grad()
def val_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    criterion: torch.nn.Module,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_scores = [], [], []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss   = criterion(logits, batch.y)

        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        all_true.extend(batch.y.cpu().tolist())
        all_pred.extend(probs.argmax(dim=1).cpu().tolist())
        all_scores.extend(probs[:, 1].cpu().tolist())

    n = len(loader)
    try:
        auc = roc_auc_score(all_true, all_scores)
    except ValueError:
        auc = float("nan")

    return {
        "loss":         total_loss / n,
        "acc":          accuracy_score(all_true, all_pred),
        "auc":          auc,
        "f1_macro":     f1_score      (all_true, all_pred, average="macro",   zero_division=0),
        "recall_N0":    recall_score  (all_true, all_pred, pos_label=0,        zero_division=0),
        "recall_N1":    recall_score  (all_true, all_pred, pos_label=1,        zero_division=0),
        "precision_N0": precision_score(all_true, all_pred, pos_label=0,       zero_division=0),
        "precision_N1": precision_score(all_true, all_pred, pos_label=1,       zero_division=0),
        "f1_N0":        f1_score      (all_true, all_pred, pos_label=0,        zero_division=0),
        "f1_N1":        f1_score      (all_true, all_pred, pos_label=1,        zero_division=0),
    }


# ── checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":        epoch,
            "val_auc":      metrics.get("auc"),
            "val_f1_macro": metrics.get("f1_macro"),
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
        },
        path,
    )
