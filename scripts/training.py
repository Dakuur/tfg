"""
training.py — Training loop utilities.

Provides:
  fix_seeds              — set all random seeds for reproducibility
  EarlyStopping          — monitor val_loss with warm-up period
  train_epoch            — one slide-level training epoch
  val_epoch              — one slide-level validation epoch
  aggregate_patient_probs — combine per-slide P(N1) → patient P(N1)
  train_epoch_patient    — one patient-level MIL training epoch
  val_epoch_patient      — one patient-level MIL validation epoch
  save_checkpoint        — save model (+ optional patient_aggregator) weights
"""

import copy
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch_geometric.data import Batch
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


# ── slide-level train / val loops ──────────────────────────────────────────────

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

    return _metrics_dict(total_loss / n, all_true, all_pred, all_scores)


# ── patient-level (MIL) aggregation & loops ────────────────────────────────────

def aggregate_patient_probs(
    slide_probs_n1: torch.Tensor,
    method:         str = "noisy_or",
) -> torch.Tensor:
    """Aggregate per-slide P(N1) → scalar patient P(N1).

    Args:
        slide_probs_n1 : 1-D tensor shape (n_slides,), values in [0, 1].
        method         : 'noisy_or' | 'max' | 'lse' | 'mean'

    Returns:
        Scalar tensor — patient-level probability of being N1.
    """
    p = slide_probs_n1.clamp(1e-7, 1.0 - 1e-7)
    if method == "noisy_or":
        # P(pat=N1) = 1 - Π(1 - p_i), computed in log-space for stability
        return (1.0 - torch.exp(torch.log1p(-p).sum())).clamp(1e-7, 1.0 - 1e-7)
    if method == "max":
        return p.max()
    if method == "lse":
        # smooth-max in logit space: sigmoid(logsumexp(logit(p_i)))
        return torch.sigmoid(torch.logsumexp(torch.logit(p), dim=0))
    if method == "mean":
        return p.mean()
    raise ValueError(f"Unknown aggregation method: {method!r}")


def train_epoch_patient(
    model:              torch.nn.Module,
    loader,
    optimizer:          torch.optim.Optimizer,
    class_weights:      torch.Tensor,
    scaler:             torch.amp.GradScaler,
    device:             torch.device,
    aggregation:        str = "noisy_or",
    patient_aggregator: Optional[torch.nn.Module] = None,
) -> dict:
    """One patient-level MIL training epoch.

    All slides from all patients in the batch are processed in a single
    forward pass; then per-patient probabilities are aggregated and a
    single BCE loss is back-propagated per patient.
    """
    model.train()
    if patient_aggregator is not None:
        patient_aggregator.train()

    total_loss = 0.0
    all_true, all_pred = [], []

    for all_graphs, labels, slide_counts in loader:
        pyg_batch = Batch.from_data_list([g.to(device) for g in all_graphs])
        labels    = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pat_probs = _forward_patient(
                model, pyg_batch, slide_counts, aggregation, patient_aggregator, device
            )
            aux_loss = model.aux_loss.to(device)

        # BCE is unsafe inside autocast (requires float32 inputs)
        labels_float = labels.float()
        sample_w = torch.where(
            labels.bool(),
            class_weights[1].to(device),
            class_weights[0].to(device),
        )
        main_loss = F.binary_cross_entropy(pat_probs.float(), labels_float, weight=sample_w)
        loss = main_loss + aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_true.extend(labels.cpu().tolist())
        all_pred.extend((pat_probs.detach() > 0.5).long().cpu().tolist())

    n = len(loader)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_true, all_pred),
    }


@torch.no_grad()
def val_epoch_patient(
    model:              torch.nn.Module,
    loader,
    class_weights:      torch.Tensor,
    device:             torch.device,
    aggregation:        str = "noisy_or",
    patient_aggregator: Optional[torch.nn.Module] = None,
) -> dict:
    """One patient-level MIL validation epoch."""
    model.eval()
    if patient_aggregator is not None:
        patient_aggregator.eval()

    total_loss = 0.0
    all_true, all_pred, all_scores = [], [], []

    for all_graphs, labels, slide_counts in loader:
        pyg_batch = Batch.from_data_list([g.to(device) for g in all_graphs])
        labels    = labels.to(device)

        pat_probs = _forward_patient(
            model, pyg_batch, slide_counts, aggregation, patient_aggregator, device
        )

        labels_float = labels.float()
        sample_w = torch.where(
            labels.bool(),
            class_weights[1].to(device),
            class_weights[0].to(device),
        )
        loss = F.binary_cross_entropy(pat_probs.float(), labels_float, weight=sample_w)

        total_loss += loss.item()
        all_true.extend(labels.cpu().tolist())
        all_pred.extend((pat_probs > 0.5).long().cpu().tolist())
        all_scores.extend(pat_probs.cpu().tolist())

    n = len(loader)
    return _metrics_dict(total_loss / n, all_true, all_pred, all_scores)


# ── internal helpers ───────────────────────────────────────────────────────────

def _forward_patient(
    model:              torch.nn.Module,
    pyg_batch:          Batch,
    slide_counts:       list[int],
    aggregation:        str,
    patient_aggregator: Optional[torch.nn.Module],
    device:             torch.device,
) -> torch.Tensor:
    """Run one batched forward pass and return patient-level P(N1) tensor.

    Returns: 1-D FloatTensor of shape (n_patients,), values in (0, 1).
    """
    pat_probs: list[torch.Tensor] = []
    start = 0

    if aggregation == "attention" and patient_aggregator is not None:
        # Gated Attention MIL: encode → aggregate per patient → mlp → P(N1)
        h_all = model.encode(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)
        for count in slide_counts:
            h_pat = h_all[start:start + count]          # (count, D)
            h_agg = patient_aggregator(h_pat)            # (1, D)
            logit = model.mlp(h_agg)                     # (1, 2)
            prob  = F.softmax(logit, dim=1)[0, 1]
            pat_probs.append(prob)
            start += count
    else:
        # Prob-based aggregation: slide probs → aggregate function
        logits_all = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)
        probs_all  = F.softmax(logits_all, dim=1)[:, 1]
        for count in slide_counts:
            slide_probs = probs_all[start:start + count]
            pat_probs.append(aggregate_patient_probs(slide_probs, method=aggregation))
            start += count

    return torch.stack(pat_probs)   # (n_patients,)


def _metrics_dict(loss: float, all_true: list, all_pred: list, all_scores: list) -> dict:
    try:
        auc = roc_auc_score(all_true, all_scores)
    except ValueError:
        auc = float("nan")
    return {
        "loss":         loss,
        "acc":          accuracy_score(all_true, all_pred),
        "auc":          auc,
        "f1_macro":     f1_score      (all_true, all_pred, average="macro",  zero_division=0),
        "recall_N0":    recall_score  (all_true, all_pred, pos_label=0,       zero_division=0),
        "recall_N1":    recall_score  (all_true, all_pred, pos_label=1,       zero_division=0),
        "precision_N0": precision_score(all_true, all_pred, pos_label=0,      zero_division=0),
        "precision_N1": precision_score(all_true, all_pred, pos_label=1,      zero_division=0),
        "f1_N0":        f1_score      (all_true, all_pred, pos_label=0,       zero_division=0),
        "f1_N1":        f1_score      (all_true, all_pred, pos_label=1,       zero_division=0),
    }


# ── checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:              torch.nn.Module,
    optimizer:          torch.optim.Optimizer,
    epoch:              int,
    metrics:            dict,
    path:               Path,
    patient_aggregator: Optional[torch.nn.Module] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":        epoch,
        "val_auc":      metrics.get("auc"),
        "val_f1_macro": metrics.get("f1_macro"),
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
    }
    if patient_aggregator is not None:
        ckpt["patient_aggregator"] = patient_aggregator.state_dict()
    torch.save(ckpt, path)
