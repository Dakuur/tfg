"""
data.py — Graph dataset loading utilities.

Provides helpers for loading pre-built PyG graph files and computing class
weights / samplers to compensate for the N0/N1 label imbalance.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def load_graphs(split_dir: Path) -> list[Data]:
    """Load all .pt graph files from a directory, sorted by name."""
    paths = sorted(split_dir.glob("*.pt"))
    if not paths:
        sys.exit(f"[ERROR] No .pt files found in {split_dir}")
    graphs = []
    for p in tqdm(paths, desc=f"Loading {split_dir.name}", unit="graph", leave=False):
        graphs.append(torch.load(p, weights_only=False))
    print(f"[INFO] {split_dir.name}: {len(graphs)} graphs loaded")
    return graphs


def compute_class_weights(graphs: list[Data]) -> torch.Tensor:
    """Balanced inverse-frequency class weights for CrossEntropyLoss.

    Uses the formula  w_c = N / (C * n_c)  so that the expected loss is the
    same regardless of class frequency (standard sklearn 'balanced' convention).
    """
    labels = np.array([g.y.item() for g in graphs])
    counts = np.bincount(labels, minlength=2).astype(float)
    n_total, n_classes = len(labels), 2
    weights = n_total / (n_classes * np.where(counts == 0, 1.0, counts))
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights — N0: {w[0]:.4f}  N1: {w[1]:.4f}  "
          f"(counts N0={int(counts[0])}, N1={int(counts[1])})")
    return w


def make_weighted_sampler(graphs: list[Data]) -> torch.utils.data.WeightedRandomSampler:
    """WeightedRandomSampler so every mini-batch is roughly class-balanced."""
    labels  = np.array([g.y.item() for g in graphs])
    counts  = np.bincount(labels, minlength=2).astype(float)
    class_w = torch.tensor(1.0 / np.where(counts == 0, 1.0, counts), dtype=torch.float32)
    sample_w = class_w[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_w, num_samples=len(sample_w), replacement=True
    )


def make_loaders(
    train_graphs: list[Data],
    val_graphs:   list[Data],
    batch_size:   int,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) with class-balanced sampling on train."""
    sampler = make_weighted_sampler(train_graphs)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader