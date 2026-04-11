"""
data.py — Graph dataset loading utilities.

Provides helpers for loading pre-built PyG graph files and computing class
weights / samplers to compensate for the N0/N1 label imbalance.

Slide-level helpers
-------------------
load_graphs            — load .pt files from a split directory
compute_class_weights  — balanced inverse-frequency weights (sklearn 'balanced')
make_weighted_sampler  — WeightedRandomSampler for class-balanced batches
make_loaders           — (train_loader, val_loader) using PyG DataLoader

Patient-level helpers (MIL training)
-------------------------------------
PatientDataset               — groups graphs by patient_id, one item per patient
patient_collate              — collate function: returns (all_graphs, labels, slide_counts)
compute_patient_class_weights — balanced weights based on patient-level label counts
make_patient_loaders          — (train_loader, val_loader) using torch DataLoader + patient_collate
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# ── Slide-level ────────────────────────────────────────────────────────────────

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
    print(f"[INFO] Slide class weights — N0: {w[0]:.4f}  N1: {w[1]:.4f}  "
          f"(counts N0={int(counts[0])}, N1={int(counts[1])})")
    return w


def make_weighted_sampler(graphs: list[Data]) -> WeightedRandomSampler:
    """WeightedRandomSampler so every mini-batch is roughly class-balanced."""
    labels  = np.array([g.y.item() for g in graphs])
    counts  = np.bincount(labels, minlength=2).astype(float)
    class_w = torch.tensor(1.0 / np.where(counts == 0, 1.0, counts), dtype=torch.float32)
    sample_w = class_w[labels]
    return WeightedRandomSampler(
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


# ── Patient-level (MIL) ────────────────────────────────────────────────────────

class PatientDataset(Dataset):
    """Groups slide graphs by patient_id.  One item = all slides of one patient."""

    def __init__(self, graphs: list[Data]):
        buckets: dict[str, list[Data]] = defaultdict(list)
        for g in graphs:
            buckets[g.patient_id].append(g)
        self.patients: list[list[Data]] = list(buckets.values())

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> list[Data]:
        return self.patients[idx]


def patient_collate(
    batch: list[list[Data]],
) -> tuple[list[Data], torch.Tensor, list[int]]:
    """Collate a batch of patient graph-lists.

    Returns
    -------
    all_graphs   : flat list of Data objects (all slides from all patients)
    labels       : LongTensor (n_patients,) — patient-level labels
    slide_counts : list[int]  — how many slides belong to each patient
                   (use to slice all_graphs back into per-patient groups)
    """
    all_graphs: list[Data] = []
    labels:     list[int]  = []
    slide_counts: list[int] = []
    for patient_graphs in batch:
        all_graphs.extend(patient_graphs)
        labels.append(patient_graphs[0].y.item())   # all slides share the patient label
        slide_counts.append(len(patient_graphs))
    return all_graphs, torch.tensor(labels, dtype=torch.long), slide_counts


def compute_patient_class_weights(graphs: list[Data]) -> torch.Tensor:
    """Balanced class weights based on *patient-level* label distribution.

    Collapses to one label per patient before computing the balanced formula.
    """
    patient_labels: dict[str, int] = {}
    for g in graphs:
        patient_labels[g.patient_id] = g.y.item()
    labels = np.array(list(patient_labels.values()))
    counts = np.bincount(labels, minlength=2).astype(float)
    n_total, n_classes = len(labels), 2
    weights = n_total / (n_classes * np.where(counts == 0, 1.0, counts))
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Patient class weights — N0: {w[0]:.4f}  N1: {w[1]:.4f}  "
          f"(patients N0={int(counts[0])}, N1={int(counts[1])})")
    return w


def make_patient_loaders(
    train_graphs: list[Data],
    val_graphs:   list[Data],
    batch_size:   int,
) -> tuple[TorchDataLoader, TorchDataLoader]:
    """Return patient-level (train_loader, val_loader) using patient_collate.

    The training loader uses a WeightedRandomSampler at patient level to
    ensure class-balanced patient mini-batches.
    """
    train_ds = PatientDataset(train_graphs)
    val_ds   = PatientDataset(val_graphs)

    # Patient-level balanced sampler
    pat_labels  = np.array([pg[0].y.item() for pg in train_ds.patients])
    counts      = np.bincount(pat_labels, minlength=2).astype(float)
    class_w     = torch.tensor(1.0 / np.where(counts == 0, 1.0, counts), dtype=torch.float32)
    sample_w    = class_w[pat_labels]
    sampler     = WeightedRandomSampler(
        weights=sample_w, num_samples=len(sample_w), replacement=True
    )

    train_loader = TorchDataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=patient_collate,
    )
    val_loader = TorchDataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=patient_collate,
    )
    return train_loader, val_loader
