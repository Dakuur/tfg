#!/usr/bin/env python3
"""
Smoke test: real data pipeline + GPU training loop sanity check.

Uses the scripts/ pipeline end-to-end:
  wsi_io         → load NPZ embeddings and patient labels
  build_dataset  → build section index, construct Delaunay graph for a random section
  graph_utils    → shared Delaunay / edge_index utilities (used internally)

A random valid section is picked each run (fixed with --seed).
The goal is to confirm the full pipeline would work before a real training run.

Usage:
    python alive_test.py
    python alive_test.py --iam_path /mnt/iam --steps 20 --seed 42
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

# Add scripts/ to path so we can import shared utilities
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "scripts"))

from wsi_io import load_all_npz, load_labels, CLS_DIR_SUBPATH          # noqa: E402
from build_dataset import build_slide_index, build_graph_for_section    # noqa: E402


# ── model ──────────────────────────────────────────────────────────────────────

class GATClassifier(torch.nn.Module):
    """Two-layer GAT with global mean pooling for graph-level binary classification."""

    def __init__(self, in_channels: int, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.gat1       = GATConv(in_channels,   hidden, heads=heads, concat=True,  dropout=0.1)
        self.gat2       = GATConv(hidden * heads, hidden, heads=1,    concat=False, dropout=0.1)
        self.classifier = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


# ── data loading ───────────────────────────────────────────────────────────────

def load_random_patient_graph(iam_path: Path, seed: int | None) -> Data:
    """
    Build a Delaunay graph for a randomly sampled patient section.

    Pipeline:
      1. load_all_npz     — load all CLS embeddings from NPZ files
      2. load_labels      — load patient metastasis labels from Excel
      3. build_slide_index — filter and index valid sections (>= MIN_BAGS_PER_SECTION)
      4. random sample    — pick one section at random
      5. build_graph_for_section — construct the torch_geometric Data object
    """
    cls_dir = iam_path / CLS_DIR_SUBPATH
    if not cls_dir.is_dir():
        sys.exit(f"[ERROR] CLS directory not found: {cls_dir}")

    print(f"[INFO] Loading NPZ embeddings from {cls_dir} ...")
    df_npz = load_all_npz(cls_dir)

    print("[INFO] Loading labels ...")
    df_labels = load_labels(iam_path)

    section_index = build_slide_index(df_npz, df_labels)
    if section_index.empty:
        sys.exit("[ERROR] No valid sections found after filtering.")

    row = section_index.sample(n=1, random_state=seed).iloc[0]

    patient_id       = str(row["Patient_ID"])
    slide_id         = str(row["Slide"])
    section_id       = str(row["Section"])
    hospital         = str(row["Hospital"])
    label            = int(row["label"])
    metastasis_score = str(row["Metastasis_score"])

    print(f"\n[INFO] Sampled   : {hospital} / patient={patient_id} / slide={slide_id} / sec={section_id}")

    data = build_graph_for_section(
        patient_id=patient_id,
        slide_id=slide_id,
        section_id=section_id,
        hospital=hospital,
        label=label,
        metastasis_score=metastasis_score,
        df_npz=df_npz,
    )

    if data is None:
        sys.exit(
            f"[ERROR] Could not build graph for {hospital}/{patient_id}/{slide_id}/sec{section_id}.\n"
            "Try running again (a different section will be sampled) or pass --seed."
        )

    print(
        f"[INFO] Graph     : {data.x.shape[0]} nodes (bags)  |  "
        f"{data.edge_index.shape[1]} directed edges  |  "
        f"label={label}  ({'N0 — no metastasis' if label == 0 else 'N1 — metastasis'})"
    )
    print(f"[INFO] Features  : {data.x.shape[1]}-dim CLS embeddings per node")
    return data


# ── training loop ──────────────────────────────────────────────────────────────

def run_training_loop(data: Data, steps: int, device: torch.device) -> None:
    """
    Move the graph to device and run `steps` training iterations, printing
    loss and GPU memory usage at each step.
    """
    batch      = torch.zeros(data.num_nodes, dtype=torch.long)
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    batch      = batch.to(device)

    model     = GATClassifier(in_channels=x.shape[1], hidden=128, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Model    : GATClassifier  ({n_params:,} parameters)")
    print(f"[INFO] Device   : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU      : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nRunning {steps} training steps ...\n")

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(x, edge_index, batch)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_str = f"  GPU mem: {torch.cuda.memory_allocated(device) / 1e9:.3f} GB"
        else:
            mem_str = ""

        pred    = logits.argmax(dim=1).item()
        correct = "correct" if pred == y.item() else "wrong"
        print(f"  step {step:3d}/{steps}  loss={loss.item():.4f}  pred={pred}  ({correct}){mem_str}")

    print("\n[INFO] Smoke test passed — training loop completed successfully.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test: random real patient graph + GPU training loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path", default="/mnt/iam",
                   help="Dataset root")
    p.add_argument("--steps",    type=int, default=20,
                   help="Number of training steps")
    p.add_argument("--seed",     type=int, default=None,
                   help="Random seed for section sampling (omit for a new random pick each run)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python  : {sys.executable}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    print()

    data = load_random_patient_graph(Path(args.iam_path), seed=args.seed)
    run_training_loop(data, steps=args.steps, device=device)


if __name__ == "__main__":
    main()
