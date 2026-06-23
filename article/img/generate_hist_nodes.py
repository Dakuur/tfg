#!/usr/bin/env python3
"""
Genera article/img/hist_nodes.pdf: histograma del nombre de nodes per secció
(grafs de la partició trainval + test), separat per classe N0/N1.

Ús:
    python article/generate_hist_nodes.py
    python article/generate_hist_nodes.py --graphs ~/outputs/graphs/per-section
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_IMG    = SCRIPT_DIR / "hist_nodes.pdf"
DEFAULT_GRAPHS = Path.home() / "outputs" / "graphs" / "per-section"


def collect_sizes(graphs_dir: Path):
    n0, n1 = [], []
    for split in ("trainval", "test"):
        split_dir = graphs_dir / split
        if not split_dir.exists():
            continue
        for pt in sorted(split_dir.glob("*.pt")):
            try:
                d = torch.load(pt, weights_only=False, map_location="cpu")
                label = int(d.y.item()) if hasattr(d, "y") else -1
                n_nodes = int(d.num_nodes)
                if label == 0:
                    n0.append(n_nodes)
                elif label == 1:
                    n1.append(n_nodes)
            except Exception:
                pass
    return np.array(n0, dtype=int), np.array(n1, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, default=DEFAULT_GRAPHS)
    ap.add_argument("--out",    type=Path, default=OUT_IMG)
    args = ap.parse_args()

    if not args.graphs.exists():
        raise FileNotFoundError(f"No s'ha trobat {args.graphs}")

    n0, n1 = collect_sizes(args.graphs)
    all_nodes = np.concatenate([n0, n1])
    print(f"N0: {len(n0)} seccions  |  N1: {len(n1)} seccions  |  Total: {len(all_nodes)}")
    print(f"Nodes — min:{all_nodes.min()}  med:{np.median(all_nodes):.0f}"
          f"  p75:{np.percentile(all_nodes,75):.0f}"
          f"  p95:{np.percentile(all_nodes,95):.0f}  max:{all_nodes.max()}")

    bins = np.arange(0, all_nodes.max() + 51, 50)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    ax.hist(n0, bins=bins, color="tab:blue",   alpha=0.65, label=f"N0  (n = {len(n0)})")
    ax.hist(n1, bins=bins, color="tab:orange", alpha=0.65, label=f"N1  (n = {len(n1)})")

    # Línies de mediana
    med0 = float(np.median(n0)) if len(n0) else 0
    med1 = float(np.median(n1)) if len(n1) else 0
    ax.axvline(med0, color="tab:blue",   lw=1.5, ls="--", alpha=0.85,
               label=f"Mediana N0 = {med0:.0f} nodes")
    ax.axvline(med1, color="tab:orange", lw=1.5, ls="--", alpha=0.85,
               label=f"Mediana N1 = {med1:.0f} nodes")

    ax.set_xlabel("Nombre de nodes per secció (patches)", fontsize=10)
    ax.set_ylabel("Nombre de seccions", fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Guardat: {args.out}")


if __name__ == "__main__":
    main()
