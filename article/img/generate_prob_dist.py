#!/usr/bin/env python3
"""
Genera article/img/prob_dist.pdf amb la distribució de P(N1) del model seleccionat
(Grid #1) sobre el test set, separada per etiqueta real (N0 / N1).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_IMG = SCRIPT_DIR / "prob_dist.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path.home() / "outputs")
    ap.add_argument("--out", type=Path, default=OUT_IMG)
    args = ap.parse_args()

    config_dir = args.outputs / "sweep" / "configs" / "config_grid1"
    probs  = np.load(config_dir / "test_probs.npy").astype(np.float64)
    labels = np.load(config_dir / "test_labels.npy").astype(np.int64)

    p_n0 = probs[labels == 0]
    p_n1 = probs[labels == 1]

    t_star = 0.133  # llindar operatiu del model seleccionat

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    bins = np.linspace(0, 1, 26)

    ax.hist(p_n0, bins=bins, alpha=0.55, color="steelblue",
            label=f"N0  (n={len(p_n0)})", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(p_n1, bins=bins, alpha=0.55, color="tomato",
            label=f"N1  (n={len(p_n1)})", density=True, edgecolor="white", linewidth=0.5)

    # KDE suavitzat
    x_grid = np.linspace(0, 1, 300)
    for vals, color in [(p_n0, "steelblue"), (p_n1, "tomato")]:
        if len(vals) > 3:
            kde = gaussian_kde(vals, bw_method=0.25)
            ax.plot(x_grid, kde(x_grid), color=color, lw=2.0)

    ax.axvline(t_star, color="black", linestyle="--", lw=1.5,
               label=f"$t^*={t_star}$ (T2, TPR$=1$)")
    ax.axvline(0.5, color="gray", linestyle=":", lw=1.3,
               label="$t=0{,}5$ (T1)")

    ax.set_xlabel("$P(\\mathrm{N1})$", fontsize=13)
    ax.set_ylabel("Densitat", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.22)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Guardat: {args.out}")
    print(f"N0: {len(p_n0)} pacients, N1: {len(p_n1)} pacients")
    print(f"t*={t_star}  →  correctes N0: {(p_n0 < t_star).sum()}/{len(p_n0)}, "
          f"N1: {(p_n1 >= t_star).sum()}/{len(p_n1)}")


if __name__ == "__main__":
    main()
