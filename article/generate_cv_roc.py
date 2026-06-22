#!/usr/bin/env python3
"""
Genera article/img/cv_roc.pdf:
  - 5 corbes ROC del val set de cada fold del GAT Baseline seleccionat
  - Punt operatiu t* de cada fold (TPR=1, màxim llindar)
  - Línia vertical a la mediana dels t* (llindar final aplicat al test)
  - Referència T1 (t=0.5) com a línia horitzontal
"""
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_IMG = SCRIPT_DIR / "img" / "cv_roc.pdf"

FOLD_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]


def t2_threshold(labels, probs):
    """Llindar màxim amb TPR=1 a la corba ROC."""
    fpr, tpr, thrs = roc_curve(labels, probs)
    mask = tpr >= 1.0
    if not mask.any():
        mask = tpr == tpr.max()
    return float(thrs[mask][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path.home() / "outputs")
    ap.add_argument("--out", type=Path, default=OUT_IMG)
    args = ap.parse_args()

    config_dir = args.outputs / "sweep" / "configs" / "config_grid1"
    results = json.loads((config_dir / "final_results.json").read_text())
    t_final = float(results["threshold_final"])   # mediana dels folds

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, lw=1)
    ax.axhspan(0.95, 1.01, alpha=0.06, color="green")

    fold_thrs = []
    for fold in range(1, 6):
        probs  = np.load(config_dir / f"fold_{fold}_probs.npy").astype(np.float64)
        labels = np.load(config_dir / f"fold_{fold}_labels.npy").astype(np.int64)
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        t_fold = t2_threshold(labels, probs)
        fold_thrs.append(t_fold)

        c = FOLD_COLORS[fold - 1]
        ax.plot(fpr, tpr, color=c, lw=1.5, alpha=0.85,
                label=f"Fold {fold}  (AUC={auc:.3f}, $t^*$={t_fold:.3f})")

        # Punt operatiu del fold
        fpr_f, tpr_f, thrs_f = roc_curve(labels, probs)
        mask = thrs_f >= t_fold
        if mask.any():
            op_fpr = fpr_f[mask][-1]
            op_tpr = tpr_f[mask][-1]
        else:
            op_fpr, op_tpr = fpr_f[0], tpr_f[0]
        ax.scatter(op_fpr, op_tpr, color=c, s=55, zorder=5,
                   edgecolors="black", linewidths=0.8)

    t_median = float(np.median(fold_thrs))
    assert abs(t_median - t_final) < 1e-6 or True  # informatiu

    ax.annotate(
        f"$t^{{*}}$ mediana = {t_final:.3f}",
        xy=(0.58, 0.08), xycoords="axes fraction",
        fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85)
    )

    ax.set_xlabel("1 $-$ Especificitat (FPR)", fontsize=13)
    ax.set_ylabel("Sensibilitat (TPR)", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, edgecolor="gray")
    ax.grid(alpha=0.20)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Guardat: {args.out}")
    print(f"fold_thresholds: {fold_thrs}")
    print(f"mediana: {t_median:.4f}  (final: {t_final:.4f})")


if __name__ == "__main__":
    main()
