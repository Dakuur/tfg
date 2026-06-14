#!/usr/bin/env python3
"""
Genera article/img/roc_top5.pdf amb:
  - Corba ROC + punt operatiu del GAT Baseline (model seleccionat, Grid #1)
  - Corba ROC + punt operatiu del GAT+DiffPool (millor del sweep, #199)
  - Punts de guies clíniques: JSCCR, NCCN, ESMO
  - Punt LightGBM (Piao et al. 2023)
  - Punt Song et al. 2024

Ús (des del directori arrel del repo tfg/):
    python article/generate_roc.py
    python article/generate_roc.py --outputs ~/outputs
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_IMG    = SCRIPT_DIR / "img" / "roc_top5.pdf"

# ── Configuració dels models GAT ────────────────────────────────────────────
# name: (dir_configs, label_llegenda, color, t_star, linestyle)
GAT_MODELS = [
    {
        "config": "grid1",
        "label":  "GAT Baseline",
        "color":  "tab:blue",
        "t_star": 0.133,
        "lw":     2.2,
        "ls":     "-",
        "zorder": 4,
    },
    {
        "config": "sweep_diffpool199",
        "label":  "GAT+DiffPool",
        "color":  "tab:orange",
        "t_star": 0.303,
        "lw":     1.8,
        "ls":     "--",
        "zorder": 3,
    },
]

# ── Punts de referència (sens, spec) ────────────────────────────────────────
REFERENCE_POINTS = [
    # Guies clíniques — quadrats
    {"label": "JSCCR",    "sens": 1.000, "spec": 0.190, "marker": "s", "color": "tab:purple",  "ms": 90},
    {"label": "NCCN",     "sens": 0.980, "spec": 0.520, "marker": "s", "color": "tab:olive",   "ms": 90},
    {"label": "ESMO",     "sens": 0.980, "spec": 0.500, "marker": "s", "color": "tab:cyan",    "ms": 90},
    # IA sobre variables clíniques — diamant
    {"label": "LightGBM (Piao et al.)", "sens": 1.000, "spec": 0.858, "marker": "D", "color": "tab:green",  "ms": 90},
    # IA sobre WSI — triangle
    {"label": "Song et al.",            "sens": 0.929, "spec": 0.576, "marker": "^", "color": "tab:red",    "ms": 90},
]


def load_roc(configs_dir: Path, config_name: str):
    """Carrega probs i labels de test d'un config i retorna (fpr, tpr, auc)."""
    from sklearn.metrics import roc_auc_score
    d = configs_dir / f"config_{config_name}"
    probs  = np.load(d / "test_probs.npy").astype(np.float64)
    labels = np.load(d / "test_labels.npy").astype(np.int64)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = float(roc_auc_score(labels, probs))
    return fpr, tpr, auc, probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path,
                    default=Path.home() / "outputs",
                    help="Directori outputs/ (default: ~/outputs)")
    ap.add_argument("--out", type=Path, default=OUT_IMG,
                    help=f"Fitxer de sortida (default: {OUT_IMG})")
    args = ap.parse_args()

    configs_dir = args.outputs / "sweep" / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"No s'ha trobat {configs_dir}")

    fig, ax = plt.subplots(figsize=(5.8, 5.5))

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, zorder=1)

    # Franja verda sens >= 95%
    ax.axhspan(0.95, 1.01, alpha=0.07, color="green", zorder=0)

    # ── Corbes ROC dels models GAT ──────────────────────────────────────────
    for m in GAT_MODELS:
        try:
            fpr, tpr, auc, probs, labels = load_roc(configs_dir, m["config"])
        except FileNotFoundError:
            print(f"  [!] Probs no trobats per a config '{m['config']}' — saltant")
            continue

        ax.plot(fpr, tpr,
                color=m["color"], lw=m["lw"], ls=m["ls"], zorder=m["zorder"],
                label=f"{m['label']}  (AUC = {auc:.3f})")

        # Punt operatiu al llindar t*
        from sklearn.metrics import roc_curve as _rc
        fpr_all, tpr_all, thr_all = _rc(labels, probs)
        # El punt més proper al t* a la corba ROC
        idx = np.argmin(np.abs(thr_all - m["t_star"]))
        ax.scatter(fpr_all[idx], tpr_all[idx],
                   color=m["color"], s=120, zorder=6,
                   edgecolors="black", linewidths=1.2)

    # ── Punts de referència ─────────────────────────────────────────────────
    for p in REFERENCE_POINTS:
        fpr_pt = 1.0 - p["spec"]
        tpr_pt = p["sens"]
        ax.scatter(fpr_pt, tpr_pt,
                   marker=p["marker"], s=p["ms"], color=p["color"], zorder=5,
                   edgecolors="black", linewidths=1.1,
                   label=p["label"])

    ax.set_xlabel("1 $-$ Especificitat (FPR)", fontsize=10)
    ax.set_ylabel("Sensibilitat (TPR)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92,
              edgecolor="gray", ncol=1)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Guardat: {args.out}")


if __name__ == "__main__":
    main()
