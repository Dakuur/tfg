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
        "lw":     2.2,
        "ls":     "-",
        "zorder": 4,
    },
    {
        "config": "sweep_diffpool199",
        "label":  "GAT+DiffPool",
        "color":  "tab:orange",
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
    """Carrega probs, labels, t* i punt operatiu real d'un config."""
    import json
    from sklearn.metrics import roc_auc_score
    d = configs_dir / f"config_{config_name}"
    probs  = np.load(d / "test_probs.npy").astype(np.float64)
    labels = np.load(d / "test_labels.npy").astype(np.int64)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = float(roc_auc_score(labels, probs))
    results = json.loads((d / "final_results.json").read_text())
    t_star = float(results["threshold_final"])
    # Sens/spec al t* real — llegits del JSON, no interpolats sobre la corba
    at_thr = results["test"]["at_threshold"]
    op_fpr = 1.0 - float(at_thr["spec"])
    op_tpr = float(at_thr["sens"])
    return fpr, tpr, auc, t_star, op_fpr, op_tpr


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

    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, zorder=1)

    # Franja verda sens >= 95%
    ax.axhspan(0.95, 1.01, alpha=0.07, color="green", zorder=0)

    # ── Corbes ROC dels models GAT ──────────────────────────────────────────
    for m in GAT_MODELS:
        try:
            fpr, tpr, auc, t_star, op_fpr, op_tpr = load_roc(configs_dir, m["config"])
        except FileNotFoundError:
            print(f"  [!] Probs no trobats per a config '{m['config']}' — saltant")
            continue

        ax.plot(fpr, tpr,
                color=m["color"], lw=m["lw"], ls=m["ls"], zorder=m["zorder"],
                label=f"{m['label']}  (AUC = {auc:.3f})")

        # Punt operatiu T2 (t*) llegit del JSON
        print(f"  {m['config']}: t*={t_star:.4f} → "
              f"sens={op_tpr:.3f}, spec={1-op_fpr:.3f}")
        ax.scatter(op_fpr, op_tpr,
                   color=m["color"], s=140, zorder=6,
                   edgecolors="black", linewidths=1.2)

        # Punt operatiu T1 (t=0.5) — només per al Baseline
        if m["config"] == "grid1":
            probs_t  = np.load(configs_dir / f"config_{m['config']}/test_probs.npy").astype(np.float64)
            labels_t = np.load(configs_dir / f"config_{m['config']}/test_labels.npy").astype(np.int64)
            pred_t1  = (probs_t >= 0.5).astype(int)
            tp_ = int(((pred_t1 == 1) & (labels_t == 1)).sum())
            fp_ = int(((pred_t1 == 1) & (labels_t == 0)).sum())
            fn_ = int(((pred_t1 == 0) & (labels_t == 1)).sum())
            tn_ = int(((pred_t1 == 0) & (labels_t == 0)).sum())
            sens_t1 = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
            spec_t1 = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0.0
            print(f"  {m['config']}: T1 t=0.5 → sens={sens_t1:.3f}, spec={spec_t1:.3f}")
            ax.scatter(1 - spec_t1, sens_t1,
                       color=m["color"], s=120, zorder=7,
                       marker="^", edgecolors="black", linewidths=1.2,
                       label=f"GAT Baseline T1 ($t\\!=\\!0{{,}}5$)")

    # ── Punts de referència ─────────────────────────────────────────────────
    for p in REFERENCE_POINTS:
        fpr_pt = 1.0 - p["spec"]
        tpr_pt = p["sens"]
        ax.scatter(fpr_pt, tpr_pt,
                   marker=p["marker"], s=p["ms"], color=p["color"], zorder=5,
                   edgecolors="black", linewidths=1.1,
                   label=p["label"])

    ax.set_xlabel("1 $-$ Especificitat (FPR)", fontsize=15)
    ax.set_ylabel("Sensibilitat (TPR)", fontsize=15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.92,
              edgecolor="gray", ncol=1)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=13)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Guardat: {args.out}")


if __name__ == "__main__":
    main()
