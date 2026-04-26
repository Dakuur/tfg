#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_search.py — Grid search + K-fold sobre grafs .pt precomputats.

Execució:
    python grid_search.py                              # usa configs/grid_gat.yaml
    python grid_search.py --config configs/grid_gat.yaml

El grid search genera combinacions vàlides a partir del config YAML:
  - gat-baseline × pooling × MIL × graph_type × lr
  - gat-diffpool  ×          MIL × graph_type × lr  (pooling fix "diff")
  - mega-graph   → MIL fix "mean" (cada .pt ja és un pacient complet)

Equivalent a PipelineGAT.py del repositori PT1Diagnosis: mateixa lògica
d'entrenament, mètriques i report. Resultats idèntics amb la mateixa seed i dades.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Batch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.model import GATClassifier  # noqa: E402


# ─── Seeds ────────────────────────────────────────────────────────────────────

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ─── Config i grid search ─────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(cfg: dict) -> list:
    """Genera les combinacions vàlides del grid search."""
    g = cfg["grid"]
    combos = []
    for model_arch in g["model_arch"]:
        for graph_type in g["graph_type"]:
            for lr in g["lr"]:
                # pooling del graf (readout): "diff" fix per gat-diffpool
                poolings = g["pooling"] if model_arch == "gat-baseline" else ["diff"]
                for pooling in poolings:
                    # MIL (agregació patient-level): "mean" fix per mega-graph
                    mils = g["mil"] if graph_type == "per-slide" else ["mean"]
                    for mil in mils:
                        combos.append({
                            "model_arch": model_arch,
                            "graph_type": graph_type,
                            "lr":         lr,
                            "pooling":    pooling,
                            "mil":        mil,
                        })
    return combos


def combo_name(c: dict) -> str:
    return (f"{c['model_arch']}|{c['graph_type']}|"
            f"pool={c['pooling']}|mil={c['mil']}|lr={c['lr']:.0e}")


# ─── Càrrega de dades ─────────────────────────────────────────────────────────

def load_patient_data(graph_type: str, cfg: dict):
    """Retorna (patient_label, patient_files) per al graph_type indicat."""
    paths = cfg["paths"]
    d = Path(paths["mega_graphs_dir"] if graph_type == "mega" else paths["slide_graphs_dir"])
    all_files = sorted(d.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {d}")

    patient_label, patient_files = {}, {}
    for f in all_files:
        data  = torch.load(f, weights_only=False)
        pid   = str(data.patient_id)
        label = int(data.y.item())
        patient_label[pid] = label
        patient_files.setdefault(pid, []).append(f)

    return patient_label, patient_files


# ─── Build patient_batch tensor ───────────────────────────────────────────────

def build_patient_batch(batch_pats, patient_files):
    """Carrega les slides d'una llista de pacients i retorna un Batch PyG.

    Returns:
        data          : Batch — totes les slides concatenades
        patient_batch : (n_slides,) LongTensor — índex de pacient per slide
        labels        : (n_patients,) LongTensor — etiqueta per pacient
    """
    graphs, p_batch, labels = [], [], []
    for j, (pid, label) in enumerate(batch_pats):
        for f in patient_files[pid]:
            graphs.append(torch.load(f, weights_only=False))
            p_batch.append(j)
        labels.append(label)
    data          = Batch.from_data_list(graphs)
    patient_batch = torch.tensor(p_batch, dtype=torch.long)
    labels_tensor = torch.tensor(labels,  dtype=torch.long)
    return data, patient_batch, labels_tensor


# ─── Loops ────────────────────────────────────────────────────────────────────

def train_loop(pat_files, pat_labels, model, loss_fn, optimizer, device, cfg, graph_type):
    tr   = cfg["training"]
    ppb  = tr["mega_patients_per_batch"] if graph_type == "mega" else tr["patients_per_batch"]
    mini = tr["minibatch_size"]

    model.train().to(device)
    loss_fn.to(device)
    patients = list(pat_files.keys())

    for _ in range(tr["epochs"]):
        random.shuffle(patients)
        optimizer.zero_grad()
        accum = 0

        for i in range(0, len(patients), ppb):
            batch_pats = [(p, pat_labels[p]) for p in patients[i:i + ppb]]
            data, patient_batch, labels = build_patient_batch(batch_pats, pat_files)

            data          = data.to(device)
            patient_batch = patient_batch.to(device)
            labels        = labels.to(device)

            output = model(data.x, data.edge_index, data.batch, patient_batch)
            loss   = loss_fn(output, labels) / mini
            if hasattr(model, "aux_loss"):
                aux = model.aux_loss
                if aux.grad_fn is not None:
                    loss = loss + aux / mini
            loss.backward()

            accum += 1
            if accum >= mini:
                optimizer.step()
                optimizer.zero_grad()
                accum = 0

        if accum > 0:
            optimizer.step()
            optimizer.zero_grad()


def val_loop(pat_files, pat_labels, model, device, cfg, graph_type):
    tr  = cfg["training"]
    ppb = tr["mega_patients_per_batch"] if graph_type == "mega" else tr["patients_per_batch"]

    model.eval().to(device)
    patients = list(pat_files.keys())
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for i in range(0, len(patients), ppb):
            batch_pats = [(p, pat_labels[p]) for p in patients[i:i + ppb]]
            data, patient_batch, labels = build_patient_batch(batch_pats, pat_files)

            data          = data.to(device)
            patient_batch = patient_batch.to(device)

            logits = model(data.x, data.edge_index, data.batch, patient_batch)
            probs  = F.softmax(logits, dim=1).cpu()
            p_n1   = probs[:, 1].tolist()

            y_true.extend(labels.tolist())
            y_scores.extend(p_n1)
            y_pred.extend([int(p > 0.5) for p in p_n1])

    return y_true, y_pred, y_scores


# ─── K-Fold ───────────────────────────────────────────────────────────────────

METRIC_KEYS = ["auc", "sensitivity", "specificity", "ppv", "npv", "f1_macro", "balanced_acc"]


def run_kfold(combo, patient_label, patient_files, cfg, device) -> dict:
    tr = cfg["training"]
    mc = cfg["model"]

    patient_list = list(patient_label.keys())
    label_list   = [patient_label[p] for p in patient_list]
    n0 = label_list.count(0)
    n1 = label_list.count(1)
    weights = torch.tensor([len(label_list) / (2 * n0), len(label_list) / (2 * n1)])

    model_kwargs = dict(
        in_channels         = mc["in_channels"],
        hidden              = mc["hidden"],
        heads               = mc["heads"],
        dropout             = mc["dropout"],
        pooling             = combo["pooling"],
        diff_clusters       = mc["diff_clusters"],
        patient_aggregation = combo["mil"],
    )

    skf     = StratifiedKFold(tr["n_folds"], shuffle=True, random_state=0)
    metrics = {k: [] for k in METRIC_KEYS}

    for fold_num, (trf, vaf) in enumerate(skf.split(patient_list, label_list), 0):
        tr_pats = np.array(patient_list)[trf]
        va_pats = np.array(patient_list)[vaf]
        tr_labs = pd.Series(label_list)[trf]
        va_labs = pd.Series(label_list)[vaf]

        print(f"  Fold {fold_num+1}/{tr['n_folds']}  "
              f"TR:{len(tr_pats)}(N0={tr_labs.value_counts().get(0,0)} "
              f"N1={tr_labs.value_counts().get(1,0)})  "
              f"VA:{len(va_pats)}(N0={va_labs.value_counts().get(0,0)} "
              f"N1={va_labs.value_counts().get(1,0)})")

        tr_pat_files  = {p: patient_files[p] for p in tr_pats}
        tr_pat_labels = {p: patient_label[p] for p in tr_pats}
        va_pat_files  = {p: patient_files[p] for p in va_pats}
        va_pat_labels = {p: patient_label[p] for p in va_pats}

        fix_seeds(fold_num)
        model     = GATClassifier(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=combo["lr"])
        loss_fn   = torch.nn.CrossEntropyLoss(weight=weights)

        train_loop(tr_pat_files, tr_pat_labels, model, loss_fn, optimizer, device, cfg, combo["graph_type"])
        y_true, y_pred, y_scores = val_loop(va_pat_files, va_pat_labels, model, device, cfg, combo["graph_type"])

        sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics["auc"].append(roc_auc_score(y_true, y_scores))
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["ppv"].append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics["npv"].append(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics["f1_macro"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["balanced_acc"].append((sensitivity + specificity) / 2)

        print(f"    AUC={metrics['auc'][-1]:.4f}  "
              f"Sens={sensitivity:.4f}  Spec={specificity:.4f}")

    return metrics


# ─── Report ───────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "auc":          "AUC          ",
    "sensitivity":  "Sensitivity  ",
    "specificity":  "Specificity  ",
    "ppv":          "PPV          ",
    "npv":          "NPV          ",
    "f1_macro":     "F1-macro     ",
    "balanced_acc": "Bal. Accuracy",
}


def save_report(all_results: list, cfg: dict, output_dir: str = "outputs"):
    """Guarda un CSV per-fold i un resum .txt a output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"grid_search_{ts}"

    # ── CSV per-fold ──────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        c       = r["combo"]
        m       = r["metrics"]
        n_folds = len(m["auc"])
        for fold in range(n_folds):
            row = {
                "combo":      r["name"],
                "model_arch": c["model_arch"],
                "graph_type": c["graph_type"],
                "pooling":    c["pooling"],
                "mil":        c["mil"],
                "lr":         c["lr"],
                "fold":       fold + 1,
            }
            for k in METRIC_KEYS:
                row[k] = round(m[k][fold], 6)
            rows.append(row)

    csv_path = base.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Text summary ──────────────────────────────────────────────────────────
    txt_path = base.with_name(base.name + "_summary.txt")
    W = 70
    lines = []
    lines.append("=" * W)
    lines.append(f"GRID SEARCH REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)
    mc = cfg["model"]
    tr = cfg["training"]
    lines.append(f"Folds   : {tr['n_folds']}    Epochs: {tr['epochs']}")
    lines.append(f"Model   : hidden={mc['hidden']}  heads={mc['heads']}  "
                 f"dropout={mc['dropout']}  diff_clusters={mc['diff_clusters']}")
    lines.append(f"Configs : {len(all_results)}")
    lines.append("")
    lines.append("Resultats (mean ± std — folds individuals entre [])")
    lines.append("-" * W)

    sorted_results = sorted(all_results,
                            key=lambda r: np.mean(r["metrics"]["auc"]), reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        m = r["metrics"]
        lines.append(f"{rank:2d}. {r['name']}")
        for k in METRIC_KEYS:
            vals_str = "  ".join(f"[{v:.4f}]" for v in m[k])
            lines.append(f"     {METRIC_LABELS[k]}: "
                         f"{np.mean(m[k]):.4f} ± {np.std(m[k]):.4f}   {vals_str}")
        lines.append("")

    lines.append("=" * W)
    txt_path.write_text("\n".join(lines))

    print(f"\nReport guardat a:")
    print(f"  CSV    : {csv_path}")
    print(f"  Resum  : {txt_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "grid_gat.yaml"))
    args = parser.parse_args()

    cfg    = load_config(args.config)
    combos = build_grid(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seeds(42)

    print(f"Grid search: {len(combos)} combinacions × {cfg['training']['n_folds']} folds  |  Device: {device}")

    # Pre-càrrega de dades per graph_type (evita recarregar a cada combo)
    graph_types = sorted(set(c["graph_type"] for c in combos))
    all_patient_data = {}
    for gt in graph_types:
        pat_label, pat_files = load_patient_data(gt, cfg)
        n0 = list(pat_label.values()).count(0)
        n1 = list(pat_label.values()).count(1)
        print(f"[{gt}] {len(pat_label)} pacients — N0={n0} N1={n1}")
        all_patient_data[gt] = (pat_label, pat_files)

    all_results = []

    for i, combo in enumerate(combos, 1):
        name = combo_name(combo)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(combos)}] {name}")
        print(f"{'='*60}")

        pat_label, pat_files = all_patient_data[combo["graph_type"]]
        metrics = run_kfold(combo, pat_label, pat_files, cfg, device)

        m = metrics
        print(f"  AUC={np.mean(m['auc']):.4f}±{np.std(m['auc']):.4f}  "
              f"Sens={np.mean(m['sensitivity']):.4f}  "
              f"Spec={np.mean(m['specificity']):.4f}  "
              f"BalAcc={np.mean(m['balanced_acc']):.4f}")
        all_results.append({"name": name, "combo": combo, "metrics": metrics})

    # ─── Resum per consola ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUM GRID SEARCH  (ordenat per AUC)")
    print(f"{'='*60}")
    all_results.sort(key=lambda r: np.mean(r["metrics"]["auc"]), reverse=True)
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['name']}")
        print(f"    AUC={np.mean(m['auc']):.4f}±{np.std(m['auc']):.4f}  "
              f"Sens={np.mean(m['sensitivity']):.4f}  "
              f"Spec={np.mean(m['specificity']):.4f}  "
              f"F1-mac={np.mean(m['f1_macro']):.4f}")
    print(f"{'='*60}")

    save_report(all_results, cfg)
