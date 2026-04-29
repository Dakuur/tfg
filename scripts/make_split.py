#!/usr/bin/env python3
"""
make_split.py — Genera split.json (train/val/test) des dels grafs existents.

Aplica exactament el mateix split 70/15/15 estratificat per pacient (seed=42)
que usa train.py i PT1Diagnosis/PipelineGAT.py. Un cop generat, el frontend
mostrarà únicament els pacients del test set per evitar data leakage.

Ús:
    python scripts/make_split.py
    python scripts/make_split.py --graphs_dir ~/outputs/graphs/per-slide
    python scripts/make_split.py --dry_run     # mostra comptes sense escriure
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split


def make_split(graphs_dir: Path, p_test: float = 0.15, p_val: float = 0.15,
               dry_run: bool = False) -> None:
    pts = sorted(graphs_dir.glob("*.pt"))
    if not pts:
        sys.exit(f"No s'han trobat fitxers .pt a {graphs_dir}")

    print(f"Carregant {len(pts)} grafs per extreure patient_id i label…")
    pat: dict[str, int] = {}  # patient_id → label
    for pt in pts:
        try:
            g = torch.load(pt, weights_only=False, map_location="cpu")
            pid   = str(getattr(g, "patient_id", pt.stem))
            label = int(g.y.item()) if hasattr(g, "y") and g.y is not None else -1
            if pid not in pat:
                pat[pid] = label
        except Exception as e:
            print(f"  SKIP {pt.name}: {e}")

    pids   = list(pat.keys())
    labels = [pat[p] for p in pids]
    print(f"Pacients únics: {len(pids)}  "
          f"N0={labels.count(0)}  N1={labels.count(1)}")

    tr_va, te = train_test_split(
        pids, test_size=p_test, stratify=labels, random_state=42,
    )
    labels_tr_va = [pat[p] for p in tr_va]
    val_frac     = p_val / (1.0 - p_test)
    tr, va = train_test_split(
        tr_va, test_size=val_frac, stratify=labels_tr_va, random_state=42,
    )

    print(f"\nSplit  70/15/15 (seed=42):")
    print(f"  Train : {len(tr):3d} pacients  "
          f"N0={sum(pat[p]==0 for p in tr)}  N1={sum(pat[p]==1 for p in tr)}")
    print(f"  Val   : {len(va):3d} pacients  "
          f"N0={sum(pat[p]==0 for p in va)}  N1={sum(pat[p]==1 for p in va)}")
    print(f"  Test  : {len(te):3d} pacients  "
          f"N0={sum(pat[p]==0 for p in te)}  N1={sum(pat[p]==1 for p in te)}")

    split_json = graphs_dir / "split.json"
    if dry_run:
        print(f"\n[dry-run] No s'ha escrit res. Hauria escrit: {split_json}")
        return

    with open(split_json, "w") as f:
        json.dump({"train": tr, "val": va, "test": te}, f, indent=2)
    print(f"\n✓ split.json guardat a {split_json}")
    print("  Reinicia el frontend perquè carregui el nou split.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs_dir", default=str(Path.home() / "outputs/graphs/per-slide"),
                   help="Directori amb els .pt (default: ~/outputs/graphs/per-slide)")
    p.add_argument("--split_test", type=float, default=0.15)
    p.add_argument("--split_val",  type=float, default=0.15)
    p.add_argument("--dry_run",    action="store_true")
    args = p.parse_args()

    graphs_dir = Path(args.graphs_dir).expanduser()
    if not graphs_dir.exists():
        sys.exit(f"Directori no trobat: {graphs_dir}")

    make_split(graphs_dir, args.split_test, args.split_val, args.dry_run)


if __name__ == "__main__":
    main()
