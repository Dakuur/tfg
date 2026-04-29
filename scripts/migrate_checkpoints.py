#!/usr/bin/env python3
"""
migrate_checkpoints.py — Migra checkpoints antics (raw state_dict) al format complet.

Per a cada .pt a ~/outputs/checkpoints/ que sigui un raw state_dict (sense clau
"model"), el re-guarda com a wrapper dict i genera el .yaml de config corresponent.

Format de sortida (compatible amb frontend/main.py i train.py):
  .pt  → {"epoch": ..., "val_auc": ..., "val_f1_macro": ..., "model": sd, "optimizer": {}}
  .yaml → {model: {pooling, hidden, heads, dropout, diff_clusters, in_channels},
            training: {aggregation, lr}}

Nom dels fitxers (format PipelineGAT.py):
  {arch}_{graph_type}_pool-{pooling}_mil-{mil}_lr{lr}_best.pt
  Exemple: gat-baseline_per-slide_pool-attention_mil-noisy_or_lr1e-04_best.pt

Ús:
    python scripts/migrate_checkpoints.py
    python scripts/migrate_checkpoints.py --dry-run   # mostra canvis sense escriure
    python scripts/migrate_checkpoints.py --dir /ruta/a/checkpoints
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import yaml


CKPT_DIR = Path.home() / "outputs" / "checkpoints"

_LR_RE = re.compile(r"lr(\d+)e[_-](\d+)", re.IGNORECASE)


def _parse_lr(token: str) -> float | None:
    """Parseja 'lr1e-03' → 0.001, 'lr1e-04' → 0.0001, etc."""
    m = _LR_RE.search(token)
    if m:
        return float(f"{m.group(1)}e-{m.group(2)}")
    # Fallback: busca el número directament
    try:
        num = re.search(r"lr([0-9.e+-]+)", token, re.IGNORECASE)
        return float(num.group(1)) if num else None
    except Exception:
        return None


def _parse_filename(stem: str) -> dict | None:
    """
    Extreu pooling, aggregation (mil) i lr del nom del fitxer.

    Format esperat: {arch}_{graph_type}_pool-{pooling}_mil-{mil}_lr{lr}_best
    Exemple: gat-baseline_per-slide_pool-attention_mil-noisy_or_lr1e-04_best
    """
    pool_m = re.search(r"pool-([a-z_]+?)(?=_mil-)", stem)
    mil_m  = re.search(r"mil-([a-z_]+?)(?=_lr)", stem)
    lr_m   = re.search(r"(lr\d+e[-_]\d+)", stem, re.IGNORECASE)

    if not (pool_m and mil_m):
        return None

    return {
        "pooling":     pool_m.group(1),
        "aggregation": mil_m.group(1),
        "lr":          _parse_lr(lr_m.group(1)) if lr_m else None,
    }


def _infer_arch(sd: dict) -> dict:
    """Infereix hidden, heads i in_channels des del state_dict."""
    hidden = sd["bn3.weight"].shape[0]
    heads  = sd["bn1.weight"].shape[0] // hidden

    in_channels = None
    for key, dim in [("conv1.lin.weight", 1), ("conv1.lin_src.weight", 1),
                     ("conv1.weight", 1), ("conv1.lin_src.weight", 0)]:
        if key in sd:
            in_channels = sd[key].shape[dim]
            break

    diff_clusters = None
    if "diff_pool1.assign_net.2.weight" in sd:
        diff_clusters = sd["diff_pool1.assign_net.2.weight"].shape[0]

    return {
        "hidden":        hidden,
        "heads":         heads,
        "in_channels":   in_channels or 1536,
        "diff_clusters": diff_clusters or 10,
    }


def migrate(ckpt_dir: Path, dry_run: bool = False) -> None:
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"No s'han trobat fitxers .pt a {ckpt_dir}")
        return

    migrated = skipped = already_ok = errors = 0

    for pt_path in pts:
        yaml_path = pt_path.with_suffix(".yaml")

        try:
            raw = torch.load(pt_path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"  ERROR carregant {pt_path.name}: {e}")
            errors += 1
            continue

        # Comprovem si ja té el format nou
        if isinstance(raw, dict) and "model" in raw:
            # Checkpoint ja en format wrapper — comprova si li falta el YAML
            if yaml_path.exists():
                already_ok += 1
                continue
            # Té wrapper però no YAML — genera el YAML
            sd = raw["model"]
        elif isinstance(raw, dict) and not any(k in raw for k in ("model", "epoch")):
            # Raw state_dict
            sd = raw
        else:
            print(f"  SKIP (format desconegut): {pt_path.name}")
            skipped += 1
            continue

        params = _parse_filename(pt_path.stem)
        if params is None:
            print(f"  SKIP (nom no parsejable): {pt_path.name}")
            skipped += 1
            continue

        arch   = _infer_arch(sd)
        pooling     = params["pooling"]
        aggregation = params["aggregation"]
        lr          = params["lr"]

        yaml_cfg = {
            "model": {
                "in_channels":   arch["in_channels"],
                "hidden":        arch["hidden"],
                "heads":         arch["heads"],
                "dropout":       0.3,
                "diff_clusters": arch["diff_clusters"],
                "pooling":       pooling,
            },
            "training": {
                "aggregation": aggregation,
                **({"lr": lr} if lr is not None else {}),
            },
        }

        # Wrapper per al .pt si era raw state_dict
        need_pt_update = not (isinstance(raw, dict) and "model" in raw)
        wrapper = {
            "epoch":        raw.get("epoch")        if isinstance(raw, dict) else None,
            "val_auc":      raw.get("val_auc")      if isinstance(raw, dict) else None,
            "val_f1_macro": raw.get("val_f1_macro") if isinstance(raw, dict) else None,
            "model":        sd,
            "optimizer":    raw.get("optimizer", {}) if isinstance(raw, dict) else {},
        }

        print(f"  {'[DRY]' if dry_run else 'OK   '} {pt_path.name}")
        print(f"         pooling={pooling}  agg={aggregation}  "
              f"hidden={arch['hidden']}  heads={arch['heads']}"
              + (f"  lr={lr}" if lr else ""))

        if not dry_run:
            if need_pt_update:
                torch.save(wrapper, pt_path)
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_cfg, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)

        migrated += 1

    print(f"\nResum: {migrated} migrats  |  {already_ok} ja correctes  "
          f"|  {skipped} omesos  |  {errors} errors")
    if dry_run:
        print("(mode dry-run: no s'ha escrit res)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir",     default=str(CKPT_DIR),
                   help="Directori de checkpoints (default: ~/outputs/checkpoints)")
    p.add_argument("--dry-run", action="store_true",
                   help="Mostra el que es faria sense escriure res")
    args = p.parse_args()

    ckpt_dir = Path(args.dir).expanduser()
    if not ckpt_dir.exists():
        sys.exit(f"Directori no trobat: {ckpt_dir}")

    print(f"Directori : {ckpt_dir}")
    print(f"Mode      : {'dry-run' if args.dry_run else 'escriptura'}\n")
    migrate(ckpt_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
