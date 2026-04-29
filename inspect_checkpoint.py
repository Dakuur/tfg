"""Inspecciona les claus dels checkpoints per diagnosticar problemes de càrrega."""
import torch
from pathlib import Path

CKPT_DIR = Path("outputs/checkpoints")

for pt in sorted(CKPT_DIR.glob("*.pt")):
    print(f"\n{'='*60}")
    print(f"Fitxer: {pt.name}")
    try:
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            print(f"  Claus top-level: {list(ckpt.keys())}")
            sd = ckpt.get("model", ckpt)
        else:
            print(f"  Tipus: {type(ckpt)}")
            sd = ckpt

        print(f"  epoch:   {ckpt.get('epoch', 'n/a') if isinstance(ckpt, dict) else 'n/a'}")
        print(f"  val_auc: {ckpt.get('val_auc', 'n/a') if isinstance(ckpt, dict) else 'n/a'}")

        print(f"\n  Claus que contenen 'conv1':")
        for k in sd:
            if "conv1" in k:
                print(f"    {k}: {tuple(sd[k].shape)}")

        print(f"\n  Claus de batch norm:")
        for k in sd:
            if k.startswith("bn"):
                print(f"    {k}: {tuple(sd[k].shape)}")

        print(f"\n  Totes les claus:")
        for k in sd:
            print(f"    {k}: {tuple(sd[k].shape)}")

    except Exception as e:
        print(f"  ERROR: {e}")
