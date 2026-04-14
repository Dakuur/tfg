#!/usr/bin/env python3
"""
train.py — Training orchestrator for GAT-based N0/N1 metastasis classification.

Loads the configuration from a YAML file (default: configs/default.yaml).
Any parameter that is specified as a list triggers a grid search: one training
run is executed per combination of listed values.

A copy of the exact config used is saved alongside every checkpoint as
  outputs/checkpoints/{run_name}_best.yaml

Prerequisites:
    python scripts/build_dataset.py    # generate outputs/graphs/{train,val}/*.pt
    wandb login                        # authenticate once

Usage:
    # Single run / grid search (lists in YAML → all combinations)
    python train.py
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --run_name exp_01

    # W&B Sweep — Bayesian hyperparameter search
    python train.py --sweep                              # create new sweep + run 1 agent
    python train.py --sweep --sweep_count 5             # create sweep + run 5 trials
    python train.py --sweep_id <id>                     # join existing sweep (1 trial)
    python train.py --sweep_id <id> --sweep_count 10   # join existing sweep (10 trials)
"""

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

import torch
import wandb
import yaml

# ── project modules ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.data     import (  # noqa: E402
    load_graphs,
    compute_class_weights, make_loaders,
    compute_patient_class_weights, make_patient_loaders,
)
from scripts.model    import GATClassifier, PatientAggregator                    # noqa: E402
from scripts.training import (  # noqa: E402
    fix_seeds, EarlyStopping,
    train_epoch,         val_epoch,
    train_epoch_patient, val_epoch_patient,
    save_checkpoint,
)

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_SWEEP  = ROOT / "configs" / "sweep.yaml"


# ── config helpers ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def expand_grid(cfg: dict) -> list[tuple[dict, dict]]:
    """Return one (cfg_instance, varied_params) tuple per grid-search point.

    Leaf values that are Python lists are treated as the axis of a grid dimension;
    scalars are fixed across all runs.
    """
    leaves: dict[str, list] = {}  # "section.key" → [values]
    for section, vals in cfg.items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            leaves[f"{section}.{k}"] = v if isinstance(v, list) else [v]

    keys   = list(leaves.keys())
    combos = list(itertools.product(*[leaves[k] for k in keys]))

    varied_keys = {k for k in keys if len(leaves[k]) > 1}

    result = []
    for combo_vals in combos:
        flat     = dict(zip(keys, combo_vals))
        cfg_inst = copy.deepcopy(cfg)
        for dotpath, val in flat.items():
            section, key = dotpath.split(".", 1)
            cfg_inst[section][key] = val
        varied = {k: flat[k] for k in varied_keys}
        result.append((cfg_inst, varied))

    return result


def resolve_run_name(base: str | None, varied: dict, idx: int, n_combos: int) -> str | None:
    """Build a run name that encodes the varied hyper-params (grid search only)."""
    if n_combos == 1:
        return base  # None → W&B auto-generates
    suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(varied.items()))
    return f"{base}_{suffix}" if base else f"gs_{idx:02d}_{suffix}"


# ── sweep helpers ──────────────────────────────────────────────────────────────

# Which sweep parameter names belong to which config section
_MODEL_PARAMS = {
    "hidden", "heads", "dropout", "pooling", "diff_clusters",
}
_TRAINING_PARAMS = {
    "lr", "weight_decay", "aggregation", "batch_size",
    "scheduler_factor", "scheduler_patience", "scheduler_min_lr",
    "warm_up", "patience",
}


def _merge_sweep_params(base_cfg: dict, sweep_cfg) -> dict:
    """Overlay wandb sweep parameters onto a deep copy of the base config."""
    cfg = copy.deepcopy(base_cfg)
    for key, val in dict(sweep_cfg).items():
        if key in _MODEL_PARAMS:
            cfg["model"][key] = val
        elif key in _TRAINING_PARAMS:
            cfg["training"][key] = val
    return cfg


def _make_sweep_fn(base_cfg: dict):
    """Return the agent function passed to wandb.agent().

    Each call is one sweep trial: wandb.init() is called here so the sweep
    controller can populate wandb.config with the trial's hyperparameters.
    """
    def sweep_fn():
        run = wandb.init(project=base_cfg["wandb"]["project"])
        cfg = _merge_sweep_params(base_cfg, wandb.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fix_seeds(cfg["training"]["seed"])
        print(f"[INFO] wandb sweep run : {run.name}  ({run.url})")
        _train_body(cfg, run, device)

    return sweep_fn


# ── training body ──────────────────────────────────────────────────────────────

def _train_body(cfg: dict, run, device: torch.device) -> None:
    """Full training loop — assumes wandb.init() has already been called."""
    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]

    patient_level = bool(t.get("patient_level", False))
    aggregation   = t.get("aggregation", "noisy_or")

    # ── Data ──────────────────────────────────────────────────────────────────
    graphs_dir   = Path(d["graphs_dir"])
    train_graphs = load_graphs(graphs_dir / "train")
    val_graphs   = load_graphs(graphs_dir / "val")

    in_ch = train_graphs[0].x.shape[1]
    print(f"[INFO] Train: {len(train_graphs)}  Val: {len(val_graphs)}  Features: {in_ch}")
    print(f"[INFO] Mode: {'patient-level MIL' if patient_level else 'slide-level'}  "
          f"aggregation={aggregation if patient_level else 'n/a'}")

    if patient_level:
        class_weights = compute_patient_class_weights(train_graphs).to(device)
        train_loader, val_loader = make_patient_loaders(
            train_graphs, val_graphs, t["batch_size"]
        )
    else:
        class_weights = compute_class_weights(train_graphs).to(device)
        train_loader, val_loader = make_loaders(
            train_graphs, val_graphs, t["batch_size"]
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GATClassifier(
        in_channels   = in_ch,
        hidden        = m["hidden"],
        heads         = m["heads"],
        dropout       = m["dropout"],
        pooling       = m["pooling"],
        diff_clusters = m.get("diff_clusters", 10),
    ).to(device)

    # PatientAggregator only for attention MIL
    patient_aggregator = None
    if patient_level and aggregation == "attention":
        pool_out = m["hidden"] * 2 if m["pooling"] in ("mean_max", "diff") else m["hidden"]
        patient_aggregator = PatientAggregator(embed_dim=pool_out).to(device)
        print(f"[INFO] PatientAggregator: embed_dim={pool_out}")

    n_params = sum(p.numel() for p in model.parameters())
    if patient_aggregator is not None:
        n_params += sum(p.numel() for p in patient_aggregator.parameters())
    print(f"[INFO] Total params: {n_params:,}  |  pooling={m['pooling']}")
    wandb.config.update({"n_params": n_params, "in_channels": in_ch}, allow_val_change=True)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    all_params = list(model.parameters())
    if patient_aggregator is not None:
        all_params += list(patient_aggregator.parameters())

    optimizer = torch.optim.Adam(
        all_params,
        lr=t["lr"],
        weight_decay=t.get("weight_decay", 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=t.get("scheduler_factor", 0.5),
        patience=t.get("scheduler_patience", 5),
        min_lr=t.get("scheduler_min_lr", 1e-6),
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights) if not patient_level else None
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── Training state ────────────────────────────────────────────────────────
    monitor        = t.get("monitor", "val_auc")
    early_stopping = EarlyStopping(warm_up=t["warm_up"], patience=t["patience"])
    best_score     = 0.0
    best_epoch     = 0
    ckpt_path      = Path(d["checkpoint_dir"]) / f"{run.name}_best.pt"
    cfg_copy_path  = ckpt_path.with_suffix(".yaml")

    print(f"\n[INFO] Device : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[INFO] Monitoring : {monitor}")
    print(f"\nTraining for up to {t['epochs']} epochs …\n")

    t0 = time.time()

    for epoch in range(1, t["epochs"] + 1):
        if patient_level:
            tr = train_epoch_patient(
                model, train_loader, optimizer, class_weights, scaler, device,
                aggregation=aggregation, patient_aggregator=patient_aggregator,
            )
            va = val_epoch_patient(
                model, val_loader, class_weights, device,
                aggregation=aggregation, patient_aggregator=patient_aggregator,
            )
        else:
            tr = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
            va = val_epoch(model, val_loader, criterion, device)

        scheduler.step(va["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch":            epoch,
            "lr":               lr_now,
            "train/loss":       tr["loss"],
            "train/acc":        tr["acc"],
            "val/loss":         va["loss"],
            "val/acc":          va["acc"],
            "val/auc":          va["auc"],
            "val/f1_macro":     va["f1_macro"],
            "val/recall_N0":    va["recall_N0"],
            "val/recall_N1":    va["recall_N1"],
            "val/precision_N0": va["precision_N0"],
            "val/precision_N1": va["precision_N1"],
            "val/f1_N0":        va["f1_N0"],
            "val/f1_N1":        va["f1_N1"],
        })

        print(
            f"Epoch {epoch:3d}/{t['epochs']}  lr={lr_now:.2e}  "
            f"train_loss={tr['loss']:.4f}  train_acc={tr['acc']:.3f}  "
            f"val_loss={va['loss']:.4f}  val_acc={va['acc']:.3f}  "
            f"val_auc={va['auc']:.4f}  val_f1m={va['f1_macro']:.4f}  "
            f"f1_N1={va['f1_N1']:.4f}"
        )

        # ── Checkpoint by monitored metric ────────────────────────────────────
        score = va.get(monitor.replace("val_", ""), 0.0)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, va, ckpt_path,
                patient_aggregator=patient_aggregator,
            )
            # Save a config copy alongside the weights
            run_cfg = copy.deepcopy(cfg)
            run_cfg["_run"] = {"name": run.name, "url": run.url}
            save_config(run_cfg, cfg_copy_path)
            wandb.summary[f"best_{monitor}"] = best_score
            wandb.summary["best_epoch"]      = best_epoch
            print(f"         ↳ new best {monitor} {best_score:.4f} — checkpoint saved")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stopping(epoch, va["loss"], model):
            print(f"\n[INFO] Early stopping at epoch {epoch} (patience={t['patience']})")
            break

    elapsed = time.time() - t0
    h, mn, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\n── Training finished ───────────────────────────────────────────────")
    print(f"  Time          : {h}h {mn}m {s}s")
    print(f"  Best {monitor:15s}: {best_score:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint    : {ckpt_path.resolve()}")
    print(f"  Config copy   : {cfg_copy_path.resolve()}")

    wandb.finish()


# ── single training run (grid search / direct) ─────────────────────────────────

def train_one(cfg: dict, run_name: str | None) -> None:
    """Execute one full training run from the given config dict."""
    m   = cfg["model"]
    t   = cfg["training"]
    d   = cfg["data"]
    w   = cfg["wandb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seeds(t["seed"])

    run = wandb.init(
        project=w["project"],
        name=run_name,
        config={
            **m, **t,
            "graphs_dir": d["graphs_dir"],
            "device": str(device),
        },
    )
    print(f"[INFO] wandb run : {run.name}  ({run.url})")
    _train_body(cfg, run, device)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HistoGAT — reads hyperparams from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Override wandb run name (also used as checkpoint base name)",
    )

    # ── Sweep args ─────────────────────────────────────────────────────────────
    sweep_grp = p.add_argument_group("W&B Sweep")
    sweep_grp.add_argument(
        "--sweep",
        action="store_true",
        help="Create a new W&B sweep (from --sweep_config) and run one agent",
    )
    sweep_grp.add_argument(
        "--sweep_id",
        default=None,
        metavar="ID",
        help="Join an existing sweep by its ID (skips sweep creation)",
    )
    sweep_grp.add_argument(
        "--sweep_count",
        type=int,
        default=None,
        metavar="N",
        help="Number of trials for this agent (default: run until sweep is done)",
    )
    sweep_grp.add_argument(
        "--sweep_config",
        default=str(DEFAULT_SWEEP),
        metavar="PATH",
        help="Path to the W&B sweep definition YAML",
    )

    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"[ERROR] Config file not found: {config_path}")

    cfg = load_config(config_path)

    # CLI --run_name overrides config
    if args.run_name:
        cfg["wandb"]["run_name"] = args.run_name

    # ── W&B Sweep mode ────────────────────────────────────────────────────────
    if args.sweep or args.sweep_id:
        project = cfg["wandb"]["project"]

        if args.sweep_id:
            sweep_id = args.sweep_id
            print(f"[INFO] Joining sweep  : {sweep_id}")
        else:
            sweep_cfg_path = Path(args.sweep_config)
            if not sweep_cfg_path.exists():
                sys.exit(f"[ERROR] Sweep config not found: {sweep_cfg_path}")
            sweep_definition = load_config(sweep_cfg_path)
            sweep_id = wandb.sweep(sweep_definition, project=project)
            print(f"[INFO] Created sweep  : {sweep_id}")
            print(f"[INFO] Project        : {project}")
            print(f"[INFO] To add agents  : python train.py --sweep_id {sweep_id}")

        count_str = str(args.sweep_count) if args.sweep_count else "∞"
        print(f"[INFO] Agent trials   : {count_str}\n")

        wandb.agent(
            sweep_id,
            function=_make_sweep_fn(cfg),
            count=args.sweep_count,
            project=project,
        )
        return

    # ── Grid search / single run ──────────────────────────────────────────────
    combos = expand_grid(cfg)
    n      = len(combos)

    if n > 1:
        print(f"[INFO] Grid search: {n} combinations")
    else:
        print("[INFO] Single training run")

    for i, (cfg_inst, varied) in enumerate(combos, start=1):
        if n > 1:
            print(f"\n{'='*60}")
            print(f"  Run {i}/{n}  —  {varied}")
            print(f"{'='*60}\n")

        base_name = cfg_inst["wandb"].get("run_name")
        run_name  = resolve_run_name(base_name, varied, i, n)
        train_one(cfg_inst, run_name)


if __name__ == "__main__":
    main()
