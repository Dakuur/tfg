#!/usr/bin/env python3
"""
GAT Histopathology Explorer — FastAPI backend.
Serves the interactive web frontend for exploring the GAT model.

Usage:
    cd /path/to/tfg
    uvicorn frontend.main:app --reload --port 8000

Predictions are aggregated at patient level:
    if ANY slide of a patient predicts N1 → patient is N1
    all slides must predict N0 → patient is N0
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data

# ── bootstrap ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pt1diagnosis" / "models"))
sys.path.insert(0, str(ROOT / "pt1diagnosis" / "scripts_david"))
from GATClassifier  import GATClassifier                           # noqa: E402
from training_utils import aggregate_patient_probs                 # noqa: E402
from wsi_io         import (find_patches_dir, find_rgb_images_dir,  # noqa: E402
                             load_slide_meta, assemble_bag_image,
                             CLS_DIR_SUBPATH)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GAT Histopathology Explorer", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── global state ───────────────────────────────────────────────────────────────
class _State:
    model:           Optional[GATClassifier] = None
    checkpoint_info: Optional[Dict]          = None
    graphs:          Dict[str, List[Dict]]   = {"train": [], "val": []}
    val_stats:       Optional[Dict]          = None
    device:          str                     = "cpu"
    aggregation:     str                     = "noisy_or"


STATE = _State()

CKPT_DIR   = Path.home() / "outputs" / "checkpoints"
GRAPHS_DIR = Path.home() / "outputs" / "graphs"

# Image serving — requires access to /mnt/iam (or IAM_PATH env var)
_IAM_PATH = Path(os.environ.get("IAM_PATH", "/mnt/iam"))
try:
    PATCHES_DIR: Optional[Path] = find_patches_dir(_IAM_PATH)
    log.info(f"Patches dir  : {PATCHES_DIR}")
except FileNotFoundError:
    PATCHES_DIR = None
    log.info("Patches directory not found — patch assembly disabled")

_RGB_DIR: Optional[Path] = find_rgb_images_dir(_IAM_PATH)
log.info(f"RGB images dir: {_RGB_DIR}")

# NPZ cache — loaded on demand, one entry per hospital
_npz_cache: Dict[str, object] = {}

def _get_npz(hospital: str):
    """Return the loaded NPZ for a hospital, caching after first load."""
    if hospital not in _npz_cache:
        npz_path = _IAM_PATH / CLS_DIR_SUBPATH / f"{hospital}_CLS_2048.npz"
        if not npz_path.exists():
            return None
        _npz_cache[hospital] = np.load(npz_path, allow_pickle=True)
        log.info(f"Loaded NPZ for '{hospital}'")
    return _npz_cache[hospital]


# ── model loading ──────────────────────────────────────────────────────────────

def _read_config_yaml(ckpt_path: Path) -> Optional[dict]:
    yaml_path = ckpt_path.with_suffix(".yaml")
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    return None


def _load_model(ckpt_path: Path) -> tuple[GATClassifier, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd   = ckpt
        ckpt = {}

    _in_ch_candidates = [
        ("conv1.lin_src.weight", 1),
        ("conv1.lin.weight",     1),
        ("conv1.weight",         1),
        ("conv1.lin_src.weight", 0),
    ]
    in_channels = None
    for key, dim in _in_ch_candidates:
        if key in sd:
            in_channels = sd[key].shape[dim]
            break
    if in_channels is None:
        conv1_keys = [k for k in sd if "conv1" in k and "weight" in k]
        raise KeyError(f"Cannot detect in_channels. conv1 keys: {conv1_keys}")

    hidden = sd["bn3.weight"].shape[0]
    heads  = sd["bn1.weight"].shape[0] // hidden

    cfg = _read_config_yaml(ckpt_path)
    if cfg and "model" in cfg:
        pooling       = cfg["model"].get("pooling",       "mean_max")
        diff_clusters = cfg["model"].get("diff_clusters", 10)
        dropout       = cfg["model"].get("dropout",       0.3)
    else:
        mlp_in        = sd["mlp.0.weight"].shape[1]
        pooling       = "mean_max" if mlp_in == hidden * 2 else "mean"
        diff_clusters = 10
        dropout       = 0.3

    aggregation = "noisy_or"
    if cfg and "training" in cfg:
        aggregation = cfg["training"].get("aggregation", "noisy_or")

    model = GATClassifier(
        in_channels=in_channels, hidden=hidden, heads=heads,
        dropout=dropout, pooling=pooling, diff_clusters=diff_clusters,
        patient_aggregation=aggregation,
    )
    model.load_state_dict(sd)
    model.eval()

    try:
        rel_name = str(ckpt_path.relative_to(CKPT_DIR).with_suffix(""))
    except ValueError:
        rel_name = ckpt_path.stem
    info = {
        "path":         str(ckpt_path),
        "name":         rel_name,
        "epoch":        ckpt.get("epoch"),
        "val_auc":      ckpt.get("val_auc"),
        "val_f1_macro": ckpt.get("val_f1_macro"),
        "in_channels":  in_channels,
        "hidden":       hidden,
        "heads":        heads,
        "dropout":      dropout,
        "pooling":      pooling,
        "aggregation":  aggregation,
        "num_params":   sum(p.numel() for p in model.parameters()),
        "has_config":   cfg is not None,
    }
    return model, info


# ── graph / patient helpers ────────────────────────────────────────────────────

def _load_pt(path: Path) -> Optional[Data]:
    try:
        raw = torch.load(path, weights_only=False)
        if isinstance(raw, Data):
            return raw
        if isinstance(raw, dict):
            return Data(**raw)
    except Exception as e:
        log.warning(f"Failed to load {path}: {e}")
    return None


def _scan_graphs() -> Dict[str, List[Dict]]:
    """Carrega únicament els grafs del test set per evitar data leakage.

    Prioritat:
    1. Directori físic per-slide/test/   (generat per build_dataset.py)
    2. Filtratge via split.json des de per-slide/  (make_split.py / llegat)
    """
    result: Dict[str, List[Dict]] = {"train": [], "val": []}

    def _add(pt_path: Path, g=None) -> None:
        if g is None:
            g = _load_pt(pt_path)
        if g is None:
            return
        num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
        result["val"].append({
            "id":               f"test/{pt_path.stem}",
            "path":             str(pt_path.resolve()),
            "split":            "test",
            "stem":             pt_path.stem,
            "num_nodes":        num_nodes,
            "num_edges":        int(g.edge_index.shape[1]) if g.edge_index is not None else 0,
            "label":            int(g.y.item()) if hasattr(g, "y") and g.y is not None else -1,
            "patient_id":       str(getattr(g, "patient_id",       pt_path.stem)),
            "hospital":         str(getattr(g, "hospital",         "Unknown")),
            "metastasis_score": str(getattr(g, "metastasis_score", "—")),
        })

    per_slide_dir = GRAPHS_DIR / "per-slide"
    if not per_slide_dir.exists():
        log.warning(f"Directori de grafs no trobat: {per_slide_dir}")
        return result

    # ── Opció 1: directori físic test/ ────────────────────────────────────────
    test_subdir = per_slide_dir / "test"
    if test_subdir.exists() and any(test_subdir.glob("*.pt")):
        log.info(f"Carregant grafs des de directori físic: {test_subdir}")
        for pt_path in sorted(test_subdir.glob("*.pt")):
            _add(pt_path)
        log.info(f"  {len(result['val'])} grafs de test carregats des de test/")
        return result

    # ── Opció 2: split.json (llegat / make_split.py) ──────────────────────────
    split_json = per_slide_dir / "split.json"
    if not split_json.exists():
        log.error(
            f"Cap directori test/ ni split.json a {per_slide_dir}. "
            "Executa build_dataset.py (genera split físic) o "
            "scripts_david/make_split.py (genera split.json)."
        )
        return result

    with open(split_json) as f:
        test_pids = set(json.load(f).get("test", []))
    log.info(f"split.json: {len(test_pids)} pacients de test (filtratge runtime)")

    for pt_path in sorted(per_slide_dir.glob("*.pt")):
        g = _load_pt(pt_path)
        if g is None:
            continue
        pid = str(getattr(g, "patient_id", ""))
        if pid not in test_pids:
            continue
        _add(pt_path, g)

    return result


def _group_by_patient(graphs_dict: Dict[str, List[Dict]]) -> List[Dict]:
    """Group graph entries by patient_id, preserving splits and slide count."""
    patients: Dict[str, dict] = {}
    for split, graphs in graphs_dict.items():
        for g in graphs:
            pid = g["patient_id"]
            if pid not in patients:
                patients[pid] = {
                    "patient_id": pid,
                    "hospital":   g["hospital"],
                    "label":      g["label"],
                    "splits":     set(),
                    "graphs":     [],
                }
            patients[pid]["graphs"].append(g)
            patients[pid]["splits"].add(g["split"])

    result = []
    for data in patients.values():
        data["splits"]     = sorted(data["splits"])
        data["num_slides"] = len(data["graphs"])
        result.append(data)
    result.sort(key=lambda x: x["patient_id"])
    return result


def _list_checkpoints() -> List[Dict]:
    if not CKPT_DIR.exists():
        return []
    ckpts = sorted(CKPT_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in ckpts:
        cfg  = _read_config_yaml(p)
        rel  = str(p.relative_to(CKPT_DIR).with_suffix(""))
        entry: dict = {
            "name":       rel,
            "file":       str(p.relative_to(CKPT_DIR)),
            "active":     STATE.checkpoint_info is not None and STATE.checkpoint_info.get("name") == rel,
            "has_config": cfg is not None,
            "mtime":      int(p.stat().st_mtime),
        }
        if cfg and "model" in cfg:
            entry["pooling"] = cfg["model"].get("pooling")
            entry["hidden"]  = cfg["model"].get("hidden")
            entry["heads"]   = cfg["model"].get("heads")
        try:
            ckpt = torch.load(p, weights_only=True, map_location="cpu")
            entry["epoch"]        = ckpt.get("epoch")
            entry["val_auc"]      = ckpt.get("val_auc")
            entry["val_f1_macro"] = ckpt.get("val_f1_macro")
        except Exception:
            pass
        result.append(entry)
    return result


def _find_latest_checkpoint() -> Optional[Path]:
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(CKPT_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


# ── inference helpers ──────────────────────────────────────────────────────────

def _infer_slide_quick(
    entry: dict,
    model: GATClassifier,
    device: str,
) -> Optional[dict]:
    """Quick per-slide inference: returns probabilities only (no attention/PCA)."""
    g = _load_pt(Path(entry["path"]))
    if g is None:
        return None
    num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
    g_dev = g.to(device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        probs = F.softmax(model(g_dev.x, g_dev.edge_index, batch), dim=1)
    pred    = int(probs.argmax(dim=1).item())
    prob_n0 = float(probs[0, 0].item())
    prob_n1 = float(probs[0, 1].item())
    return {
        "graph_id":  entry["id"],
        "split":     entry["split"],
        "num_nodes": num_nodes,
        "num_edges": int(g.edge_index.shape[1]),
        "pred":      pred,
        "label_name": {0: "N0", 1: "N1"}.get(pred, "?"),
        "prob_n0":   prob_n0,
        "prob_n1":   prob_n1,
    }


def _infer_slide_full(
    entry:  dict,
    model:  GATClassifier,
    device: str,
    debug:  bool = False,
) -> dict:
    """Full per-slide inference with attention extraction and PCA."""
    debug_log: List[dict] = []

    def dlog(msg: str, level: str = "info"):
        if debug:
            debug_log.append({"t": round(time.time() * 1000), "level": level, "msg": msg})

    dlog(f"Carregant graf: {entry['id']}")
    g = _load_pt(Path(entry["path"]))
    if g is None:
        raise HTTPException(500, "Failed to load graph")

    num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
    num_edges = int(g.edge_index.shape[1])
    dlog(f"Graf carregat: {num_nodes} nodes, {num_edges} arestes")

    g_dev = g.to(device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    attention_layers: Dict[str, dict] = {}

    with torch.no_grad():
        x, ei = g_dev.x, g_dev.edge_index

        if model.pooling_type == "diff":
            # Hierarchical DiffPool changes node count and topology between layers;
            # skip per-layer attention breakdown and run the full forward pass.
            dlog("Pooling jeràrquic (diff): forward pass directe, sense desglossar atenció per capa")
            logits  = model(x, ei, batch)
            probs   = F.softmax(logits, dim=1)
        else:
            # GAT Capa 1
            dlog(f"GAT Capa 1 — heads={model.conv1.heads}")
            x1_raw, (ei1, a1) = model.conv1(x, ei, return_attention_weights=True)
            x1 = F.elu(model.bn1(x1_raw))
            x1 = F.dropout(x1, p=model.dropout, training=False)
            a1_mean = a1.mean(dim=1).cpu().float()

            # GAT Capa 2
            dlog(f"GAT Capa 2 — heads={model.conv2.heads}")
            x2_raw, (ei2, a2) = model.conv2(x1, ei, return_attention_weights=True)
            x2 = F.elu(model.bn2(x2_raw))
            x2 = F.dropout(x2, p=model.dropout, training=False)
            a2_mean = a2.mean(dim=1).cpu().float()

            # GAT Capa 3
            dlog(f"GAT Capa 3 — heads={model.conv3.heads}")
            x3_raw, (ei3, a3) = model.conv3(x2, ei, return_attention_weights=True)
            x3 = F.elu(model.bn3(x3_raw))
            a3_mean = a3.mean(dim=1).cpu().float()

            # Aggregate per-node attention
            for name, ei_l, a_mean, a_full in [
                ("layer1", ei1, a1_mean, a1),
                ("layer2", ei2, a2_mean, a2),
                ("layer3", ei3, a3_mean, a3),
            ]:
                ei_cpu    = ei_l.cpu()
                node_attn = torch.zeros(num_nodes)
                counts    = torch.zeros(num_nodes)
                for k in range(ei_cpu.shape[1]):
                    dst = ei_cpu[1, k].item()
                    if dst < num_nodes:
                        node_attn[dst] += a_mean[k].item()
                        counts[dst]    += 1
                counts    = counts.clamp(min=1)
                attention_layers[name] = {
                    "edge_index":       ei_l.cpu().numpy().tolist(),
                    "weights_mean":     a_mean.numpy().tolist(),
                    "weights_per_head": a_full.cpu().float().numpy().tolist(),
                    "node_attention":   (node_attn / counts).numpy().tolist(),
                    "num_heads":        a_full.shape[1],
                }

            # Pooling + MLP
            dlog(f"Pooling ({model.pooling_type}) + MLP…")
            h      = model.pool_readout(x3, ei, batch)
            logits = model.mlp(h)
            probs  = F.softmax(logits, dim=1)

        pred    = int(probs.argmax(dim=1).item())
        prob_n0 = float(probs[0, 0].item())
        prob_n1 = float(probs[0, 1].item())

    true_label = entry.get("label", -1)
    correct    = (pred == true_label) if true_label >= 0 else None
    label_names = {0: "N0", 1: "N1"}

    if debug:
        dlog(f"Predicció: {label_names.get(pred, '?')} — P(N0)={prob_n0:.3f}, P(N1)={prob_n1:.3f}")
        if true_label >= 0:
            dlog(
                f"Etiqueta real: {label_names.get(true_label, '?')} "
                f"→ {'✓ CORRECTE' if correct else '✗ INCORRECTE'}",
                level="success" if correct else "error",
            )

    return {
        "prediction":      pred,
        "label":           label_names.get(pred, "?"),
        "confidence":      float(probs[0, pred].item()),
        "prob_n0":         prob_n0,
        "prob_n1":         prob_n1,
        "true_label":      true_label,
        "true_label_name": label_names.get(true_label, "Unknown"),
        "correct":         correct,
        "num_nodes":       num_nodes,
        "num_edges":       num_edges,
        "patient_id":      entry.get("patient_id", ""),
        "hospital":        entry.get("hospital", ""),
        "slide_id":        str(getattr(g, "slide_id", "")),
        "pooling_type":    model.pooling_type,
        "attention":       attention_layers,
        "node_positions":  g.pos.cpu().numpy().tolist() if (hasattr(g, "pos") and g.pos is not None) else None,
        "edge_index":      g.edge_index.cpu().numpy().tolist(),
        "feature_norms":   g.x.cpu().float().norm(dim=1).numpy().tolist(),
        "patch_j":         g.patch_j.cpu().numpy().tolist()   if hasattr(g, "patch_j")   and g.patch_j   is not None else None,
        "patch_i":         g.patch_i.cpu().numpy().tolist()   if hasattr(g, "patch_i")   and g.patch_i   is not None else None,
        "patch_idx":       g.patch_idx.cpu().numpy().tolist() if hasattr(g, "patch_idx") and g.patch_idx is not None else None,
        "section_id":      str(g.section_id) if hasattr(g, "section_id") else None,
        "debug_log":       debug_log,
    }


# ── patient-level stats ────────────────────────────────────────────────────────

def _compute_val_stats(
    model:       GATClassifier,
    val_entries: List[Dict],
    device:      str,
    aggregation: str = "noisy_or",
) -> Dict:
    """Compute validation metrics aggregated at patient level.

    Uses aggregate_patient_probs with the given aggregation method (same as training).
    Attention is not supported here (no PatientAggregator); falls back to noisy_or.
    """
    model.eval()

    # Group val entries by patient
    patients: Dict[str, dict] = {}
    for entry in val_entries:
        pid = entry["patient_id"]
        if pid not in patients:
            patients[pid] = {"entries": [], "true_label": entry["label"]}
        patients[pid]["entries"].append(entry)

    all_true, all_pred, all_scores = [], [], []

    for pid, pdata in patients.items():
        slide_probs_n1 = []
        for entry in pdata["entries"]:
            try:
                res = _infer_slide_quick(entry, model, device)
                if res is None:
                    continue
                slide_probs_n1.append(res["prob_n1"])
            except Exception as e:
                log.warning(f"Stats failed for {entry['id']}: {e}")

        if not slide_probs_n1:
            continue

        _eff_agg = aggregation if aggregation != "attention" else "noisy_or"
        probs_t = torch.tensor(slide_probs_n1)
        patient_score = float(aggregate_patient_probs(probs_t, method=_eff_agg).item())
        patient_pred  = 1 if patient_score > 0.5 else 0

        all_true.append(pdata["true_label"])
        all_pred.append(patient_pred)
        all_scores.append(patient_score)

    if not all_true:
        return {}

    precision, recall, _ = precision_recall_curve(all_true, all_scores)
    fpr, tpr, _          = roc_curve(all_true, all_scores)
    cm                   = confusion_matrix(all_true, all_pred, labels=[0, 1]).tolist()
    try:
        auc_val = float(roc_auc_score(all_true, all_scores))
    except Exception:
        auc_val = None

    sensitivity  = float(recall_score(all_true, all_pred, pos_label=1, zero_division=0))
    specificity  = float(recall_score(all_true, all_pred, pos_label=0, zero_division=0))
    ppv          = float(precision_score(all_true, all_pred, pos_label=1, zero_division=0))
    npv          = float(precision_score(all_true, all_pred, pos_label=0, zero_division=0))
    balanced_acc = (sensitivity + specificity) / 2

    return {
        "accuracy":         float(accuracy_score(all_true, all_pred)),
        "auc":              auc_val,
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "ppv":              ppv,
        "npv":              npv,
        "balanced_acc":     balanced_acc,
        "confusion_matrix": cm,
        "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist()},
        "roc":              {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "class_distribution": {
            "N0": sum(1 for t in all_true if t == 0),
            "N1": sum(1 for t in all_true if t == 1),
        },
        "total_samples":    len(all_true),
        "aggregation":      aggregation,
        "level":            "patient",
    }


# ── startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    STATE.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {STATE.device.upper()}")

    ckpt = _find_latest_checkpoint()
    if ckpt:
        try:
            model, info           = _load_model(ckpt)
            STATE.model           = model.to(STATE.device)
            STATE.checkpoint_info = info
            STATE.aggregation     = info.get("aggregation", "noisy_or")
            auc_str = f"{info['val_auc']:.4f}" if info.get("val_auc") is not None else "n/a"
            log.info(f"Model: {info['name']}  epoch={info['epoch']}  "
                     f"val_auc={auc_str}  pooling={info['pooling']}  "
                     f"aggregation={STATE.aggregation}")
        except Exception as e:
            log.warning(f"Model load failed: {e}")
    else:
        log.info("No checkpoint found — running without model")

    STATE.graphs = _scan_graphs()
    n_te = len(STATE.graphs["val"])
    log.info(f"Graphs: {n_te} test")

    if STATE.model and STATE.graphs["val"]:
        log.info("Pre-computing patient-level test statistics…")
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device, STATE.aggregation)
        if STATE.val_stats:
            log.info(f"  acc={STATE.val_stats['accuracy']:.3f}  auc={STATE.val_stats.get('auc', 'n/a')}"
                     f"  patients={STATE.val_stats['total_samples']}")


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    all_ckpts = sorted(CKPT_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if CKPT_DIR.exists() else []
    return {
        "model_loaded":    STATE.model is not None,
        "checkpoint":      STATE.checkpoint_info,
        "device":          STATE.device,
        "aggregation":     STATE.aggregation,
        "num_test_graphs": len(STATE.graphs["val"]),
        "val_stats_ready": STATE.val_stats is not None,
        "search_paths": {
            "checkpoints_dir":        str(CKPT_DIR),
            "graphs_dir":             str(GRAPHS_DIR),
            "checkpoints_dir_exists": CKPT_DIR.exists(),
            "graphs_dir_exists":      GRAPHS_DIR.exists(),
            "all_checkpoints":        [str(p.relative_to(CKPT_DIR)) for p in all_ckpts],
        },
    }


@app.get("/api/checkpoints")
async def list_checkpoints():
    return {"checkpoints": _list_checkpoints()}


class SelectModelRequest(BaseModel):
    name: str


@app.post("/api/select_model")
async def select_model(req: SelectModelRequest):
    if not CKPT_DIR.exists():
        raise HTTPException(404, "Checkpoint directory not found")
    ckpt_path = CKPT_DIR / f"{req.name}.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / req.name
    if not ckpt_path.exists():
        raise HTTPException(404, f"Checkpoint not found: {req.name}")
    try:
        model, info           = _load_model(ckpt_path)
        STATE.model           = model.to(STATE.device)
        STATE.checkpoint_info = info
        STATE.aggregation     = info.get("aggregation", "noisy_or")
        STATE.val_stats       = None
        log.info(f"Model switched to: {info['name']}  pooling={info['pooling']}  aggregation={STATE.aggregation}")
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")
    if STATE.graphs["val"]:
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device, STATE.aggregation)
    return {"success": True, "checkpoint": STATE.checkpoint_info, "val_stats": STATE.val_stats}


@app.get("/api/graphs")
async def list_graphs(split: Optional[str] = None):
    if split:
        return {"graphs": STATE.graphs.get(split, [])}
    return {
        "graphs": STATE.graphs["train"] + STATE.graphs["val"],
        "train":  STATE.graphs["train"],
        "val":    STATE.graphs["val"],
    }


@app.get("/api/patients")
async def list_patients():
    """Return all graphs grouped by patient_id."""
    return {"patients": _group_by_patient(STATE.graphs)}


@app.get("/api/graphs/{graph_id:path}")
async def graph_data(graph_id: str):
    all_g  = STATE.graphs["train"] + STATE.graphs["val"]
    entry  = next((g for g in all_g if g["id"] == graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {graph_id}")
    g = _load_pt(Path(entry["path"]))
    if g is None:
        raise HTTPException(500, "Failed to load graph file")
    return {
        **entry,
        "edge_index":    g.edge_index.numpy().tolist(),
        "pos":           g.pos.numpy().tolist() if (hasattr(g, "pos") and g.pos is not None) else None,
        "feature_norms": g.x.norm(dim=1).numpy().tolist() if g.x is not None else None,
    }


# ── inference endpoints ────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    graph_id: str
    debug:    bool = False


@app.post("/api/inference")
async def inference(req: InferenceRequest):
    """Single-slide inference (returns attention + PCA for visualization)."""
    if STATE.model is None:
        raise HTTPException(503, "No model checkpoint loaded")
    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == req.graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {req.graph_id}")
    return _infer_slide_full(entry, STATE.model, STATE.device, debug=req.debug)


class PatientInferenceRequest(BaseModel):
    patient_id: str
    debug:      bool = False


@app.post("/api/inference_patient")
async def inference_patient(req: PatientInferenceRequest):
    """Patient-level inference: aggregates all slides.

    Rule: if ANY slide predicts N1 → patient = N1.
    Returns per-slide breakdown + full visualization from the most informative slide.
    """
    if STATE.model is None:
        raise HTTPException(503, "No model checkpoint loaded")

    all_g          = STATE.graphs["train"] + STATE.graphs["val"]
    patient_graphs = [g for g in all_g if g["patient_id"] == req.patient_id]
    if not patient_graphs:
        raise HTTPException(404, f"Patient not found: {req.patient_id}")

    # Quick inference on every slide
    slide_results = []
    for entry in patient_graphs:
        res = _infer_slide_quick(entry, STATE.model, STATE.device)
        if res is None:
            continue
        slide_results.append(res)

    if not slide_results:
        raise HTTPException(500, "Failed to run inference on any slide")

    # Patient-level aggregation using the same method as training
    _eff_agg = STATE.aggregation if STATE.aggregation != "attention" else "noisy_or"
    probs_t = torch.tensor([s["prob_n1"] for s in slide_results])
    patient_score = float(aggregate_patient_probs(probs_t, method=_eff_agg).item())
    patient_pred  = 1 if patient_score > 0.5 else 0

    # Sort slides: N1 predictions first, then by P(N1) descending
    slide_results.sort(key=lambda s: (-s["pred"], -s["prob_n1"]))

    true_label  = patient_graphs[0]["label"]
    correct     = (patient_pred == true_label) if true_label >= 0 else None
    label_names = {0: "N0", 1: "N1"}

    # Pick best slide for visualization:
    # → highest P(N1) if any slide is N1; else the slide with most nodes
    if patient_pred == 1:
        best_id    = max(slide_results, key=lambda s: s["prob_n1"])["graph_id"]
    else:
        best_id    = max(patient_graphs, key=lambda e: e["num_nodes"])["id"]
    best_entry = next(g for g in patient_graphs if g["id"] == best_id)

    # Full inference on best slide for visualisation
    viz = _infer_slide_full(best_entry, STATE.model, STATE.device, debug=req.debug)

    return {
        # Patient-level result
        "patient_id":      req.patient_id,
        "hospital":        patient_graphs[0]["hospital"],
        "true_label":      true_label,
        "true_label_name": label_names.get(true_label, "Unknown"),
        "prediction":      patient_pred,
        "label":           label_names.get(patient_pred, "?"),
        "prob_n1":         patient_score,
        "correct":         correct,
        "num_slides":      len(slide_results),
        "slide_results":   slide_results,
        # Per-slide breakdown includes slide id + predictions
        "viz_graph_id":    best_id,
        # Visualization from the best slide
        "num_nodes":       viz["num_nodes"],
        "num_edges":       viz["num_edges"],
        "attention":       viz["attention"],
        "node_positions": viz["node_positions"],
        "edge_index":     viz["edge_index"],
        "feature_norms":  viz["feature_norms"],
        "pooling_type":   viz["pooling_type"],
        "patch_j":        viz.get("patch_j"),
        "patch_i":        viz.get("patch_i"),
        "patch_idx":      viz.get("patch_idx"),
        "section_id":     viz.get("section_id"),
        "hospital":       viz.get("hospital", patient_graphs[0]["hospital"]),
        "slide_id":       viz.get("slide_id", ""),
        "debug_log":      viz["debug_log"],
    }


@app.get("/api/stats")
async def stats():
    if STATE.val_stats is not None:
        return STATE.val_stats
    if STATE.model is None:
        return JSONResponse({"error": "No model loaded"}, status_code=503)
    if not STATE.graphs["val"]:
        return JSONResponse({"error": "No validation graphs found"}, status_code=404)
    STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device, STATE.aggregation)
    return STATE.val_stats or JSONResponse({"error": "Could not compute statistics"}, status_code=500)


def _slide_dir_patch_index(search_dir: Path, hospital: str, patient_id: str, slide_id: str
                           ) -> dict:
    """
    Scan search_dir for .jpg patch files and build a {(j,i): Path} index.

    Handles two filename formats:
      Old: {hospital}_{patient}_{slide}_{j}_{i}.jpg        (2 coord parts)
      New: {hospital}_{patient}_{slide}_{level}_{j}_{i}.jpg (3 coord parts)
    In both cases the last two underscore-separated tokens are j and i.
    """
    prefix = f"{hospital}_{patient_id}_{slide_id}_"
    index: dict = {}
    for p in search_dir.glob("*.jpg"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        coord_part = stem[len(prefix):]
        parts = coord_part.split("_")
        if len(parts) < 2:
            continue
        try:
            index[(int(parts[-2]), int(parts[-1]))] = p
        except ValueError:
            continue
    return index


def _nearest_patch(index: dict, j: float, i: float) -> Optional[Path]:
    """Return the path of the patch whose (j,i) is closest to the query point."""
    if not index:
        return None
    coords = np.array(list(index.keys()), dtype=np.float32)   # (M, 2)
    dists  = np.sum((coords - [j, i]) ** 2, axis=1)
    best   = int(np.argmin(dists))
    if dists[best] > (8192 ** 2):   # > 8192 px away → probably wrong slide
        return None
    return index[tuple(coords[best].astype(int))]


@app.get("/api/bag_image")
async def bag_image(
    graph_id: str = Query(...),
    node_idx: int = Query(...),
):
    """
    Assemble and return the full 4096×4096 bag image for a graph node.

    Loads all 256 PNG patches from the NPZ paths field, assembles them into
    a 16×16 grid with 2 px black borders between patches, and returns JPEG.
    """
    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {graph_id}")

    g = _load_pt(Path(entry["path"]))
    if g is None:
        raise HTTPException(500, "Could not load graph")
    if not (hasattr(g, "patch_idx") and g.patch_idx is not None):
        raise HTTPException(422, "Graph has no patch_idx — rebuild with build_dataset.py")

    hospital   = str(g.hospital)
    patient_id = str(g.patient_id)
    slide_id   = str(g.slide_id)
    section_id = str(g.section_id)
    target_idx = int(g.patch_idx[node_idx].item())

    npz = _get_npz(hospital)
    if npz is None:
        raise HTTPException(503, f"NPZ not found for hospital '{hospital}'")

    mask = (
        (npz["patient_list"].astype(str) == patient_id) &
        (npz["slides"].astype(str)       == slide_id)   &
        (npz["sections"].astype(str)     == section_id)
    )
    if not mask.any():
        raise HTTPException(404, "No bags found in NPZ for this section")

    paths_all  = npz["paths"][mask]   # (N_bags, 256)
    coords_all = npz["coords"][mask]  # (N_bags, 256, 2)

    # Find the bag whose central patch index matches target_idx
    found_paths = found_coords = None
    for bag_paths, bag_coords in zip(paths_all, coords_all):
        centroid    = bag_coords.mean(axis=0)
        dists       = np.linalg.norm(bag_coords - centroid, axis=1)
        central_i   = int(dists.argmin())
        basename    = str(bag_paths[central_i]).replace("\\", "/").split("/")[-1]
        stem        = basename.rsplit(".", 1)[0]
        if int(stem.split("_")[-1]) == target_idx:
            found_paths  = bag_paths
            found_coords = bag_coords
            break

    if found_paths is None:
        raise HTTPException(404, f"Bag with central patch_idx={target_idx} not found in NPZ")

    if PATCHES_DIR is None:
        raise HTTPException(503, "Patches directory not available on this server")

    slide_dir = PATCHES_DIR / hospital / patient_id / slide_id / "patches"
    canvas    = assemble_bag_image(slide_dir, found_paths, found_coords, border=2)

    # Downscale to 900 px max side for fast transfer
    img = PILImage.fromarray(canvas)
    if max(img.size) > 900:
        scale    = 900 / max(img.size)
        img      = img.resize((int(img.width * scale), int(img.height * scale)),
                               PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg")


@app.get("/api/patch_image")
async def patch_image(
    hospital:   str           = Query(...),
    patient_id: str           = Query(...),
    slide_id:   str           = Query(...),
    section_id: Optional[str] = Query(None),
    patch_idx:  Optional[int] = Query(None),
    j:          Optional[int] = Query(None),
    i:          Optional[int] = Query(None),
):
    """Serve a patch image for a graph node.

    Primary:  j + i → Patches2048/{h}/{p}/{s}/patches/{h}_{p}_{s}_{j}_{i}.jpg
    Fallback: section_id + patch_idx → legacy sequential PNG format
    """
    if PATCHES_DIR is None:
        raise HTTPException(503, "Patches directory not available on this server")

    # New format: patches are in {slide}/patches/ subdir
    slide_dir        = PATCHES_DIR / hospital / patient_id / slide_id
    slide_patches_dir = slide_dir / "patches"
    search_dir        = slide_patches_dir if slide_patches_dir.exists() else slide_dir

    if not slide_dir.exists():
        raise HTTPException(404, f"Slide directory not found: {slide_dir}")

    # ── JPG by (j, i) coordinates (primary for PEARSON2 format) ───────────────
    if j is not None and i is not None:
        # Exact match (old format: no intermediate fields)
        fname    = f"{hospital}_{patient_id}_{slide_id}_{j}_{i}.jpg"
        img_path = search_dir / fname
        if img_path.exists():
            return FileResponse(str(img_path), media_type="image/jpeg")

        # Glob match: handles new format with extra field, e.g. *_{level}_{j}_{i}.jpg
        prefix  = f"{hospital}_{patient_id}_{slide_id}_"
        matches = list(search_dir.glob(f"{prefix}*_{j}_{i}.jpg"))
        if matches:
            return FileResponse(str(matches[0]), media_type="image/jpeg")

        # Nearest-patch fallback (builds full index; tolerates coord offsets ≤8192 px)
        index = _slide_dir_patch_index(search_dir, hospital, patient_id, slide_id)
        best  = _nearest_patch(index, j, i)
        if best:
            log.info(f"patch_image nearest match: {best.name}  (asked j={j} i={i})")
            return FileResponse(str(best), media_type="image/jpeg")

    # ── PNG fallback (legacy sequential-index format) ──────────────────────────
    if section_id is not None and patch_idx is not None:
        fname    = f"{hospital}_{patient_id}_{slide_id}_{section_id}_{patch_idx}.png"
        img_path = slide_dir / fname
        if img_path.exists():
            return FileResponse(str(img_path), media_type="image/png")

    if j is None or i is None:
        raise HTTPException(422, "Provide either (j + i) or (section_id + patch_idx)")

    raise HTTPException(404, f"No patch found near ({j},{i}) in {search_dir}")


@app.get("/api/slide_meta/{graph_id:path}")
async def slide_meta(graph_id: str):
    """
    Return WSI extent info so the frontend can align the background image
    with the node positions.  j_base/i_base/w/h are in WSI level-0 pixels.
    """
    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {graph_id}")

    g          = _load_pt(Path(entry["path"]))
    hospital   = entry["hospital"]
    patient_id = str(getattr(g, "patient_id", "")) if g else ""
    slide_id   = str(getattr(g, "slide_id",   "")) if g else ""

    meta   = load_slide_meta(_IAM_PATH, hospital, patient_id, slide_id)
    has_bg = False

    if _RGB_DIR:
        low = _RGB_DIR / hospital / patient_id / f"{hospital}_{slide_id}_low.png"
        has_bg = low.exists()

    return {
        "has_bg":  has_bg,
        "j_base":  meta["j_base"] if meta else None,
        "i_base":  meta["i_base"] if meta else None,
        "w":       meta["w"]      if meta else None,
        "h":       meta["h"]      if meta else None,
    }


@app.get("/api/slide_bg/{graph_id:path}")
async def slide_background(graph_id: str):
    """
    Return a small JPEG overview of the slide for use as graph background.
    Priority: _low.jpg → full PNG (resized) → assembled from patches.
    """
    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {graph_id}")

    g          = _load_pt(Path(entry["path"]))
    hospital   = entry["hospital"]
    patient_id = str(getattr(g, "patient_id", "")) if g else ""
    slide_id   = str(getattr(g, "slide_id",   "")) if g else ""

    cache_hdr = {"Cache-Control": "public, max-age=3600"}

    if _RGB_DIR:
        low = _RGB_DIR / hospital / patient_id / f"{hospital}_{slide_id}_low.png"
        if low.exists():
            return FileResponse(str(low), media_type="image/png", headers=cache_hdr)

    expected = f"{hospital}_{slide_id}_low.png"
    raise HTTPException(
        404,
        f"Imatge de fons no disponible: no s'ha trobat '{expected}' "
        f"al directori RGB_Images/{hospital}/{patient_id}/",
    )


@app.post("/api/reload")
async def reload():
    reload_log: List[str] = []
    STATE.model = None
    STATE.checkpoint_info = None
    STATE.val_stats = None

    reload_log.append(f"📁 Directori checkpoints: {CKPT_DIR}")
    if CKPT_DIR.exists():
        all_ckpts = sorted(CKPT_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        reload_log.append(f"   Fitxers .pt trobats: {len(all_ckpts)}")
        for p in all_ckpts:
            reload_log.append(f"     • {p.relative_to(CKPT_DIR)}")
    else:
        reload_log.append("   ⚠️  El directori no existeix")

    ckpt = _find_latest_checkpoint()
    if ckpt:
        reload_log.append(f"✅ Carregant: {ckpt.name}")
        try:
            model, info           = _load_model(ckpt)
            STATE.model           = model.to(STATE.device)
            STATE.checkpoint_info = info
            STATE.aggregation     = info.get("aggregation", "noisy_or")
            reload_log.append(
                f"   epoch={info['epoch']}  val_auc={info.get('val_auc')}  "
                f"pooling={info['pooling']}  aggregation={STATE.aggregation}  params={info['num_params']:,}"
            )
        except Exception as e:
            reload_log.append(f"❌ Error carregant model: {e}")
            return {"success": False, "error": str(e), "log": reload_log}
    else:
        reload_log.append("❌ Cap checkpoint trobat")

    STATE.graphs = _scan_graphs()
    n_te = len(STATE.graphs["val"])
    reload_log.append(f"   Grafs test: {n_te}")

    if STATE.model and STATE.graphs["val"]:
        reload_log.append("📊 Calculant estadístiques del test set (per pacient)…")
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device, STATE.aggregation)
        if STATE.val_stats:
            acc, auc = STATE.val_stats.get("accuracy", 0), STATE.val_stats.get("auc")
            n_pat    = STATE.val_stats.get("total_samples", 0)
            reload_log.append(
                f"   {n_pat} pacients  acc={acc:.3f}  auc={auc:.4f}" if auc
                else f"   {n_pat} pacients  acc={acc:.3f}"
            )

    log.info(f"Reload: model={'OK' if STATE.model else 'NO'}  graphs={n_te}")
    return {
        "success":      True,
        "model_loaded": STATE.model is not None,
        "num_test":     n_te,
        "checkpoint":   STATE.checkpoint_info,
        "log":          reload_log,
    }
