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

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data

# ── bootstrap ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.model import GATClassifier  # noqa: E402

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


STATE = _State()

CKPT_DIR   = ROOT / "outputs" / "checkpoints"
GRAPHS_DIR = ROOT / "outputs" / "graphs"


# ── model loading ──────────────────────────────────────────────────────────────

def _read_config_yaml(ckpt_path: Path) -> Optional[dict]:
    yaml_path = ckpt_path.with_suffix(".yaml")
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    return None


def _load_model(ckpt_path: Path) -> tuple[GATClassifier, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd   = ckpt["model"]

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

    model = GATClassifier(
        in_channels=in_channels, hidden=hidden, heads=heads,
        dropout=dropout, pooling=pooling, diff_clusters=diff_clusters,
    )
    model.load_state_dict(sd)
    model.eval()

    info = {
        "path":         str(ckpt_path),
        "name":         ckpt_path.stem,
        "epoch":        ckpt.get("epoch"),
        "val_auc":      ckpt.get("val_auc"),
        "val_f1_macro": ckpt.get("val_f1_macro"),
        "in_channels":  in_channels,
        "hidden":       hidden,
        "heads":        heads,
        "dropout":      dropout,
        "pooling":      pooling,
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
    result: Dict[str, List[Dict]] = {"train": [], "val": []}
    for split in ("train", "val"):
        split_dir = GRAPHS_DIR / split
        if not split_dir.exists():
            continue
        for pt_path in sorted(split_dir.glob("*.pt")):
            g = _load_pt(pt_path)
            if g is None:
                continue
            num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
            result[split].append({
                "id":               f"{split}/{pt_path.stem}",
                "path":             str(pt_path.resolve()),
                "split":            split,
                "stem":             pt_path.stem,
                "num_nodes":        num_nodes,
                "num_edges":        int(g.edge_index.shape[1]) if g.edge_index is not None else 0,
                "label":            int(g.y.item()) if hasattr(g, "y") and g.y is not None else -1,
                "patient_id":       str(getattr(g, "patient_id",       pt_path.stem)),
                "hospital":         str(getattr(g, "hospital",         "Unknown")),
                "metastasis_score": str(getattr(g, "metastasis_score", "—")),
            })
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
    ckpts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in ckpts:
        cfg   = _read_config_yaml(p)
        entry: dict = {
            "name":       p.stem,
            "file":       p.name,
            "active":     STATE.checkpoint_info is not None and STATE.checkpoint_info.get("name") == p.stem,
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
    ckpts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
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

    attention_layers:    Dict[str, dict] = {}
    node_embeddings_pca: Dict[str, dict] = {}

    with torch.no_grad():
        x, ei = g_dev.x, g_dev.edge_index

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

        # PCA of node embeddings
        for name, emb in [("layer1", x1), ("layer2", x2), ("layer3", x3)]:
            arr   = emb.cpu().float().numpy()
            N, D  = arr.shape
            n_comp = min(2, N - 1, D)
            if N >= 2 and D >= 2 and n_comp == 2:
                pca       = PCA(n_components=2)
                coords_2d = pca.fit_transform(arr).tolist()
                var_exp   = pca.explained_variance_ratio_.tolist()
            elif N >= 2 and D >= 1:
                coords_2d = [[float(arr[i, 0]), 0.0] for i in range(N)]
                var_exp   = [1.0, 0.0]
            else:
                coords_2d = [[0.0, 0.0]] * N
                var_exp   = [0.0, 0.0]
            node_embeddings_pca[name] = {"coords": coords_2d, "variance_explained": var_exp}

        # Pooling + MLP
        dlog(f"Pooling ({model.pooling_type}) + MLP…")
        h      = model.pool_readout(x3, ei, batch)
        logits = model.mlp(h)
        probs  = F.softmax(logits, dim=1)
        pred   = int(probs.argmax(dim=1).item())
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
        "pooling_type":    model.pooling_type,
        "attention":       attention_layers,
        "node_embeddings": node_embeddings_pca,
        "node_positions":  g.pos.cpu().numpy().tolist() if (hasattr(g, "pos") and g.pos is not None) else None,
        "edge_index":      g.edge_index.cpu().numpy().tolist(),
        "feature_norms":   g.x.cpu().float().norm(dim=1).numpy().tolist(),
        "debug_log":       debug_log,
    }


# ── patient-level stats ────────────────────────────────────────────────────────

def _compute_val_stats(
    model:       GATClassifier,
    val_entries: List[Dict],
    device:      str,
) -> Dict:
    """Compute validation metrics aggregated at patient level.

    Aggregation rule: if ANY slide of a patient is predicted N1 → patient = N1.
    Score used for AUC: max P(N1) across all slides of the patient.
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

        patient_score = max(slide_probs_n1)                   # soft score for AUC
        patient_pred  = 1 if any(p > 0.5 for p in slide_probs_n1) else 0

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

    return {
        "accuracy":         float(accuracy_score(all_true, all_pred)),
        "auc":              auc_val,
        "confusion_matrix": cm,
        "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist()},
        "roc":              {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "class_distribution": {
            "N0": sum(1 for t in all_true if t == 0),
            "N1": sum(1 for t in all_true if t == 1),
        },
        "total_samples":    len(all_true),
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
            log.info(f"Model: {info['name']}  epoch={info['epoch']}  "
                     f"val_auc={info.get('val_auc'):.4f}  pooling={info['pooling']}")
        except Exception as e:
            log.warning(f"Model load failed: {e}")
    else:
        log.info("No checkpoint found — running without model")

    STATE.graphs = _scan_graphs()
    n_tr, n_va   = len(STATE.graphs["train"]), len(STATE.graphs["val"])
    log.info(f"Graphs: {n_tr} train, {n_va} val")

    if STATE.model and STATE.graphs["val"]:
        log.info("Pre-computing patient-level validation statistics…")
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)
        if STATE.val_stats:
            log.info(f"  acc={STATE.val_stats['accuracy']:.3f}  auc={STATE.val_stats.get('auc', 'n/a')}"
                     f"  patients={STATE.val_stats['total_samples']}")


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    all_ckpts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if CKPT_DIR.exists() else []
    return {
        "model_loaded":     STATE.model is not None,
        "checkpoint":       STATE.checkpoint_info,
        "device":           STATE.device,
        "num_train_graphs": len(STATE.graphs["train"]),
        "num_val_graphs":   len(STATE.graphs["val"]),
        "val_stats_ready":  STATE.val_stats is not None,
        "search_paths": {
            "checkpoints_dir":        str(CKPT_DIR),
            "graphs_dir":             str(GRAPHS_DIR),
            "checkpoints_dir_exists": CKPT_DIR.exists(),
            "graphs_dir_exists":      GRAPHS_DIR.exists(),
            "all_checkpoints":        [p.name for p in all_ckpts],
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
        STATE.val_stats       = None
        log.info(f"Model switched to: {info['name']}  pooling={info['pooling']}")
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")
    if STATE.graphs["val"]:
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)
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

    # Patient-level aggregation: any N1 → N1
    patient_pred  = 1 if any(s["pred"] == 1 for s in slide_results) else 0
    patient_score = max(s["prob_n1"] for s in slide_results)

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
        "node_embeddings": viz["node_embeddings"],
        "node_positions":  viz["node_positions"],
        "edge_index":      viz["edge_index"],
        "feature_norms":   viz["feature_norms"],
        "pooling_type":    viz["pooling_type"],
        "debug_log":       viz["debug_log"],
    }


@app.get("/api/stats")
async def stats():
    if STATE.val_stats is not None:
        return STATE.val_stats
    if STATE.model is None:
        return JSONResponse({"error": "No model loaded"}, status_code=503)
    if not STATE.graphs["val"]:
        return JSONResponse({"error": "No validation graphs found"}, status_code=404)
    STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)
    return STATE.val_stats or JSONResponse({"error": "Could not compute statistics"}, status_code=500)


@app.post("/api/reload")
async def reload():
    reload_log: List[str] = []
    STATE.model = None
    STATE.checkpoint_info = None
    STATE.val_stats = None

    reload_log.append(f"📁 Directori checkpoints: {CKPT_DIR}")
    if CKPT_DIR.exists():
        all_ckpts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        reload_log.append(f"   Fitxers .pt trobats: {len(all_ckpts)}")
        for p in all_ckpts:
            reload_log.append(f"     • {p.name}")
    else:
        reload_log.append("   ⚠️  El directori no existeix")

    ckpt = _find_latest_checkpoint()
    if ckpt:
        reload_log.append(f"✅ Carregant: {ckpt.name}")
        try:
            model, info           = _load_model(ckpt)
            STATE.model           = model.to(STATE.device)
            STATE.checkpoint_info = info
            reload_log.append(
                f"   epoch={info['epoch']}  val_auc={info.get('val_auc')}  "
                f"pooling={info['pooling']}  params={info['num_params']:,}"
            )
        except Exception as e:
            reload_log.append(f"❌ Error carregant model: {e}")
            return {"success": False, "error": str(e), "log": reload_log}
    else:
        reload_log.append("❌ Cap checkpoint trobat")

    STATE.graphs = _scan_graphs()
    n_tr, n_va   = len(STATE.graphs["train"]), len(STATE.graphs["val"])
    reload_log.append(f"   Grafs train: {n_tr}  |  val: {n_va}")

    if STATE.model and STATE.graphs["val"]:
        reload_log.append("📊 Calculant estadístiques de validació (per pacient)…")
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)
        if STATE.val_stats:
            acc, auc = STATE.val_stats.get("accuracy", 0), STATE.val_stats.get("auc")
            n_pat    = STATE.val_stats.get("total_samples", 0)
            reload_log.append(
                f"   {n_pat} pacients  acc={acc:.3f}  auc={auc:.4f}" if auc
                else f"   {n_pat} pacients  acc={acc:.3f}"
            )

    log.info(f"Reload: model={'OK' if STATE.model else 'NO'}  graphs={n_tr + n_va}")
    return {
        "success":      True,
        "model_loaded": STATE.model is not None,
        "num_train":    n_tr,
        "num_val":      n_va,
        "checkpoint":   STATE.checkpoint_info,
        "log":          reload_log,
    }
