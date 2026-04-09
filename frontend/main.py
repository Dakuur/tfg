#!/usr/bin/env python3
"""
GAT Histopathology Explorer — FastAPI backend.
Serves the interactive web frontend for exploring the GAT model.

Usage:
    cd /path/to/tfg
    uvicorn frontend.main:app --reload --port 8000
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
from torch_geometric.nn import global_max_pool, global_mean_pool

# ── bootstrap ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from train import GATClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GAT Histopathology Explorer", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── global mutable state ───────────────────────────────────────────────────────
class _State:
    model: Optional[GATClassifier] = None
    checkpoint_info: Optional[Dict] = None
    graphs: Dict[str, List[Dict]] = {"train": [], "val": []}
    val_stats: Optional[Dict] = None
    device: str = "cpu"


STATE = _State()


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint() -> Optional[Path]:
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def _load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model"]

    # Auto-detect architecture from state dict shapes
    in_channels = sd["conv1.lin_src.weight"].shape[1]
    # bn3 output = hidden (conv3 has heads=1, concat=False)
    hidden = sd["bn3.weight"].shape[0]
    # bn1 output = hidden * heads (conv1 has concat=True)
    heads = sd["bn1.weight"].shape[0] // hidden

    model = GATClassifier(in_channels=in_channels, hidden=hidden, heads=heads, dropout=0.3)
    model.load_state_dict(sd)
    model.eval()

    return model, {
        "path": str(ckpt_path),
        "name": ckpt_path.stem,
        "epoch": ckpt.get("epoch"),
        "val_auc": ckpt.get("val_auc"),
        "in_channels": in_channels,
        "hidden": hidden,
        "heads": heads,
        "dropout": 0.3,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


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
    graphs_dir = ROOT / "outputs" / "graphs"
    result: Dict[str, List[Dict]] = {"train": [], "val": []}
    for split in ("train", "val"):
        split_dir = graphs_dir / split
        if not split_dir.exists():
            continue
        for pt_path in sorted(split_dir.glob("*.pt")):
            g = _load_pt(pt_path)
            if g is None:
                continue
            num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
            result[split].append({
                "id": f"{split}/{pt_path.stem}",
                "path": str(pt_path.resolve()),
                "split": split,
                "stem": pt_path.stem,
                "num_nodes": num_nodes,
                "num_edges": int(g.edge_index.shape[1]) if g.edge_index is not None else 0,
                "label": int(g.y.item()) if hasattr(g, "y") and g.y is not None else -1,
                "patient_id": str(getattr(g, "patient_id", pt_path.stem)),
                "hospital": str(getattr(g, "hospital", "Unknown")),
                "metastasis_score": str(getattr(g, "metastasis_score", "—")),
            })
    return result


def _compute_val_stats(model: GATClassifier, val_entries: List[Dict], device: str) -> Dict:
    model.eval()
    all_true, all_pred, all_scores = [], [], []

    for entry in val_entries:
        try:
            g = _load_pt(Path(entry["path"]))
            if g is None or not hasattr(g, "y") or g.y is None:
                continue
            g = g.to(device)
            batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
            with torch.no_grad():
                probs = F.softmax(model(g.x, g.edge_index, batch), dim=1)
            all_true.append(int(g.y.item()))
            all_pred.append(int(probs.argmax(dim=1).item()))
            all_scores.append(float(probs[0, 1].item()))
        except Exception as e:
            log.warning(f"Stats failed for {entry['id']}: {e}")

    if not all_true:
        return {}

    precision, recall, _ = precision_recall_curve(all_true, all_scores)
    fpr, tpr, _ = roc_curve(all_true, all_scores)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1]).tolist()
    try:
        auc_val = float(roc_auc_score(all_true, all_scores))
    except Exception:
        auc_val = None

    return {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "auc": auc_val,
        "confusion_matrix": cm,
        "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "class_distribution": {
            "N0": sum(1 for t in all_true if t == 0),
            "N1": sum(1 for t in all_true if t == 1),
        },
        "total_samples": len(all_true),
    }


# ── startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    STATE.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {STATE.device.upper()}")

    ckpt = _find_latest_checkpoint()
    if ckpt:
        try:
            model, info = _load_model(ckpt)
            STATE.model = model.to(STATE.device)
            STATE.checkpoint_info = info
            log.info(f"Model: {info['name']}  epoch={info['epoch']}  val_auc={info['val_auc']:.4f}")
        except Exception as e:
            log.warning(f"Model load failed: {e}")
    else:
        log.info("No checkpoint found — running without model")

    STATE.graphs = _scan_graphs()
    n_tr, n_va = len(STATE.graphs["train"]), len(STATE.graphs["val"])
    log.info(f"Graphs: {n_tr} train, {n_va} val")

    if STATE.model and STATE.graphs["val"]:
        log.info("Pre-computing validation statistics…")
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)
        if STATE.val_stats:
            log.info(f"  acc={STATE.val_stats['accuracy']:.3f}  auc={STATE.val_stats.get('auc', 'n/a')}")


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "model_loaded": STATE.model is not None,
        "checkpoint": STATE.checkpoint_info,
        "device": STATE.device,
        "num_train_graphs": len(STATE.graphs["train"]),
        "num_val_graphs": len(STATE.graphs["val"]),
        "val_stats_ready": STATE.val_stats is not None,
    }


@app.get("/api/graphs")
async def list_graphs(split: Optional[str] = None):
    if split:
        return {"graphs": STATE.graphs.get(split, [])}
    return {
        "graphs": STATE.graphs["train"] + STATE.graphs["val"],
        "train": STATE.graphs["train"],
        "val": STATE.graphs["val"],
    }


@app.get("/api/graphs/{graph_id:path}")
async def graph_data(graph_id: str):
    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {graph_id}")

    g = _load_pt(Path(entry["path"]))
    if g is None:
        raise HTTPException(500, "Failed to load graph file")

    pos = g.pos.numpy().tolist() if (hasattr(g, "pos") and g.pos is not None) else None
    feat_norms = g.x.norm(dim=1).numpy().tolist() if g.x is not None else None

    return {
        **entry,
        "edge_index": g.edge_index.numpy().tolist(),
        "pos": pos,
        "feature_norms": feat_norms,
    }


class InferenceRequest(BaseModel):
    graph_id: str
    debug: bool = False


@app.post("/api/inference")
async def inference(req: InferenceRequest):
    if STATE.model is None:
        raise HTTPException(503, "No model checkpoint loaded")

    all_g = STATE.graphs["train"] + STATE.graphs["val"]
    entry = next((g for g in all_g if g["id"] == req.graph_id), None)
    if not entry:
        raise HTTPException(404, f"Graph not found: {req.graph_id}")

    debug_log: List[Dict] = []

    def dlog(msg: str, level: str = "info"):
        debug_log.append({"t": round(time.time() * 1000), "level": level, "msg": msg})

    dlog(f"Carregant graf: {entry['id']}")
    g = _load_pt(Path(entry["path"]))
    if g is None:
        raise HTTPException(500, "Failed to load graph")

    num_nodes = int(g.num_nodes) if g.num_nodes is not None else int(g.x.shape[0])
    num_edges = int(g.edge_index.shape[1])
    dlog(f"Graf carregat: {num_nodes} nodes, {num_edges} arestes dirigides")

    g = g.to(STATE.device)
    model = STATE.model
    batch = torch.zeros(num_nodes, dtype=torch.long, device=STATE.device)

    attention_layers: Dict[str, Dict] = {}
    node_embeddings_pca: Dict[str, Dict] = {}

    dlog("Iniciant forward pass amb extracció d'atenció…")

    with torch.no_grad():
        x, ei = g.x, g.edge_index

        # ── Layer 1 ────────────────────────────────────────────────────────────
        dlog(f"GAT Capa 1 — heads={model.conv1.heads}, concat=True")
        x1_raw, (ei1, a1) = model.conv1(x, ei, return_attention_weights=True)
        x1 = F.elu(model.bn1(x1_raw))
        x1 = F.dropout(x1, p=model.dropout, training=False)
        a1_mean = a1.mean(dim=1).cpu().float()
        dlog(f"  → {ei1.shape[1]} arestes (incl. self-loops), {a1.shape[1]} caps d'atenció")

        # ── Layer 2 ────────────────────────────────────────────────────────────
        dlog(f"GAT Capa 2 — heads={model.conv2.heads}, concat=True")
        x2_raw, (ei2, a2) = model.conv2(x1, ei, return_attention_weights=True)
        x2 = F.elu(model.bn2(x2_raw))
        x2 = F.dropout(x2, p=model.dropout, training=False)
        a2_mean = a2.mean(dim=1).cpu().float()
        dlog(f"  → {ei2.shape[1]} arestes, {a2.shape[1]} caps d'atenció")

        # ── Layer 3 ────────────────────────────────────────────────────────────
        dlog(f"GAT Capa 3 — heads={model.conv3.heads}, concat=False")
        x3_raw, (ei3, a3) = model.conv3(x2, ei, return_attention_weights=True)
        x3 = F.elu(model.bn3(x3_raw))
        a3_mean = a3.mean(dim=1).cpu().float()
        dlog(f"  → {ei3.shape[1]} arestes, {a3.shape[1]} caps d'atenció")

        # ── Aggregate per-node attention ───────────────────────────────────────
        for name, ei_l, a_mean in [
            ("layer1", ei1, a1_mean),
            ("layer2", ei2, a2_mean),
            ("layer3", ei3, a3_mean),
        ]:
            ei_cpu = ei_l.cpu()
            node_attn = torch.zeros(num_nodes)
            counts = torch.zeros(num_nodes)
            for k in range(ei_cpu.shape[1]):
                dst = ei_cpu[1, k].item()
                if dst < num_nodes:
                    node_attn[dst] += a_mean[k].item()
                    counts[dst] += 1
            counts = counts.clamp(min=1)
            node_attn = (node_attn / counts).numpy().tolist()

            heads_used = a1.shape[1] if name == "layer1" else (a2.shape[1] if name == "layer2" else a3.shape[1])
            a_full = a1 if name == "layer1" else (a2 if name == "layer2" else a3)
            attention_layers[name] = {
                "edge_index": ei_l.cpu().numpy().tolist(),
                "weights_mean": a_mean.numpy().tolist(),
                "weights_per_head": a_full.cpu().float().numpy().tolist(),
                "node_attention": node_attn,
                "num_heads": heads_used,
            }

        # ── PCA of node embeddings ─────────────────────────────────────────────
        dlog("Reducció PCA dels embeddings de nodes (→ 2D)…")
        for name, emb in [("layer1", x1), ("layer2", x2), ("layer3", x3)]:
            arr = emb.cpu().float().numpy()
            N, D = arr.shape
            if N >= 2 and D >= 2:
                n_comp = min(2, N - 1, D)
                if n_comp == 2:
                    pca = PCA(n_components=2)
                    coords_2d = pca.fit_transform(arr).tolist()
                    var_exp = pca.explained_variance_ratio_.tolist()
                else:
                    coords_2d = [[float(arr[i, 0]), 0.0] for i in range(N)]
                    var_exp = [1.0, 0.0]
            else:
                coords_2d = [[0.0, 0.0]] * N
                var_exp = [0.0, 0.0]
            node_embeddings_pca[name] = {"coords": coords_2d, "variance_explained": var_exp}

        # ── Global pooling + MLP ───────────────────────────────────────────────
        dlog("Global pooling (mean + max) → cap MLP…")
        h = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)
        logits = model.mlp(h)
        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        prob_n0 = float(probs[0, 0].item())
        prob_n1 = float(probs[0, 1].item())

    true_label = entry.get("label", -1)
    correct = (pred == true_label) if true_label >= 0 else None

    label_names = {0: "N0", 1: "N1"}
    dlog(f"Predicció: {label_names.get(pred, '?')} — P(N0)={prob_n0:.3f}, P(N1)={prob_n1:.3f}")
    if true_label >= 0:
        dlog(
            f"Etiqueta real: {label_names.get(true_label, '?')} → {'✓ CORRECTE' if correct else '✗ INCORRECTE'}",
            level="success" if correct else "error",
        )

    pos = g.pos.cpu().numpy().tolist() if (hasattr(g, "pos") and g.pos is not None) else None

    return {
        "prediction": pred,
        "label": label_names.get(pred, "?"),
        "confidence": float(probs[0, pred].item()),
        "prob_n0": prob_n0,
        "prob_n1": prob_n1,
        "true_label": true_label,
        "true_label_name": label_names.get(true_label, "Unknown"),
        "correct": correct,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "patient_id": entry.get("patient_id", ""),
        "hospital": entry.get("hospital", ""),
        "attention": attention_layers,
        "node_embeddings": node_embeddings_pca,
        "node_positions": pos,
        "edge_index": g.edge_index.cpu().numpy().tolist(),
        "feature_norms": g.x.cpu().float().norm(dim=1).numpy().tolist(),
        "debug_log": debug_log,
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
    ckpt = _find_latest_checkpoint()
    if ckpt:
        try:
            model, info = _load_model(ckpt)
            STATE.model = model.to(STATE.device)
            STATE.checkpoint_info = info
        except Exception as e:
            return {"success": False, "error": str(e)}

    STATE.graphs = _scan_graphs()
    STATE.val_stats = None

    if STATE.model and STATE.graphs["val"]:
        STATE.val_stats = _compute_val_stats(STATE.model, STATE.graphs["val"], STATE.device)

    return {
        "success": True,
        "model_loaded": STATE.model is not None,
        "num_train": len(STATE.graphs["train"]),
        "num_val": len(STATE.graphs["val"]),
    }
