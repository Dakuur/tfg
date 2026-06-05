"""Bateja-calcula el test AUC de tots els checkpoints a `final/` i ompli
el fitxer `test_auc_cache.json`.

Ús (des de l'arrel del repo):

    .venv/bin/python frontend/compute_test_aucs.py

Llegeix els grafs de test des de `outputs/graphs/per-section/test/` (o el
subdir mega segons el `graph_type` del YAML del checkpoint), carrega cada
model, fa forward pacient-a-pacient amb la seva agregació MIL i guarda
l'AUC a la cache. Es pot reexecutar — només calcula els que falten.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Afegeix `frontend/` al path per reaprofitar `_load_model` i utilitats
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from main import (
    CKPT_DIR, TEST_AUC_CACHE, GRAPHS_DIR,
    _load_model, _read_config_yaml,
)


def load_test_graphs(graph_type: str):
    sub = "per-pacient" if graph_type == "mega" else "per-section"
    test_dir = GRAPHS_DIR / sub / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"No existeix {test_dir}")
    files = sorted(test_dir.glob("*.pt"))
    graphs = []
    for f in files:
        g = torch.load(f, weights_only=False, map_location="cpu")
        graphs.append(g)
    return graphs


def aggregate_patient(probs: list[float], method: str) -> float:
    p = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        return float(p.mean())
    if method == "max":
        return float(p.max())
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - p))
    if method == "lse":
        return float(np.log(np.exp(p).sum()))
    return float(p.mean())


def compute_test_auc(ckpt_path: Path, device: torch.device,
                     graphs_cache: dict) -> float | None:
    cfg = _read_config_yaml(ckpt_path)
    graph_type = "per-section"
    if cfg and "training" in cfg:
        graph_type = cfg["training"].get("graph_type", "per-section")
    # També consultem el slug — si conté "mega", forcem mega
    if "mega" in ckpt_path.stem:
        graph_type = "mega"
    aggregation = "mean"
    if cfg and "training" in cfg:
        aggregation = cfg["training"].get("aggregation", "mean")

    if graph_type not in graphs_cache:
        graphs_cache[graph_type] = load_test_graphs(graph_type)
    graphs = graphs_cache[graph_type]

    try:
        model, _info = _load_model(ckpt_path)
    except Exception as e:
        print(f"  [skip load_model] {e}")
        return None
    model.eval().to(device)

    # Agrupem per pacient (a per-section múltiples grafs comparteixen patient_id)
    patient_probs: dict[str, list[float]] = {}
    patient_label: dict[str, int]         = {}
    with torch.no_grad():
        for g in graphs:
            pid = getattr(g, "patient_id", None)
            if pid is None:
                continue
            y = int(g.y.item()) if hasattr(g.y, "item") else int(g.y)
            patient_label[pid] = y
            x  = g.x.to(device)
            ei = g.edge_index.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            logits = model(x, ei, batch)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            patient_probs.setdefault(pid, []).append(prob)

    if len(patient_probs) < 2:
        return None
    pids = sorted(patient_probs)
    y_true  = np.array([patient_label[p] for p in pids])
    y_score = np.array([aggregate_patient(patient_probs[p], aggregation) for p in pids])
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoints dir: {CKPT_DIR}")
    print(f"Test AUC cache : {TEST_AUC_CACHE}")

    cache = json.loads(TEST_AUC_CACHE.read_text()) if TEST_AUC_CACHE.exists() else {}
    print(f"Cache existent : {len(cache)} entrades\n")

    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    todo = [p for p in ckpts if str(p.relative_to(CKPT_DIR).with_suffix("")) not in cache]
    print(f"Total checkpoints: {len(ckpts)}   Pendents: {len(todo)}\n")

    graphs_cache: dict[str, list] = {}
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        rel = str(p.relative_to(CKPT_DIR).with_suffix(""))
        try:
            auc = compute_test_auc(p, device, graphs_cache)
        except Exception as e:
            print(f"[{i:3}/{len(todo)}] {rel}  →  ERROR: {e}")
            continue
        if auc is None:
            print(f"[{i:3}/{len(todo)}] {rel}  →  None (sense classes)")
            continue
        cache[rel] = auc
        print(f"[{i:3}/{len(todo)}] {rel:90s}  AUC={auc:.4f}")
        # Anem desant cada 10 perquè un crash no perdi tot el progrés
        if i % 10 == 0:
            TEST_AUC_CACHE.write_text(json.dumps(cache, indent=2))

    TEST_AUC_CACHE.write_text(json.dumps(cache, indent=2))
    print(f"\nFet en {time.time()-t0:.0f}s.  Cache total: {len(cache)} entrades.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
