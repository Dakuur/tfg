#!/usr/bin/env python3
"""
Graph construction utilities shared across scripts.

Functions
---------
build_delaunay_edges  – Delaunay triangulation with long-edge pruning
filter_edges_by_mask  – Remove edges that cross non-tissue mask regions
make_edge_index       – Build bidirectional torch edge_index from an edge array
export_graph          – Save a graph dict as a .pt file for torch_geometric

Constants
---------
DISTANCE_FACTOR       – Default long-edge pruning multiplier (2.0)
MIN_BAGS_PER_SECTION  – Minimum bags required per section for Delaunay (3)
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

# ── shared constants ───────────────────────────────────────────────────────────
DISTANCE_FACTOR      = 2.0  # prune edges longer than DISTANCE_FACTOR × mean_length
MIN_BAGS_PER_SECTION = 3    # Delaunay requires ≥ 3 non-collinear points


# ── core graph algorithms ──────────────────────────────────────────────────────

def build_delaunay_edges(
    coords: np.ndarray,
    distance_factor: float = DISTANCE_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Delaunay triangulation and prune long edges.

    Parameters
    ----------
    coords          : (N, 2) float array of node positions
    distance_factor : edges longer than distance_factor × mean_length are pruned

    Returns
    -------
    edges   : (M, 2) int64 array of undirected node-index pairs
    lengths : (M,)   float64 array of edge lengths (same order as edges)
    """
    tri = Delaunay(coords)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = simplex[a], simplex[b]
            edge_set.add((min(u, v), max(u, v)))
    edges   = np.array(list(edge_set), dtype=np.int64)
    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    threshold = distance_factor * lengths.mean()
    keep = lengths <= threshold
    return edges[keep], lengths[keep]


def filter_edges_by_mask(
    edges: np.ndarray,
    coords: np.ndarray,
    mask_img: np.ndarray,
    j_base: float,
    i_base: float,
    slide_w: float,
    slide_h: float,
    patch_size: int,
    n_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove edges whose midline passes through a black (background) region
    in the segmentation mask.

    The mask covers WSI region [j_base, j_base+slide_w] × [i_base, i_base+slide_h]
    (level-0 pixels), stored at annotation-thumbnail resolution (mask_h × mask_w).

    Parameters
    ----------
    edges      : (M, 2) int array from build_delaunay_edges
    coords     : (N, 2) node positions in WSI level-0 pixels (j, i)
    mask_img   : (mask_h, mask_w, 3) uint8 RGB segmentation mask
    j_base     : x origin of the slide crop in WSI level-0 pixels
    i_base     : y origin of the slide crop in WSI level-0 pixels
    slide_w    : width  of the slide crop in WSI level-0 pixels
    slide_h    : height of the slide crop in WSI level-0 pixels
    patch_size : patch side length in WSI level-0 pixels
    n_samples  : number of points sampled along each edge for the tissue check

    Returns
    -------
    kept_edges    : (K, 2) int64 array
    removed_edges : (R, 2) int64 array
    """
    mask_h, mask_w = mask_img.shape[:2]
    sx = mask_w / slide_w
    sy = mask_h / slide_h

    tissue = np.any(mask_img > 0, axis=-1)  # (mask_h, mask_w)

    # Patch centres mapped to mask pixel space
    cx_m = (coords[:, 0] + patch_size / 2 - j_base) * sx
    cy_m = (coords[:, 1] + patch_size / 2 - i_base) * sy

    kept, removed = [], []
    for u, v in edges:
        xs = np.linspace(cx_m[u], cx_m[v], n_samples)
        ys = np.linspace(cy_m[u], cy_m[v], n_samples)
        xi = np.clip(xs.astype(int), 0, mask_w - 1)
        yi = np.clip(ys.astype(int), 0, mask_h - 1)
        if np.all(tissue[yi, xi]):
            kept.append((u, v))
        else:
            removed.append((u, v))

    kept_arr    = np.array(kept,    dtype=np.int64) if kept    else np.empty((0, 2), dtype=np.int64)
    removed_arr = np.array(removed, dtype=np.int64) if removed else np.empty((0, 2), dtype=np.int64)
    return kept_arr, removed_arr


def make_edge_index(edges: np.ndarray) -> torch.Tensor:
    """
    Convert an undirected (M, 2) edge array to a bidirectional (2, 2M) edge_index
    tensor as expected by torch_geometric.

    Parameters
    ----------
    edges : (M, 2) int array of (src, dst) pairs

    Returns
    -------
    edge_index : (2, 2M) int64 tensor
    """
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, M)
    return torch.cat([ei, ei.flip(0)], dim=1)                    # (2, 2M)


def export_graph(
    coords: np.ndarray,
    edges: np.ndarray,
    out_path: Path,
    feature_dim: int = 1536,
) -> None:
    """
    Save a graph to a .pt file compatible with torch_geometric.Data.

    The saved dict contains:
      x          : (N, feature_dim) float32  — random placeholder features
      edge_index : (2, 2M) int64             — bidirectional edge list
      pos        : (N, 2)  float32           — (j, i) WSI level-0 coordinates
      num_nodes  : int

    Load example::

        data_dict = torch.load("graph.pt", weights_only=False)
        from torch_geometric.data import Data
        data = Data(**data_dict)

    Parameters
    ----------
    coords      : (N, 2) node positions in WSI level-0 pixels
    edges       : (M, 2) undirected edge array from build_delaunay_edges
    out_path    : target path; .pt extension is enforced
    feature_dim : dimensionality of the random placeholder feature vectors
    """
    n_nodes    = len(coords)
    x          = torch.randn(n_nodes, feature_dim)
    pos        = torch.tensor(coords, dtype=torch.float32)
    edge_index = make_edge_index(edges)

    data = {
        "x":          x,
        "edge_index": edge_index,
        "pos":        pos,
        "num_nodes":  n_nodes,
    }

    graph_path = Path(out_path).with_suffix(".pt")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, graph_path)
    print(
        f"[INFO] Graph saved : {graph_path.resolve()}"
        f"  ({n_nodes} nodes, {edge_index.shape[1]} directed edges, {feature_dim} features/node)"
    )
