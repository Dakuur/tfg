#!/usr/bin/env python3
"""
Delaunay graph visualization over a single WSI slide.

Creates a figure with two panels:
  Left:  patch collage with the full RGB slide at 0.5 opacity in the background.
  Right: patch collage with the segmentation mask (or full RGB — see OVERLAY_MODE)
         at 0.5 opacity on top, and the Delaunay graph superimposed.
         Edges that cross black (non-tissue) mask regions are removed.

─── single-line toggles ────────────────────────────────────────────────────────
  OVERLAY_MODE = "mask"  →  right panel shows the segmentation mask at 0.5
  OVERLAY_MODE = "rgb"   →  right panel shows the full RGB image at 0.5

Usage example (defaults work out of the box):
    python scripts/delaunay_vis.py

Custom example:
    python scripts/delaunay_vis.py \\
        --hospital "H. Bellvitge" \\
        --max_patches 300 \\
        --output outputs/bellvitge_delaunay.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import Delaunay

Image.MAX_IMAGE_PIXELS = 400_000_000


# ── dataset paths ──────────────────────────────────────────────────────────────
PATCHES_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches_2048"
)
RGB_IMAGES_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images"
)
MASKS_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks"
)
SLIDE_META_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patient_Images_metadata.csv"
)

# ── single-line toggle ─────────────────────────────────────────────────────────
OVERLAY_MODE = "mask"   # "mask" → segmentation mask | "rgb" → full RGB image


# ── dataset helpers ────────────────────────────────────────────────────────────

def find_patches_dir(iam_path: Path) -> Path:
    for candidate in [iam_path / "Patches", iam_path / PATCHES_SUBPATH]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find a 'Patches' directory under '{iam_path}'.\n"
        "Pass --iam_path pointing to the dataset root (/mnt/iam) or the Images/ folder."
    )


def find_rgb_images_dir(iam_path: Path) -> Path | None:
    for candidate in [iam_path / "RGB_Images", iam_path / RGB_IMAGES_SUBPATH]:
        if candidate.is_dir():
            return candidate
    return None


def find_masks_dir(iam_path: Path) -> Path | None:
    for candidate in [iam_path / "Segmentation_Masks", iam_path / MASKS_SUBPATH]:
        if candidate.is_dir():
            return candidate
    return None


def load_metadata(patches_dir: Path, hospital: str) -> pd.DataFrame:
    meta_path = patches_dir / hospital / f"metadata_{hospital}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    df = pd.read_csv(meta_path)
    for col in ["i", "j", "blurriness", "non_white_area", "affected_percentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_slide_meta(
    iam_path: Path, hospital: str, patient: str, slide: str
) -> dict | None:
    """
    Read per-slide WSI bounds from Patient_Images_metadata.csv.

    Returns dict with j_base, i_base, w, h (all in WSI level-0 pixels),
    or None if the file/row is unavailable.
    """
    meta_path = iam_path / SLIDE_META_SUBPATH
    if not meta_path.exists():
        print(f"[WARN] Slide metadata CSV not found: {meta_path}")
        print("       Background alignment will be approximate; edge mask-filter disabled.")
        return None
    df = pd.read_csv(meta_path, encoding="utf-8")
    row = df[
        (df["hospital"].astype(str) == hospital) &
        (df["patient_ID"].astype(str) == patient) &
        (df["slide_ID"].astype(str) == slide)
    ]
    if row.empty:
        print(f"[WARN] {hospital}/{patient}/{slide} not found in slide metadata CSV.")
        return None
    r = row.iloc[-1]
    return {
        "j_base": float(r["j"]),   # mrxs_cmin — WSI level-0 x origin of the slide crop
        "i_base": float(r["i"]),   # mrxs_rmin — WSI level-0 y origin of the slide crop
        "w":      float(r["w"]),   # width  of the slide crop in WSI level-0 pixels
        "h":      float(r["h"]),   # height of the slide crop in WSI level-0 pixels
    }


def filter_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Quality filter: discard blank and blurry patches."""
    mask = (df["non_white_area"] >= 0.3) & (df["blurriness"] <= 100)
    return df[mask].copy()


def select_best_slide(df: pd.DataFrame) -> tuple[str, str]:
    """Return (patient_id, slide_id) for the slide with the most patches."""
    counts = df.groupby(["patient_ID", "slide_ID"]).size()
    patient_id, slide_id = counts.idxmax()
    return str(patient_id), str(slide_id)


# ── image loading ──────────────────────────────────────────────────────────────

def load_mask_image(
    iam_path: Path, hospital: str, patient: str, slide: str
) -> np.ndarray | None:
    """Load the segmentation mask for the given slide."""
    masks_dir = find_masks_dir(iam_path)
    if masks_dir is None:
        print("[WARN] Segmentation_Masks directory not found.")
        return None
    path = masks_dir / hospital / patient / f"{hospital}_{slide}_mask.png"
    if not path.exists():
        print(f"[WARN] Mask not found: {path}")
        return None
    print(f"[INFO] Mask      : {path}")
    return np.array(Image.open(path).convert("RGB"))


def load_rgb_image(
    iam_path: Path, hospital: str, patient: str, slide: str
) -> np.ndarray | None:
    """Load the full RGB slide image."""
    rgb_dir = find_rgb_images_dir(iam_path)
    if rgb_dir is None:
        print("[WARN] RGB_Images directory not found.")
        return None
    path = rgb_dir / hospital / patient / f"{hospital}_{slide}.png"
    if not path.exists():
        print(f"[WARN] RGB image not found: {path}")
        return None
    print(f"[INFO] RGB image : {path}")
    return np.array(Image.open(path).convert("RGB"))


# ── patch loading ──────────────────────────────────────────────────────────────

def _patch_filename(hospital: str, patient: str, slide: str,
                    i: int, j: int) -> str:
    # Files on disk are named {j}_{i} (column-coord first, row-coord second)
    return f"{hospital}_{patient}_{slide}_{j}_{i}.jpg"


def load_patches(
    patches_dir: Path,
    df_slide: pd.DataFrame,
    hospital: str,
    patient: str,
    slide: str,
    max_patches: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load up to *max_patches* images for one slide.

    Returns
    -------
    images    : list of (2048, 2048, 3) uint8 arrays
    coords    : (N, 2) float array — columns are (j, i) = (x, y) in WSI level-0 pixels
    non_white : (N,) float array
    """
    if len(df_slide) > max_patches:
        df_slide = df_slide.sample(n=max_patches, random_state=42).reset_index(drop=True)

    slide_dir = patches_dir / hospital / patient / slide
    images: list[np.ndarray] = []
    coords_list: list[tuple[float, float]] = []
    non_white_list: list[float] = []

    _debug_printed = False
    for _, row in df_slide.iterrows():
        if not _debug_printed:
            _fname = _patch_filename(
                hospital, str(row["patient_ID"]), str(row["slide_ID"]),
                int(row["i"]), int(row["j"]),
            )
            print(f"[DEBUG] First path: {slide_dir / _fname}")
            print(f"[DEBUG] Exists    : {(slide_dir / _fname).exists()}")
            _debug_printed = True
        fname = _patch_filename(
            hospital, str(row["patient_ID"]), str(row["slide_ID"]),
            int(row["i"]), int(row["j"]),
        )
        img_path = slide_dir / fname
        if not img_path.exists():
            continue
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            continue
        images.append(img)
        coords_list.append((float(row["j"]), float(row["i"])))
        non_white_list.append(float(row["non_white_area"]))

    coords    = np.array(coords_list,   dtype=np.float64)
    non_white = np.array(non_white_list, dtype=np.float64)
    return images, coords, non_white


# ── graph construction ─────────────────────────────────────────────────────────

def build_delaunay_edges(
    coords: np.ndarray,
    distance_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delaunay triangulation + long-edge pruning.

    Returns
    -------
    edges   : (M, 2) int array of node-index pairs
    lengths : (M,)   float array of edge lengths in pixels
    """
    tri = Delaunay(coords)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = simplex[a], simplex[b]
            edge_set.add((min(u, v), max(u, v)))
    edges = np.array(list(edge_set), dtype=np.int64)
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
) -> np.ndarray:
    """
    Remove edges whose line between patch centres passes through a black
    (background, value == 0) region in the segmentation mask.

    The mask covers WSI region [j_base, j_base+slide_w] × [i_base, i_base+slide_h]
    in level-0 pixels, stored at annotation-thumbnail resolution (mask_h × mask_w).

    Parameters
    ----------
    n_samples : number of points sampled along each edge for the tissue check
    """
    mask_h, mask_w = mask_img.shape[:2]
    sx = mask_w / slide_w
    sy = mask_h / slide_h

    # Binary map: True = tissue present (any channel > 0)
    tissue = np.any(mask_img > 0, axis=-1)  # (mask_h, mask_w)

    # Patch centres in mask pixel coordinates
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


# ── graph export ───────────────────────────────────────────────────────────────

def export_graph(
    coords: np.ndarray,
    edges: np.ndarray,
    out_path: Path,
    feature_dim: int = 1536,
) -> None:
    """
    Export the graph to a .pt file ready for torch_geometric.

    The file contains a dict with:
      x          : (N, feature_dim) float32 — artificial node features
      edge_index : (2, 2*M) int64  — bidirectional edge list (undirected graph)
      pos        : (N, 2) float32  — (j, i) WSI level-0 coordinates per node
      num_nodes  : int

    Load example::

        data_dict = torch.load("graph.pt")
        from torch_geometric.data import Data
        data = Data(**data_dict)
    """
    n_nodes = len(coords)
    x = torch.randn(n_nodes, feature_dim)
    pos = torch.tensor(coords, dtype=torch.float32)

    if len(edges) > 0:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, M)
        edge_index = torch.cat([ei, ei.flip(0)], dim=1)              # (2, 2M)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = {
        "x": x,
        "edge_index": edge_index,
        "pos": pos,
        "num_nodes": n_nodes,
    }

    graph_path = out_path.with_suffix(".pt")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, graph_path)
    print(
        f"[INFO] Graph export : {graph_path.resolve()}"
        f"  ({n_nodes} nodes, {edge_index.shape[1]} directed edges, {feature_dim} features/node)"
    )


# ── canvas reconstruction ──────────────────────────────────────────────────────

def build_canvas(
    images: list[np.ndarray],
    coords: np.ndarray,
    patch_size: int = 2048,
    max_side: int = 4096,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Assemble patches onto a white canvas.

    Returns
    -------
    canvas    : (H, W, 3) uint8 array
    has_patch : (H, W) bool array — True where a patch was placed (used for transparency)
    j_min     : left-most j coordinate in WSI level-0 pixels
    i_min     : top-most  i coordinate in WSI level-0 pixels
    scale     : pixel-scaling factor applied to fit within max_side
    """
    j_coords = coords[:, 0]
    i_coords = coords[:, 1]
    j_min = j_coords.min()
    i_min = i_coords.min()
    j_max = j_coords.max() + patch_size
    i_max = i_coords.max() + patch_size

    width_px  = int(j_max - j_min)
    height_px = int(i_max - i_min)

    scale    = min(1.0, max_side / max(width_px, height_px))
    canvas_w = max(1, int(width_px  * scale))
    canvas_h = max(1, int(height_px * scale))
    ps       = max(1, int(patch_size * scale))

    canvas    = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    has_patch = np.zeros((canvas_h, canvas_w), dtype=bool)

    for img, (j, i) in zip(images, coords):
        x  = int((j - j_min) * scale)
        y  = int((i - i_min) * scale)
        x2 = min(x + ps, canvas_w)
        y2 = min(y + ps, canvas_h)
        resized = np.array(Image.fromarray(img).resize((ps, ps), Image.LANCZOS))
        canvas[y:y2, x:x2]    = resized[: y2 - y, : x2 - x]
        has_patch[y:y2, x:x2] = True

    return canvas, has_patch, j_min, i_min, scale


# ── plotting ───────────────────────────────────────────────────────────────────

def _bg_extent(
    j_min: float, i_min: float, scale: float,
    slide_meta: dict | None, canvas_w: int, canvas_h: int,
) -> list[float]:
    """
    Compute imshow extent [left, right, bottom, top] in canvas pixel coordinates
    for an image that covers the full WSI slide crop.

    matplotlib's imshow extent convention (with origin="upper"):
      extent = [xmin, xmax, ymax, ymin]  (ymax > ymin, since y increases downward)
    """
    if slide_meta is not None:
        x0 = (slide_meta["j_base"] - j_min) * scale
        x1 = (slide_meta["j_base"] + slide_meta["w"] - j_min) * scale
        y0 = (slide_meta["i_base"] - i_min) * scale          # top  (smaller y)
        y1 = (slide_meta["i_base"] + slide_meta["h"] - i_min) * scale  # bottom (larger y)
    else:
        # Fallback: stretch background to the full canvas
        x0, x1, y0, y1 = 0.0, float(canvas_w), 0.0, float(canvas_h)
    return [x0, x1, y1, y0]  # [left, right, bottom, top]


def render(
    canvas: np.ndarray,
    has_patch: np.ndarray,
    coords: np.ndarray,
    edges: np.ndarray,
    removed_edges: np.ndarray,
    feat_dim0: np.ndarray,
    non_white: np.ndarray,
    j_min: float,
    i_min: float,
    scale: float,
    patch_size: int,
    title: str,
    out_path: Path,
    mask_img: np.ndarray | None = None,
    rgb_img: np.ndarray | None = None,
    slide_meta: dict | None = None,
    overlay_mode: str = "mask",
) -> None:
    """
    Draw the two-panel figure and save it.

    Left panel  (back → front): RGB image @ 0.5 · patch collage @ 1.0
    Right panel (back → front): patch collage @ 1.0 · [mask|rgb] @ 0.5 · graph
    """
    half_patch = (patch_size * scale) / 2

    # Canvas-space centre of each patch
    cx = (coords[:, 0] - j_min) * scale + half_patch
    cy = (coords[:, 1] - i_min) * scale + half_patch

    canvas_h, canvas_w = canvas.shape[:2]

    # RGBA canvas: patch pixels fully opaque, gaps transparent
    canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas_rgba[..., :3] = canvas
    canvas_rgba[..., 3]  = np.where(has_patch, 255, 0)

    # Background image for the right panel
    bg_img = mask_img if overlay_mode == "mask" else rgb_img

    # Extent of background images in canvas pixel coordinates
    ext = _bg_extent(j_min, i_min, scale, slide_meta, canvas_w, canvas_h)

    node_sizes = 4

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=10, y=0.995)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

    # ── left panel: RGB @ 0.5 | patches @ 1.0 ────────────────────────────────
    if rgb_img is not None:
        axes[0].imshow(
            rgb_img, extent=ext, origin="upper", aspect="auto", alpha=0.5, zorder=1,
        )
    axes[0].imshow(canvas_rgba, origin="upper", zorder=2)
    axes[0].set_xlim(-0.5, canvas_w - 0.5)
    axes[0].set_ylim(canvas_h - 0.5, -0.5)
    axes[0].set_title("Slide Reconstruction", color="white", fontsize=12, pad=6)

    # ── right panel: patches @ 1.0 | [mask/rgb] @ 0.5 | graph ───────────────
    axes[1].imshow(canvas, origin="upper", zorder=1)
    if bg_img is not None:
        axes[1].imshow(
            bg_img, extent=ext, origin="upper", aspect="auto", alpha=0.5, zorder=2,
        )
    axes[1].set_aspect("equal")  # restore after aspect="auto" from bg overlay
    axes[1].set_xlim(-0.5, canvas_w - 0.5)
    axes[1].set_ylim(canvas_h - 0.5, -0.5)
    axes[1].set_title(
        f"Delaunay Graph  [{overlay_mode} overlay]", color="white", fontsize=12, pad=6
    )

    # Removed edges (crossed non-tissue mask region) — drawn in red
    for u, v in removed_edges:
        axes[1].plot(
            [cx[u], cx[v]], [cy[u], cy[v]],
            color="red", alpha=0.6, linewidth=0.8, zorder=3,
        )

    # Kept edges
    for u, v in edges:
        axes[1].plot(
            [cx[u], cx[v]], [cy[u], cy[v]],
            color="black", alpha=0.8, linewidth=1.0, zorder=3,
        )

    # Nodes
    axes[1].scatter(
        cx, cy,
        color="black",
        s=node_sizes,
        alpha=0.85,
        linewidths=0,
        zorder=4,
    )

    # # Nodes coloured by first feature dimension (uncomment to enable)
    # sc = axes[1].scatter(
    #     cx, cy, c=feat_dim0, cmap="plasma", s=node_sizes,
    #     alpha=0.85, linewidths=0.4, edgecolors="white", zorder=4,
    # )
    # cbar = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.02)
    # cbar.set_label("Feature dim 0", color="white", fontsize=8)
    # cbar.ax.yaxis.set_tick_params(color="white")
    # plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delaunay graph visualization over a WSI slide.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iam_path",
        default="/mnt/iam",
        help="Path to the dataset root (/mnt/iam) or the Images/ folder",
    )
    p.add_argument("--hospital",    default=None, help="Hospital name (first available if omitted)")
    p.add_argument("--patient_id",  default=None, help="Patient ID (best slide if omitted)")
    p.add_argument("--slide_id",    default=None, help="Slide ID (best slide if omitted)")
    p.add_argument("--output",      default="outputs/delaunay_overlay.png",
                   help="Output PNG path")
    p.add_argument("--max_patches", type=int, default=500,
                   help="Maximum number of patches to load (for speed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iam_path = Path(args.iam_path)

    # ── locate Patches directory ───────────────────────────────────────────────
    patches_dir = find_patches_dir(iam_path)

    # ── select hospital ────────────────────────────────────────────────────────
    hospitals = sorted(p.name for p in patches_dir.iterdir() if p.is_dir())
    if not hospitals:
        sys.exit(f"[ERROR] No hospital directories found in {patches_dir}")

    hospital = args.hospital or hospitals[0]
    if hospital not in hospitals:
        sys.exit(
            f"[ERROR] Hospital '{hospital}' not found.\n"
            f"Available hospitals:\n  " + "\n  ".join(hospitals)
        )
    print(f"[INFO] Hospital : {hospital}")

    # ── load & quality-filter metadata ────────────────────────────────────────
    df = load_metadata(patches_dir, hospital)
    # df_filtered = filter_patches(df)
    df_filtered = df
    print(f"[INFO] Patches after quality filter: {len(df_filtered)}")
    if df_filtered.empty:
        sys.exit("[ERROR] No patches passed the quality filter.")

    # ── select patient / slide ─────────────────────────────────────────────────
    if args.patient_id and args.slide_id:
        patient_id = str(args.patient_id)
        slide_id   = str(args.slide_id)
    elif args.patient_id:
        patient_id = str(args.patient_id)
        df_pat = df_filtered[df_filtered["patient_ID"].astype(str) == patient_id]
        if df_pat.empty:
            sys.exit(f"[ERROR] Patient '{patient_id}' not found after quality filter.")
        slide_id = str(df_pat.groupby("slide_ID").size().idxmax())
    else:
        patient_id, slide_id = select_best_slide(df_filtered)

    df_slide = df_filtered[
        (df_filtered["patient_ID"].astype(str) == patient_id) &
        (df_filtered["slide_ID"].astype(str)   == slide_id)
    ]
    if df_slide.empty:
        sys.exit(f"[ERROR] No filtered patches for patient={patient_id}, slide={slide_id}")

    print(f"[INFO] Patient  : {patient_id}")
    print(f"[INFO] Slide    : {slide_id}")
    print(f"[INFO] Patches  : {len(df_slide)} (will load up to {args.max_patches})")

    # ── load patch images ──────────────────────────────────────────────────────
    images, coords, non_white = load_patches(
        patches_dir, df_slide.copy(), hospital, patient_id, slide_id, args.max_patches
    )
    if len(images) < 3:
        sys.exit(
            f"[ERROR] Only {len(images)} patch(es) loaded — need at least 3 for Delaunay. "
            "Check that the patch files exist under the expected path."
        )
    print(f"[INFO] Loaded   : {len(images)} patch images")

    # ── load slide-level metadata (for background alignment) ──────────────────
    slide_meta = load_slide_meta(iam_path, hospital, patient_id, slide_id)

    # ── load background images ─────────────────────────────────────────────────
    # Mask is always loaded: used for edge filtering regardless of OVERLAY_MODE
    mask_img = load_mask_image(iam_path, hospital, patient_id, slide_id)
    # RGB image for left panel (and right panel if OVERLAY_MODE == "rgb")
    rgb_img  = load_rgb_image(iam_path, hospital, patient_id, slide_id)

    # ── random feature placeholder (1536-dim per node) ────────────────────────
    features  = torch.randn(len(images), 1536)
    feat_dim0 = features[:, 0].numpy()

    # ── build Delaunay graph ───────────────────────────────────────────────────
    edges, _ = build_delaunay_edges(coords, distance_factor=2.0)
    print(f"[INFO] Graph    : {len(images)} nodes | {len(edges)} edges (before mask filter)")

    # ── filter edges that cross non-tissue mask regions ────────────────────────
    removed_edges = np.empty((0, 2), dtype=np.int64)
    if mask_img is not None and slide_meta is not None:
        edges, removed_edges = filter_edges_by_mask(
            edges, coords, mask_img,
            j_base=slide_meta["j_base"],
            i_base=slide_meta["i_base"],
            slide_w=slide_meta["w"],
            slide_h=slide_meta["h"],
            patch_size=2048,
        )
        print(
            f"[INFO] Edges after mask filter: {len(edges)} kept, {len(removed_edges)} removed"
        )
    else:
        if mask_img is None:
            print("[WARN] Mask not available — skipping edge mask filter.")
        if slide_meta is None:
            print("[WARN] Slide metadata not available — skipping edge mask filter.")

    # ── export graph for torch_geometric ──────────────────────────────────────
    export_graph(coords, edges, out_path=Path(args.output))

    # ── build slide canvas ─────────────────────────────────────────────────────
    canvas, has_patch, j_min, i_min, scale = build_canvas(images, coords)

    # ── render & save ──────────────────────────────────────────────────────────
    title = (
        f"Hospital: {hospital}  |  Patient: {patient_id}  |  Slide: {slide_id}  |  "
        f"{len(images)} nodes  |  {len(edges)} edges kept  |  {len(removed_edges)} removed  |  "
        f"overlay: {OVERLAY_MODE}"
    )
    out_path = Path(args.output)
    render(
        canvas=canvas,
        has_patch=has_patch,
        coords=coords,
        edges=edges,
        removed_edges=removed_edges,
        feat_dim0=feat_dim0,
        non_white=non_white,
        j_min=j_min,
        i_min=i_min,
        scale=scale,
        patch_size=2048,
        title=title,
        out_path=out_path,
        mask_img=mask_img,
        rgb_img=rgb_img,
        slide_meta=slide_meta,
        overlay_mode=OVERLAY_MODE,
    )


if __name__ == "__main__":
    main()
