#!/usr/bin/env python3
"""
Delaunay graph visualization over a single WSI slide.

Creates a figure with two panels:
  Left:  approximate slide reconstruction from individual 2048×2048 patches.
  Right: same reconstruction with the Delaunay graph superimposed.

Usage example (defaults work out of the box):
    python scripts/delaunay_visualization.py

Custom example:
    python scripts/delaunay_visualization.py \
        --hospital "H. Bellvitge" \
        --max_patches 300 \
        --output outputs/bellvitge_delaunay.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import Delaunay

# ── dataset helpers ───────────────────────────────────────────────────────────

PATCHES_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches_2048"
)


def find_patches_dir(iam_path: Path) -> Path:
    """Locate the Patches directory starting from iam_path."""
    # Accept either the Images folder or the full /mnt/iam root
    for candidate in [
        iam_path / "Patches",
        iam_path / PATCHES_SUBPATH,
    ]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find a 'Patches' directory under '{iam_path}'.\n"
        "Pass --iam_path pointing to the dataset root (/mnt/iam) or the "
        "Images/ folder."
    )


def load_metadata(patches_dir: Path, hospital: str) -> pd.DataFrame:
    meta_path = patches_dir / hospital / f"metadata_{hospital}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    df = pd.read_csv(meta_path)
    for col in ["i", "j", "blurriness", "non_white_area", "affected_percentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Quality filter: discard blank and blurry patches."""
    mask = (df["non_white_area"] >= 0.3) & (df["blurriness"] <= 100)
    return df[mask].copy()


def select_best_slide(df: pd.DataFrame) -> tuple[str, str]:
    """Return (patient_id, slide_id) for the slide with the most patches."""
    counts = df.groupby(["patient_ID", "slide_ID"]).size()
    patient_id, slide_id = counts.idxmax()
    return str(patient_id), str(slide_id)


# ── patch loading ─────────────────────────────────────────────────────────────

def _patch_filename(hospital: str, patient: str, slide: str,
                    i: int, j: int) -> str:
    return f"{hospital}_{patient}_{slide}_{i}_{j}.jpg"


def load_patches(
    patches_dir: Path,
    df_slide: pd.DataFrame,
    hospital: str,
    patient: str,
    slide: str,
    max_patches: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load up to *max_patches* PNG images for one slide.

    Returns
    -------
    images      : list of (2048, 2048, 3) uint8 arrays
    coords      : (N, 2) float array — columns are (j, i) = (x, y) in WSI pixels
    non_white   : (N,) float array — non_white_area metric per loaded patch
    """
    if len(df_slide) > max_patches:
        df_slide = df_slide.sample(n=max_patches, random_state=42).reset_index(drop=True)

    slide_dir = patches_dir / hospital / patient / slide

    images: list[np.ndarray] = []
    coords_list: list[tuple[float, float]] = []
    non_white_list: list[float] = []

    for _, row in df_slide.iterrows():
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

    coords = np.array(coords_list, dtype=np.float64)      # (N, 2)
    non_white = np.array(non_white_list, dtype=np.float64)  # (N,)
    return images, coords, non_white


# ── graph construction ────────────────────────────────────────────────────────

def build_delaunay_edges(
    coords: np.ndarray,
    distance_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delaunay triangulation + long-edge pruning.

    Parameters
    ----------
    coords          : (N, 2) array of (x, y) positions
    distance_factor : edges longer than (factor × mean neighbour distance)
                      are removed to avoid connections across empty tissue areas.

    Returns
    -------
    edges   : (M, 2) int array of node-index pairs
    lengths : (M,)   float array of edge lengths in pixels
    """
    tri = Delaunay(coords)

    # Collect unique edges from simplices
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = simplex[a], simplex[b]
            edge_set.add((min(u, v), max(u, v)))

    edges = np.array(list(edge_set), dtype=np.int64)          # (M_all, 2)
    lengths = np.linalg.norm(
        coords[edges[:, 0]] - coords[edges[:, 1]], axis=1
    )

    threshold = distance_factor * lengths.mean()
    keep = lengths <= threshold
    return edges[keep], lengths[keep]


# ── canvas reconstruction ─────────────────────────────────────────────────────

def build_canvas(
    images: list[np.ndarray],
    coords: np.ndarray,
    patch_size: int = 2048,
    max_side: int = 4096,
) -> tuple[np.ndarray, float, float, float]:
    """
    Assemble patches onto a white canvas.

    Returns
    -------
    canvas  : (H, W, 3) uint8 array
    j_min   : left-most j coordinate (used to convert WSI coords → canvas pixels)
    i_min   : top-most  i coordinate
    scale   : pixel-scaling factor applied to fit within max_side
    """
    j_coords = coords[:, 0]
    i_coords = coords[:, 1]
    j_min = j_coords.min()
    i_min = i_coords.min()
    j_max = j_coords.max() + patch_size
    i_max = i_coords.max() + patch_size

    width_px  = int(j_max - j_min)
    height_px = int(i_max - i_min)

    scale = min(1.0, max_side / max(width_px, height_px))
    canvas_w = max(1, int(width_px  * scale))
    canvas_h = max(1, int(height_px * scale))
    ps = max(1, int(patch_size * scale))

    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)

    for img, (j, i) in zip(images, coords):
        x  = int((j - j_min) * scale)
        y  = int((i - i_min) * scale)
        x2 = min(x + ps, canvas_w)
        y2 = min(y + ps, canvas_h)
        resized = np.array(Image.fromarray(img).resize((ps, ps), Image.LANCZOS))
        canvas[y:y2, x:x2] = resized[: y2 - y, : x2 - x]

    return canvas, j_min, i_min, scale


# ── plotting ──────────────────────────────────────────────────────────────────

def render(
    canvas: np.ndarray,
    coords: np.ndarray,
    edges: np.ndarray,
    feat_dim0: np.ndarray,
    non_white: np.ndarray,
    j_min: float,
    i_min: float,
    scale: float,
    patch_size: int,
    title: str,
    out_path: Path,
) -> None:
    """Draw the two-panel figure and save it."""
    half_patch = (patch_size * scale) / 2

    # Canvas-space centre of each patch (for node plotting)
    cx = (coords[:, 0] - j_min) * scale + half_patch
    cy = (coords[:, 1] - i_min) * scale + half_patch

    # Node sizes proportional to tissue content
    node_sizes = 15 + 70 * np.clip(non_white, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=10, y=0.995)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

    # ── left panel: slide reconstruction ─────────────────────────────────────
    axes[0].imshow(canvas, origin="upper")
    axes[0].set_title("Slide Reconstruction", color="white", fontsize=12, pad=6)

    # ── right panel: Delaunay overlay ─────────────────────────────────────────
    axes[1].imshow(canvas, origin="upper")
    axes[1].set_title("Delaunay Graph Overlay", color="white", fontsize=12, pad=6)

    # Edges
    for u, v in edges:
        axes[1].plot(
            [cx[u], cx[v]], [cy[u], cy[v]],
            color="white", alpha=0.4, linewidth=0.5, zorder=2,
        )

    # Nodes (coloured by first feature dimension)
    sc = axes[1].scatter(
        cx, cy,
        c=feat_dim0,
        cmap="plasma",
        s=node_sizes,
        alpha=0.85,
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_label("Feature dim 0 (random placeholder)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

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
                   help="Output PNG path (relative paths resolved from CWD)")
    p.add_argument("--max_patches", type=int, default=500,
                   help="Maximum number of patches to load (for speed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iam_path = Path(args.iam_path)

    # ── locate Patches directory ──────────────────────────────────────────────
    patches_dir = find_patches_dir(iam_path)

    # ── select hospital ───────────────────────────────────────────────────────
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

    # ── load & quality-filter metadata ───────────────────────────────────────
    df = load_metadata(patches_dir, hospital)
    df_filtered = filter_patches(df)
    print(f"[INFO] Patches after quality filter: {len(df_filtered)}")
    if df_filtered.empty:
        sys.exit("[ERROR] No patches passed the quality filter.")

    # ── select patient / slide ────────────────────────────────────────────────
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

    # ── load patch images ─────────────────────────────────────────────────────
    images, coords, non_white = load_patches(
        patches_dir, df_slide.copy(), hospital, patient_id, slide_id, args.max_patches
    )
    if len(images) < 3:
        sys.exit(
            f"[ERROR] Only {len(images)} patch(es) loaded — need at least 3 for Delaunay. "
            "Check that the PNG files exist under the expected path."
        )
    print(f"[INFO] Loaded   : {len(images)} patch images")

    # ── random feature placeholder (128-dim per node) ─────────────────────────
    features  = torch.randn(len(images), 128)
    feat_dim0 = features[:, 0].numpy()

    # ── build Delaunay graph ──────────────────────────────────────────────────
    edges, _ = build_delaunay_edges(coords, distance_factor=2.0)
    print(f"[INFO] Graph    : {len(images)} nodes | {len(edges)} edges")

    # ── build slide canvas ────────────────────────────────────────────────────
    canvas, j_min, i_min, scale = build_canvas(images, coords)

    # ── render & save ─────────────────────────────────────────────────────────
    title = (
        f"Hospital: {hospital}  |  Patient: {patient_id}  |  Slide: {slide_id}  |  "
        f"{len(images)} nodes  |  {len(edges)} edges"
    )
    out_path = Path(args.output)
    render(
        canvas=canvas,
        coords=coords,
        edges=edges,
        feat_dim0=feat_dim0,
        non_white=non_white,
        j_min=j_min,
        i_min=i_min,
        scale=scale,
        patch_size=2048,
        title=title,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()