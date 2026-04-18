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

─── slide selection ────────────────────────────────────────────────────────────
  (no flags)                     →  best slide of the first hospital
  --hospital H                   →  best slide of hospital H
  --patient_id P                 →  best slide of patient P (within hospital H)
  --slide_id S                   →  specific slide (patient looked up automatically)
  --patient_id P --slide_id S    →  specific patient + slide
  --list                         →  print available patients/slides and exit

─── node filtering ─────────────────────────────────────────────────────────────
  --filtered    Keep only the patches that have a CLS embedding in the NPZ.
                Each bag centroid in the NPZ is matched to its nearest patch in
                the metadata CSV; unmatched patches are discarded.
                Use this to reproduce exactly the same set of nodes as the .pt
                training graphs (and the frontend visualisation).

─── usage examples ─────────────────────────────────────────────────────────────
  # default: picks the slide with the most patches automatically
  python scripts/delaunay_vis.py

  # list all patients/slides for a hospital to find valid IDs
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --list

  # specific patient (best slide chosen automatically)
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --patient_id "12345"

  # specific patient + slide
  python scripts/delaunay_vis.py \\
      --hospital "H. Bellvitge" \\
      --patient_id "12345" --slide_id "12345_A1" \\
      --output outputs/12345_A1_delaunay.png

  # slide alone (patient resolved automatically)
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --slide_id "12345_A1"

  # only show nodes that exist in the CLS NPZ (same as frontend)
  python scripts/delaunay_vis.py \\
      --hospital "H. Bellvitge" --slide_id "12345_A1" --filtered
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── local imports ──────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from graph_utils import build_delaunay_edges, filter_edges_by_mask, export_graph  # noqa: E402
from wsi_io import (  # noqa: E402
    CLS_DIR_SUBPATH,
    find_patches_dir,
    load_all_npz,
    load_slide_meta,
    load_mask_image,
    load_rgb_image,
)

# ── single-line toggle ─────────────────────────────────────────────────────────
OVERLAY_MODE = "mask"   # "mask" → segmentation mask | "rgb" → full RGB image


# ── slide selection helpers ────────────────────────────────────────────────────

def _estimate_patch_size(coord_arrays: np.ndarray) -> int:
    """Estimate WSI level-0 patch size from the coordinate step in one bag."""
    bag = coord_arrays[0]   # (256, 2)
    js  = np.unique(bag[:, 0])
    if len(js) >= 2:
        return int(abs(js[1] - js[0]))
    is_ = np.unique(bag[:, 1])
    if len(is_) >= 2:
        return int(abs(is_[1] - is_[0]))
    return 256   # fallback


# ── canvas reconstruction ──────────────────────────────────────────────────────

def build_canvas(
    images: list,
    coords: np.ndarray,
    patch_size: int = 2048,
    max_side: int = 4096,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Assemble patch images onto a white canvas, scaled to fit within max_side.

    Parameters
    ----------
    images     : list of (2048, 2048, 3) uint8 arrays
    coords     : (N, 2) float array — (j, i) WSI level-0 pixel coordinates
    patch_size : original patch side length in WSI level-0 pixels
    max_side   : maximum canvas dimension in output pixels

    Returns
    -------
    canvas    : (H, W, 3) uint8 array
    has_patch : (H, W) bool array — True where a patch was placed
    j_min     : left-most j coordinate in WSI level-0 pixels
    i_min     : top-most  i coordinate in WSI level-0 pixels
    scale     : scaling factor applied to fit within max_side
    """
    from PIL import Image as _Image

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
        resized = np.array(_Image.fromarray(img).resize((ps, ps), _Image.LANCZOS))
        canvas[y:y2, x:x2]    = resized[: y2 - y, : x2 - x]
        has_patch[y:y2, x:x2] = True

    return canvas, has_patch, j_min, i_min, scale


# ── plotting ───────────────────────────────────────────────────────────────────

def _bg_extent(
    j_min: float, i_min: float, scale: float,
    slide_meta: "dict | None", canvas_w: int, canvas_h: int,
) -> list[float]:
    """
    Compute imshow extent [left, right, bottom, top] in canvas pixel coordinates
    for a background image covering the full WSI slide crop.

    matplotlib imshow extent with origin="upper": [xmin, xmax, ymax, ymin]
    """
    if slide_meta is not None:
        x0 = (slide_meta["j_base"] - j_min) * scale
        x1 = (slide_meta["j_base"] + slide_meta["w"] - j_min) * scale
        y0 = (slide_meta["i_base"] - i_min) * scale
        y1 = (slide_meta["i_base"] + slide_meta["h"] - i_min) * scale
    else:
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
    mask_img: "np.ndarray | None" = None,
    rgb_img: "np.ndarray | None" = None,
    slide_meta: "dict | None" = None,
    overlay_mode: str = "mask",
) -> None:
    """
    Draw the two-panel figure and save it.

    Left panel  (back → front): RGB image @ 0.5 · patch collage @ 1.0
    Right panel (back → front): patch collage @ 1.0 · [mask|rgb] @ 0.5 · graph
    """
    half_patch = (patch_size * scale) / 2

    cx = (coords[:, 0] - j_min) * scale + half_patch
    cy = (coords[:, 1] - i_min) * scale + half_patch

    canvas_h, canvas_w = canvas.shape[:2]

    canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas_rgba[..., :3] = canvas
    canvas_rgba[..., 3]  = np.where(has_patch, 255, 0)

    bg_img = mask_img if overlay_mode == "mask" else rgb_img
    ext    = _bg_extent(j_min, i_min, scale, slide_meta, canvas_w, canvas_h)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=10, y=0.995)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

    # ── left panel: RGB @ 0.5 | patches @ 1.0 ────────────────────────────────
    if rgb_img is not None:
        axes[0].imshow(rgb_img, extent=ext, origin="upper", aspect="auto", alpha=0.5, zorder=1)
    axes[0].imshow(canvas_rgba, origin="upper", zorder=2)
    axes[0].set_xlim(-0.5, canvas_w - 0.5)
    axes[0].set_ylim(canvas_h - 0.5, -0.5)
    axes[0].set_title("Slide Reconstruction", color="white", fontsize=12, pad=6)

    # ── right panel: patches @ 1.0 | [mask/rgb] @ 0.5 | graph ───────────────
    axes[1].imshow(canvas, origin="upper", zorder=1)
    if bg_img is not None:
        axes[1].imshow(bg_img, extent=ext, origin="upper", aspect="auto", alpha=0.5, zorder=2)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-0.5, canvas_w - 0.5)
    axes[1].set_ylim(canvas_h - 0.5, -0.5)
    axes[1].set_title(
        f"Delaunay Graph  [{overlay_mode} overlay]", color="white", fontsize=12, pad=6
    )

    # Removed edges (crossed non-tissue) — drawn in red
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
    axes[1].scatter(cx, cy, color="black", s=4, alpha=0.85, linewidths=0, zorder=4)

    # # Nodes coloured by first feature dimension (uncomment to enable)
    # sc = axes[1].scatter(
    #     cx, cy, c=feat_dim0, cmap="plasma", s=4,
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
    p.add_argument("--iam_path",    default="/mnt/iam",
                   help="Path to the dataset root (/mnt/iam) or the Images/ folder")
    p.add_argument("--hospital",    default=None,
                   help="Hospital name (first available if omitted)")
    p.add_argument("--patient_id",  default=None,
                   help="Patient ID; best slide chosen if --slide_id is omitted")
    p.add_argument("--slide_id",    default=None,
                   help="Slide ID; patient resolved automatically if --patient_id is omitted")
    p.add_argument("--list",        action="store_true",
                   help="List available patients and slides for the selected hospital and exit")
    p.add_argument("--filtered",    action="store_true",
                   help="(Legacy flag — ignored; nodes always come from CLS NPZ)")
    p.add_argument("--output",      default="outputs/delaunay_overlay.png",
                   help="Output PNG path")
    p.add_argument("--max_patches", type=int, default=500,
                   help="Maximum number of patches to load (for speed)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    iam_path = Path(args.iam_path)

    # ── locate Patches directory ───────────────────────────────────────────────
    patches_dir = find_patches_dir(iam_path)

    # ── load NPZ (source of truth for nodes + image paths) ────────────────────
    cls_dir = iam_path / CLS_DIR_SUBPATH
    df_npz  = load_all_npz(cls_dir)

    # ── select hospital ────────────────────────────────────────────────────────
    hospitals = sorted(df_npz["Hospital"].unique())
    hospital  = args.hospital or hospitals[0]
    if hospital not in hospitals:
        sys.exit(
            f"[ERROR] Hospital '{hospital}' not found in NPZ.\n"
            "Available hospitals:\n  " + "\n  ".join(hospitals)
        )
    print(f"[INFO] Hospital : {hospital}")

    df_hosp = df_npz[df_npz["Hospital"] == hospital]

    # ── --list: print available patients/slides and exit ──────────────────────
    if args.list:
        summary = (
            df_hosp.groupby(["Patient_ID", "Slide"])
            .size()
            .reset_index(name="bags")
            .sort_values(["Patient_ID", "Slide"])
        )
        print(f"\nAvailable patients/slides for hospital '{hospital}':\n")
        print(f"  {'Patient_ID':<20} {'Slide':<30} bags")
        print(f"  {'-'*20} {'-'*30} ----")
        for _, row in summary.iterrows():
            print(f"  {str(row['Patient_ID']):<20} {str(row['Slide']):<30} {row['bags']}")
        sys.exit(0)

    # ── select patient / slide ─────────────────────────────────────────────────
    if args.patient_id and args.slide_id:
        patient_id = str(args.patient_id)
        slide_id   = str(args.slide_id)
    elif args.patient_id:
        patient_id = str(args.patient_id)
        df_pat     = df_hosp[df_hosp["Patient_ID"] == patient_id]
        if df_pat.empty:
            sys.exit(f"[ERROR] Patient '{patient_id}' not found in NPZ for this hospital.")
        slide_id = str(df_pat.groupby("Slide").size().idxmax())
    elif args.slide_id:
        slide_id   = str(args.slide_id)
        df_sl      = df_hosp[df_hosp["Slide"] == slide_id]
        if df_sl.empty:
            sys.exit(f"[ERROR] Slide '{slide_id}' not found in NPZ for this hospital.")
        patient_id = str(df_sl.groupby("Patient_ID").size().idxmax())
    else:
        counts     = df_hosp.groupby(["Patient_ID", "Slide"]).size()
        patient_id, slide_id = counts.idxmax()

    df_slide = df_hosp[
        (df_hosp["Patient_ID"] == patient_id) &
        (df_hosp["Slide"]      == slide_id)
    ]
    if df_slide.empty:
        sys.exit(f"[ERROR] No bags in NPZ for patient={patient_id}, slide={slide_id}")

    print(f"[INFO] Patient  : {patient_id}")
    print(f"[INFO] Slide    : {slide_id}")
    print(f"[INFO] Bags     : {len(df_slide)} (will load up to {args.max_patches})")

    if len(df_slide) > args.max_patches:
        df_slide = df_slide.sample(n=args.max_patches, random_state=42).reset_index(drop=True)

    # ── compute bag centroids (node positions) ─────────────────────────────────
    coord_arrays = np.stack(df_slide["coords_bag"].tolist(), axis=0)  # (N, 256, 2)
    centroids    = coord_arrays.mean(axis=1)                           # (N, 2)

    # ── load one representative patch image per bag ────────────────────────────
    from PIL import Image as _PILImage
    slide_dir   = patches_dir / hospital / patient_id / slide_id
    images:     list[np.ndarray]        = []
    coords_out: list[np.ndarray]        = []
    non_white:  list[float]             = []

    for n in range(len(df_slide)):
        bag_coords = coord_arrays[n]
        centroid   = centroids[n]
        dists      = np.linalg.norm(bag_coords - centroid, axis=1)
        central_idx = int(dists.argmin())

        path_str  = str(np.array(df_slide.iloc[n]["paths_bag"])[central_idx])
        basename  = path_str.replace("\\", "/").split("/")[-1]
        img_path  = slide_dir / basename

        if not img_path.exists():
            continue
        try:
            img = np.array(_PILImage.open(img_path).convert("RGB"))
        except Exception:
            continue

        images.append(img)
        coords_out.append(centroids[n])
        non_white.append(1.0)   # no quality metadata in Patches/; treat all as valid

    coords = np.array(coords_out)

    if len(images) < 3:
        sys.exit(
            f"[ERROR] Only {len(images)} patch(es) loaded — need at least 3 for Delaunay.\n"
            f"  Slide dir : {slide_dir}\n"
            f"  Exists    : {slide_dir.exists()}"
        )
    print(f"[INFO] Loaded   : {len(images)} patch images")

    # detect patch size from coord spacing (used for canvas assembly and edge filter)
    patch_size = _estimate_patch_size(coord_arrays)

    # ── load slide-level metadata (for background alignment) ──────────────────
    slide_meta = load_slide_meta(iam_path, hospital, patient_id, slide_id)

    # ── load background images ─────────────────────────────────────────────────
    mask_img = load_mask_image(iam_path, hospital, patient_id, slide_id)
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
            patch_size=patch_size,
        )
        print(f"[INFO] Edges after mask filter: {len(edges)} kept, {len(removed_edges)} removed")
    else:
        if mask_img is None:
            print("[WARN] Mask not available — skipping edge mask filter.")
        if slide_meta is None:
            print("[WARN] Slide metadata not available — skipping edge mask filter.")

    # ── export graph for torch_geometric ──────────────────────────────────────
    export_graph(coords, edges, out_path=Path(args.output))

    # ── build slide canvas ─────────────────────────────────────────────────────
    canvas, has_patch, j_min, i_min, scale = build_canvas(images, coords, patch_size=patch_size)

    # ── render & save ──────────────────────────────────────────────────────────
    title = (
        f"Hospital: {hospital}  |  Patient: {patient_id}  |  Slide: {slide_id}  |  "
        f"{len(images)} nodes  |  {len(edges)} edges kept  |  {len(removed_edges)} removed  |  "
        f"overlay: {OVERLAY_MODE}"
    )
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
        patch_size=patch_size,
        title=title,
        out_path=Path(args.output),
        mask_img=mask_img,
        rgb_img=rgb_img,
        slide_meta=slide_meta,
        overlay_mode=OVERLAY_MODE,
    )


if __name__ == "__main__":
    main()
