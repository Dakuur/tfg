#!/usr/bin/env python3
"""
Delaunay graph visualization over a single WSI slide.

Shows one panel: segmentation mask (or full RGB) as background at 0.5 opacity,
with the Delaunay graph superimposed. Nodes are rendered as dots — no patch
images are loaded. Use patch_vis.py for the patch-image visualization.

─── slide selection ────────────────────────────────────────────────────────────
  (no flags)                     →  best slide of the first hospital
  --hospital H                   →  best slide of hospital H
  --patient_id P                 →  best slide of patient P
  --slide_id S                   →  specific slide (patient resolved automatically)
  --patient_id P --slide_id S    →  specific patient + slide
  --list                         →  print available patients/slides and exit

─── usage examples ─────────────────────────────────────────────────────────────
  python scripts/delaunay_vis.py
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --list
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --patient_id "12345"
  python scripts/delaunay_vis.py --hospital "H. Bellvitge" --slide_id "12345_A1"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from graph_utils import build_delaunay_edges, filter_edges_by_mask, export_graph  # noqa: E402
from wsi_io import (  # noqa: E402
    CLS_DIR_SUBPATH,
    load_all_npz,
    load_slide_meta,
    load_mask_image,
    load_rgb_image,
)

OVERLAY_MODE = "mask"   # "mask" → segmentation mask | "rgb" → full RGB image


# ── plotting ───────────────────────────────────────────────────────────────────

def render(
    coords: np.ndarray,
    edges: np.ndarray,
    removed_edges: np.ndarray,
    title: str,
    out_path: Path,
    rgb_img: np.ndarray,
    slide_meta: dict,
    mask_img: "np.ndarray | None" = None,
    overlay_mode: str = "mask",
) -> None:
    """Single-panel figure: RGB slide + optional mask overlay + Delaunay graph."""
    rgb_h, rgb_w = rgb_img.shape[:2]
    scale  = rgb_w / slide_meta["w"]
    j_base = slide_meta["j_base"]
    i_base = slide_meta["i_base"]

    # WSI level-0 coords → display pixel coords (same transform as patch_vis.py)
    cx = (coords[:, 0] - j_base) * scale
    cy = (coords[:, 1] - i_base) * scale

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=9, pad=6)

    ax.imshow(rgb_img, origin="upper", zorder=1)

    if overlay_mode == "mask" and mask_img is not None:
        alpha_ch  = (mask_img.sum(axis=-1) > 0).astype(np.uint8) * 180
        mask_rgba = np.dstack([mask_img, alpha_ch])
        ax.imshow(mask_rgba, origin="upper", alpha=0.45, zorder=2)

    # removed edges (crossed non-tissue) in red
    for u, v in removed_edges:
        ax.plot([cx[u], cx[v]], [cy[u], cy[v]],
                color="#ff4444", alpha=0.7, linewidth=1.0, zorder=3)

    # kept edges
    for u, v in edges:
        ax.plot([cx[u], cx[v]], [cy[u], cy[v]],
                color="black", alpha=0.85, linewidth=1.2, zorder=4)

    # nodes
    ax.scatter(cx, cy, color="#00d4ff", s=14, alpha=0.9,
               linewidths=0.4, edgecolors="white", zorder=5)

    ax.set_xlim(-0.5, rgb_w - 0.5)
    ax.set_ylim(rgb_h - 0.5, -0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delaunay graph visualization (dot nodes, no patch images).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path",   default="/mnt/iam")
    p.add_argument("--hospital",   default=None)
    p.add_argument("--patient_id", default=None)
    p.add_argument("--slide_id",   default=None)
    p.add_argument("--list",       action="store_true")
    p.add_argument("--output",     default="outputs/delaunay_overlay.png")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    iam_path = Path(args.iam_path)

    cls_dir = iam_path / CLS_DIR_SUBPATH
    df_npz  = load_all_npz(cls_dir)

    hospitals = sorted(df_npz["Hospital"].unique())
    hospital  = args.hospital or hospitals[0]
    if hospital not in hospitals:
        sys.exit(
            f"[ERROR] Hospital '{hospital}' not found.\nAvailable: "
            + ", ".join(hospitals)
        )
    print(f"[INFO] Hospital : {hospital}")
    df_hosp = df_npz[df_npz["Hospital"] == hospital]

    if args.list:
        summary = (
            df_hosp.groupby(["Patient_ID", "Slide"])
            .size().reset_index(name="bags")
            .sort_values(["Patient_ID", "Slide"])
        )
        print(f"\nPatients/slides for '{hospital}':\n")
        print(f"  {'Patient_ID':<20} {'Slide':<30} bags")
        print(f"  {'-'*20} {'-'*30} ----")
        for _, row in summary.iterrows():
            print(f"  {str(row['Patient_ID']):<20} {str(row['Slide']):<30} {row['bags']}")
        sys.exit(0)

    if args.patient_id and args.slide_id:
        patient_id, slide_id = str(args.patient_id), str(args.slide_id)
    elif args.patient_id:
        patient_id = str(args.patient_id)
        df_pat = df_hosp[df_hosp["Patient_ID"] == patient_id]
        if df_pat.empty:
            sys.exit(f"[ERROR] Patient '{patient_id}' not found.")
        slide_id = str(df_pat.groupby("Slide").size().idxmax())
    elif args.slide_id:
        slide_id = str(args.slide_id)
        df_sl = df_hosp[df_hosp["Slide"] == slide_id]
        if df_sl.empty:
            sys.exit(f"[ERROR] Slide '{slide_id}' not found.")
        patient_id = str(df_sl.groupby("Patient_ID").size().idxmax())
    else:
        counts = df_hosp.groupby(["Patient_ID", "Slide"]).size()
        patient_id, slide_id = counts.idxmax()

    df_slide = df_hosp[
        (df_hosp["Patient_ID"] == patient_id) &
        (df_hosp["Slide"]      == slide_id)
    ]
    if df_slide.empty:
        sys.exit(f"[ERROR] No bags for patient={patient_id}, slide={slide_id}")

    print(f"[INFO] Patient  : {patient_id}")
    print(f"[INFO] Slide    : {slide_id}")
    print(f"[INFO] Bags     : {len(df_slide)}")

    # bag centroids as node positions (j, i) in WSI level-0 pixels
    coord_arrays = np.stack(df_slide["coords_bag"].tolist(), axis=0)  # (N, 256, 2)
    centroids    = coord_arrays.mean(axis=1)                           # (N, 2)

    slide_meta = load_slide_meta(iam_path, hospital, patient_id, slide_id)
    rgb_img    = load_rgb_image(iam_path, hospital, patient_id, slide_id)
    mask_img   = load_mask_image(iam_path, hospital, patient_id, slide_id)

    if rgb_img is None or slide_meta is None:
        sys.exit("[ERROR] RGB image and slide metadata are required for rendering.")

    edges, _ = build_delaunay_edges(centroids, distance_factor=2.0)
    print(f"[INFO] Graph    : {len(centroids)} nodes | {len(edges)} edges (before mask filter)")

    removed_edges = np.empty((0, 2), dtype=np.int64)
    if mask_img is not None:
        edges, removed_edges = filter_edges_by_mask(
            edges, centroids, mask_img,
            j_base=slide_meta["j_base"], i_base=slide_meta["i_base"],
            slide_w=slide_meta["w"],     slide_h=slide_meta["h"],
            patch_size=4096,
        )
        print(f"[INFO] Edges after mask filter: {len(edges)} kept, {len(removed_edges)} removed")
    else:
        print("[WARN] Mask not available — skipping edge mask filter.")

    export_graph(centroids, edges, out_path=Path(args.output))

    title = (
        f"{hospital}  ·  {patient_id}  ·  {slide_id}  |  "
        f"{len(centroids)} nodes  ·  {len(edges)} edges kept  ·  "
        f"{len(removed_edges)} removed  ·  overlay: {OVERLAY_MODE}"
    )
    render(
        coords=centroids,
        edges=edges,
        removed_edges=removed_edges,
        title=title,
        out_path=Path(args.output),
        rgb_img=rgb_img,
        slide_meta=slide_meta,
        mask_img=mask_img,
        overlay_mode=OVERLAY_MODE,
    )


if __name__ == "__main__":
    main()
