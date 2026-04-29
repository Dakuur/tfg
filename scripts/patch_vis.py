#!/usr/bin/env python3
"""
Patch visualizer over a single WSI slide.

Each NPZ row is now one 2048×2048 patch (not a bag of 256 sub-patches).
Patches are stored as JPEGs inside:
  Patches2048/{hospital}/{patient}/{slide}/patches/{basename}.jpg

Two display modes:
  simple_square  – draws a 2048×2048 rectangle per patch on the full RGB slide
  image          – loads each JPEG and composites them onto a canvas at their
                   WSI coordinates

─── slide selection ────────────────────────────────────────────────────────────
  (no flags)                     →  slide with the most patches for the first hospital
  --hospital H                   →  best slide of hospital H
  --patient_id P                 →  best slide of patient P
  --slide_id S                   →  specific slide (patient resolved automatically)
  --patient_id P --slide_id S    →  specific patient + slide
  --list                         →  print available patients/slides and exit

─── usage examples ─────────────────────────────────────────────────────────────
  python scripts/patch_vis.py --mode simple_square

  python scripts/patch_vis.py \\
      --hospital "H. Bellvitge" --patient_id "12345" --slide_id "12345_A1" \\
      --mode image --output outputs/12345_A1_patches.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from wsi_io import (  # noqa: E402
    CLS_DIR_SUBPATH,
    find_patches_dir,
    load_all_npz,
    load_slide_meta,
    load_mask_image,
    load_rgb_image,
)

PATCH_PX = 2048  # WSI level-0 side length of each 2048×2048 patch


# ── helpers ────────────────────────────────────────────────────────────────────

def _wsi_to_display(
    j: float, i: float,
    j_base: float, i_base: float,
    scale: float,
) -> tuple[float, float]:
    """Convert WSI level-0 pixel coords to display pixel coords on the RGB image."""
    return (j - j_base) * scale, (i - i_base) * scale


def _load_patch(slide_patches_dir: Path, basename: str) -> "np.ndarray | None":
    """Load one patch JPEG from disk. Returns None on failure."""
    from PIL import Image as _PILImage
    p = slide_patches_dir / basename
    if not p.exists():
        return None
    try:
        return np.array(_PILImage.open(p).convert("RGB"))
    except Exception:
        return None


# ── render modes ───────────────────────────────────────────────────────────────

def render_simple_square(
    rgb_img: np.ndarray,
    mask_img: "np.ndarray | None",
    coord_arrays: np.ndarray,   # (N, 2) — (j, i) WSI level-0 coord per patch
    slide_meta: dict,
    title: str,
    out_path: Path,
    show_mask: bool = False,
    rect_kw: "dict | None" = None,
) -> None:
    """
    Overlays one 2048×2048 rectangle per patch on the full RGB slide image.
    """
    rgb_h, rgb_w = rgb_img.shape[:2]
    scale = rgb_w / slide_meta["w"]

    if rect_kw is None:
        rect_kw = dict(linewidth=1, edgecolor="#00e5ff", facecolor="none", alpha=0.75)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    fig.suptitle(title, color="white", fontsize=9, y=0.998)

    ax.imshow(rgb_img, origin="upper")

    if show_mask and mask_img is not None:
        alpha_ch = (mask_img.sum(axis=-1) > 0).astype(np.uint8) * 180
        mask_rgba = np.dstack([mask_img, alpha_ch])
        ax.imshow(mask_rgba, origin="upper", alpha=0.45)

    rect_px = PATCH_PX * scale
    for coord in coord_arrays:
        j_c, i_c = float(coord[0]), float(coord[1])
        cx, cy   = _wsi_to_display(j_c, i_c, slide_meta["j_base"], slide_meta["i_base"], scale)
        rect = mpatches.Rectangle((cx, cy), rect_px, rect_px, **rect_kw)
        ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


def render_image(
    rgb_img: "np.ndarray | None",
    mask_img: "np.ndarray | None",
    coord_arrays: np.ndarray,    # (N, 2) — (j, i) WSI level-0 coord per patch
    paths_list: list[str],       # N Windows path strings
    slide_patches_dir: Path,     # …/Patches2048/{h}/{p}/{s}/patches/
    slide_meta: "dict | None",
    title: str,
    out_path: Path,
    show_mask: bool = False,
    max_canvas_side: int = 6000,
) -> None:
    """
    Loads each 2048×2048 JPEG patch and composites them onto a canvas at
    their WSI coordinates.
    """
    from PIL import Image as _PILImage

    patch_imgs:   list[np.ndarray] = []
    patch_coords: list[np.ndarray] = []
    n_attempted = 0

    for coord, path_str in zip(coord_arrays, paths_list):
        n_attempted += 1
        basename = str(path_str).replace("\\", "/").split("/")[-1]
        img = _load_patch(slide_patches_dir, basename)
        if img is None:
            continue
        patch_imgs.append(img)
        patch_coords.append(np.array([float(coord[0]), float(coord[1])]))

    if not patch_imgs:
        print(f"[WARN] No patch images loaded from {n_attempted} patches in {slide_patches_dir}")
        return

    coords = np.array(patch_coords)   # (M, 2)
    print(f"[INFO] Loaded {len(patch_imgs)}/{n_attempted} patches")

    # ── determine canvas extent in WSI level-0 pixels ─────────────────────────
    j_min_all = coords[:, 0].min()
    i_min_all = coords[:, 1].min()
    j_max_all = coords[:, 0].max() + PATCH_PX
    i_max_all = coords[:, 1].max() + PATCH_PX

    wsi_w = j_max_all - j_min_all
    wsi_h = i_max_all - i_min_all

    canvas_scale = min(1.0, max_canvas_side / max(wsi_w, wsi_h))
    canvas_w     = max(1, int(wsi_w * canvas_scale))
    canvas_h     = max(1, int(wsi_h * canvas_scale))
    patch_px_out = max(1, int(PATCH_PX * canvas_scale))

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for img, (j, i) in zip(patch_imgs, coords):
        x = int((j - j_min_all) * canvas_scale)
        y = int((i - i_min_all) * canvas_scale)
        img_resized = np.array(
            _PILImage.fromarray(img).resize((patch_px_out, patch_px_out), _PILImage.LANCZOS)
        )
        x2 = min(x + patch_px_out, canvas_w)
        y2 = min(y + patch_px_out, canvas_h)
        canvas[y:y2, x:x2] = img_resized[: y2 - y, : x2 - x]

    # ── figure ────────────────────────────────────────────────────────────────
    n_panels = 1 + (rgb_img is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(14 * n_panels, 10))
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=9, y=0.998)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

    if slide_meta is not None:
        ex_x0 = (slide_meta["j_base"] - j_min_all) * canvas_scale
        ex_x1 = (slide_meta["j_base"] + slide_meta["w"] - j_min_all) * canvas_scale
        ex_y0 = (slide_meta["i_base"] - i_min_all) * canvas_scale
        ex_y1 = (slide_meta["i_base"] + slide_meta["h"] - i_min_all) * canvas_scale
        extent = [ex_x0, ex_x1, ex_y1, ex_y0]   # matplotlib origin="upper"
    else:
        extent = [0, canvas_w, canvas_h, 0]

    axes[0].imshow(canvas, origin="upper")
    axes[0].set_xlim(-0.5, canvas_w - 0.5)
    axes[0].set_ylim(canvas_h - 0.5, -0.5)
    axes[0].set_title("Assembled patches (2048 px)", color="white", fontsize=11, pad=6)

    if rgb_img is not None:
        axes[1].imshow(canvas, origin="upper", zorder=1)
        axes[1].imshow(rgb_img, extent=extent, origin="upper", aspect="auto",
                       alpha=0.45, zorder=2)
        if show_mask and mask_img is not None:
            axes[1].imshow(mask_img, extent=extent, origin="upper", aspect="auto",
                           alpha=0.4, zorder=3)
        axes[1].set_xlim(-0.5, canvas_w - 0.5)
        axes[1].set_ylim(canvas_h - 0.5, -0.5)
        axes[1].set_title("Patches + RGB overlay", color="white", fontsize=11, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch visualizer over a WSI slide (2048×2048 px patches).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path",    default="/mnt/iam",
                   help="Dataset root (/mnt/iam)")
    p.add_argument("--hospital",    default=None)
    p.add_argument("--patient_id",  default=None)
    p.add_argument("--slide_id",    default=None)
    p.add_argument("--list",        action="store_true",
                   help="List available patients/slides and exit")
    p.add_argument("--mode",        default="simple_square",
                   choices=["simple_square", "image"],
                   help="simple_square: rectangles on RGB | image: composite patch JPEGs")
    p.add_argument("--show_mask",   action="store_true",
                   help="Overlay segmentation mask")
    p.add_argument("--max_patches", type=int, default=500,
                   help="Max patches to load (random sample if exceeded)")
    p.add_argument("--output",      default="outputs/patch_vis.png")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    iam_path = Path(args.iam_path)

    patches_dir = find_patches_dir(iam_path)

    cls_dir = iam_path / CLS_DIR_SUBPATH
    df_npz  = load_all_npz(cls_dir)

    hospitals = sorted(df_npz["Hospital"].unique())
    hospital  = args.hospital or hospitals[0]
    if hospital not in hospitals:
        sys.exit(
            f"[ERROR] Hospital '{hospital}' not found in NPZ.\n"
            "Available:\n  " + "\n  ".join(hospitals)
        )
    print(f"[INFO] Hospital : {hospital}")

    df_hosp = df_npz[df_npz["Hospital"] == hospital]

    if args.list:
        summary = (
            df_hosp.groupby(["Patient_ID", "Slide"])
            .size()
            .reset_index(name="patches")
            .sort_values(["Patient_ID", "Slide"])
        )
        print(f"\nAvailable patients/slides for hospital '{hospital}':\n")
        print(f"  {'Patient_ID':<20} {'Slide':<30} patches")
        print(f"  {'-'*20} {'-'*30} -------")
        for _, row in summary.iterrows():
            print(f"  {str(row['Patient_ID']):<20} {str(row['Slide']):<30} {row['patches']}")
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
        sys.exit(f"[ERROR] No patches in NPZ for patient={patient_id}, slide={slide_id}")

    if len(df_slide) > args.max_patches:
        df_slide = df_slide.sample(n=args.max_patches, random_state=42).reset_index(drop=True)

    print(f"[INFO] Patient  : {patient_id}")
    print(f"[INFO] Slide    : {slide_id}")
    print(f"[INFO] Patches  : {len(df_slide)}")

    # coords_bag is now (2,) per patch — stack gives (N, 2)
    coord_arrays = np.stack(df_slide["coords_bag"].tolist(), axis=0)   # (N, 2)
    paths_list   = df_slide["paths_bag"].tolist()                       # list of N strings

    slide_meta = load_slide_meta(iam_path, hospital, patient_id, slide_id)
    rgb_img    = load_rgb_image(iam_path, hospital, patient_id, slide_id)
    mask_img   = load_mask_image(iam_path, hospital, patient_id, slide_id) if args.show_mask else None

    title    = (f"{hospital}  |  {patient_id}  |  {slide_id}  |  "
                f"{len(df_slide)} patches (2048 px)  |  mode: {args.mode}")
    out_path = Path(args.output)

    if args.mode == "simple_square":
        if rgb_img is None or slide_meta is None:
            sys.exit("[ERROR] simple_square mode requires the RGB image and slide metadata.")
        render_simple_square(
            rgb_img=rgb_img,
            mask_img=mask_img,
            coord_arrays=coord_arrays,
            slide_meta=slide_meta,
            title=title,
            out_path=out_path,
            show_mask=args.show_mask,
        )

    elif args.mode == "image":
        # Patches live in …/Patches2048/{hospital}/{patient}/{slide}/patches/
        slide_patches_dir = patches_dir / hospital / patient_id / slide_id / "patches"
        if not slide_patches_dir.exists():
            print(f"[WARN] patches/ subdir not found: {slide_patches_dir}")
            print("       Falling back to slide root dir.")
            slide_patches_dir = patches_dir / hospital / patient_id / slide_id
        render_image(
            rgb_img=rgb_img,
            mask_img=mask_img,
            coord_arrays=coord_arrays,
            paths_list=paths_list,
            slide_patches_dir=slide_patches_dir,
            slide_meta=slide_meta,
            title=title,
            out_path=out_path,
            show_mask=args.show_mask,
        )


if __name__ == "__main__":
    main()
