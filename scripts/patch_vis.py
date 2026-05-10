#!/usr/bin/env python3
"""
Patch visualizer over a single WSI slide.

Each NPZ row is one 2048×2048 patch.
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
      --hospital "Consorci Sanitari de Terrassa" --list

  python scripts/patch_vis.py \\
      --hospital "H. Bellvitge" --patient_id "1234-1" --slide_id "12345 A1" \\
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

# Import wsi_io from the canonical location (pt1diagnosis submodule)
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DAVID = _ROOT / "pt1diagnosis" / "scripts_david"
if _SCRIPTS_DAVID.exists():
    sys.path.insert(0, str(_SCRIPTS_DAVID))
else:
    # Fallback: same directory as this script (old layout)
    sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from PIL import Image as _PILImage

from wsi_io import (  # noqa: E402
    CLS_DIR_SUBPATH,
    find_patches_dir,
    load_all_npz,
)

_PILImage.MAX_IMAGE_PIXELS = 400_000_000

PATCH_PX = 2048

# ── PEARSON2-aware path constants ─────────────────────────────────────────────
_BASE = "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD"
# Slide metadata: PEARSON2 (new columns: patient/slide); fallback PEARSON (patient_ID/slide_ID)
_META_CANDIDATES = [
    (_BASE + "/PEARSON2/Patient_Images_metadata.csv",       "patient",    "slide"),
    (_BASE + "/PEARSON/Images/Patient_Images_metadata.csv", "patient_ID", "slide_ID"),
]
# RGB images remain in old PEARSON (not yet in PEARSON2)
_RGB_CANDIDATES = [
    _BASE + "/PEARSON/Images/RGB_Images",
    _BASE + "/PEARSON2/RGB_Images",
]
# Masks: PEARSON2 first, then old PEARSON
_MASK_CANDIDATES = [
    _BASE + "/PEARSON2/Segmentation_Masks",
    _BASE + "/PEARSON/Images/Segmentation_Masks",
]


# ── local I/O helpers (override wsi_io for PEARSON2 compatibility) ─────────────

def load_slide_meta(iam_path: Path, hospital: str, patient: str, slide: str):
    """Try PEARSON2 metadata CSV first (patient/slide cols), then old PEARSON."""
    for rel_path, pat_col, slide_col in _META_CANDIDATES:
        meta_path = iam_path / rel_path
        if not meta_path.exists():
            continue
        try:
            df = pd.read_csv(meta_path, encoding="utf-8")
        except Exception:
            continue
        if pat_col not in df.columns or slide_col not in df.columns:
            continue
        row = df[
            (df["hospital"].astype(str) == hospital) &
            (df[pat_col].astype(str)    == patient)  &
            (df[slide_col].astype(str)  == slide)
        ]
        if not row.empty:
            r = row.iloc[-1]
            return {"j_base": float(r["j"]), "i_base": float(r["i"]),
                    "w": float(r["w"]), "h": float(r["h"])}
    print(f"[WARN] {hospital}/{patient}/{slide} not found in any slide metadata CSV.")
    return None


def load_rgb_image(iam_path: Path, hospital: str, patient: str, slide: str):
    """Load RGB slide image; searches PEARSON and PEARSON2 locations."""
    for rel in _RGB_CANDIDATES:
        rgb_dir = iam_path / rel
        if not rgb_dir.is_dir():
            continue
        path = rgb_dir / hospital / patient / f"{hospital}_{slide}.png"
        if path.exists():
            print(f"[INFO] RGB image : {path}")
            return np.array(_PILImage.open(path).convert("RGB"))
    print(f"[WARN] RGB image not found for {hospital}/{patient}/{slide}")
    return None


def load_mask_image(iam_path: Path, hospital: str, patient: str, slide: str):
    """Load segmentation mask; searches PEARSON2 then PEARSON."""
    for rel in _MASK_CANDIDATES:
        masks_dir = iam_path / rel
        if not masks_dir.is_dir():
            continue
        path = masks_dir / hospital / patient / f"{hospital}_{slide}_mask.png"
        if path.exists():
            print(f"[INFO] Mask      : {path}")
            return np.array(_PILImage.open(path).convert("RGB"))
    print(f"[WARN] Mask not found for {hospital}/{patient}/{slide}")
    return None


# ── helpers ────────────────────────────────────────────────────────────────────

def _wsi_to_display(j, i, j_base, i_base, scale):
    return (j - j_base) * scale, (i - i_base) * scale


def _load_patch(slide_patches_dir, basename):
    p = slide_patches_dir / basename
    if not p.exists():
        return None
    try:
        return np.array(_PILImage.open(p).convert("RGB"))
    except Exception:
        return None


# ── render modes ───────────────────────────────────────────────────────────────

def render_simple_square(rgb_img, mask_img, coord_arrays, slide_meta,
                         title, out_path, show_mask=False, rect_kw=None):
    """
    Draws a 2048×2048 rectangle per patch.
    If rgb_img is None (RGB not available for this slide), uses a dark blank canvas
    scaled from the WSI metadata bounds. If slide_meta is also None, falls back to
    plotting patches on a coordinate canvas derived from the patch positions.
    """
    if rect_kw is None:
        rect_kw = dict(linewidth=1, edgecolor="#00e5ff", facecolor="none", alpha=0.85)

    if rgb_img is not None and slide_meta is not None:
        # Normal mode: RGB background
        rgb_h, rgb_w = rgb_img.shape[:2]
        scale  = rgb_w / slide_meta["w"]
        j_base = slide_meta["j_base"]
        i_base = slide_meta["i_base"]
        canvas = rgb_img
    elif slide_meta is not None:
        # No RGB: blank canvas sized from WSI metadata
        print("[INFO] No RGB image available — rendering patch positions on blank canvas.")
        canvas_w = min(2000, max(800, int(slide_meta["w"] / 20)))
        canvas_h = min(1500, max(600, int(slide_meta["h"] / 20)))
        scale  = canvas_w / slide_meta["w"]
        j_base = slide_meta["j_base"]
        i_base = slide_meta["i_base"]
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
    else:
        # No RGB and no metadata: build canvas from patch coordinates
        print("[INFO] No RGB or metadata — rendering patch positions from coordinates.")
        js = coord_arrays[:, 0].astype(float)
        is_ = coord_arrays[:, 1].astype(float)
        j_base = js.min()
        i_base = is_.min()
        wsi_w  = js.max() - js.min() + PATCH_PX
        wsi_h  = is_.max() - is_.min() + PATCH_PX
        target_w = 1800
        scale  = target_w / max(wsi_w, 1)
        canvas_w = int(wsi_w * scale)
        canvas_h = int(wsi_h * scale)
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    fig.suptitle(title, color="white", fontsize=9, y=0.998)
    ax.imshow(canvas, origin="upper")

    if show_mask and mask_img is not None:
        alpha_ch = (mask_img.sum(axis=-1) > 0).astype(np.uint8) * 180
        mask_rgba = np.dstack([mask_img, alpha_ch])
        ax.imshow(mask_rgba, origin="upper", alpha=0.45)

    rect_px = PATCH_PX * scale
    for coord in coord_arrays:
        j_c, i_c = float(coord[0]), float(coord[1])
        cx, cy   = _wsi_to_display(j_c, i_c, j_base, i_base, scale)
        ax.add_patch(mpatches.Rectangle((cx, cy), rect_px, rect_px, **rect_kw))

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


def render_image(rgb_img, mask_img, coord_arrays, paths_list, slide_patches_dir,
                 slide_meta, title, out_path, show_mask=False, max_canvas_side=6000):
    from PIL import Image as _PILImage

    patch_imgs, patch_coords = [], []
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

    coords = np.array(patch_coords)
    print(f"[INFO] Loaded {len(patch_imgs)}/{n_attempted} patches")

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
        x2, y2 = min(x + patch_px_out, canvas_w), min(y + patch_px_out, canvas_h)
        canvas[y:y2, x:x2] = img_resized[:y2 - y, :x2 - x]

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
        extent = [ex_x0, ex_x1, ex_y1, ex_y0]
    else:
        extent = [0, canvas_w, canvas_h, 0]

    axes[0].imshow(canvas, origin="upper")
    axes[0].set_xlim(-0.5, canvas_w - 0.5)
    axes[0].set_ylim(canvas_h - 0.5, -0.5)
    axes[0].set_title("Assembled patches (2048 px)", color="white", fontsize=11, pad=6)

    if rgb_img is not None:
        axes[1].imshow(canvas, origin="upper", zorder=1)
        axes[1].imshow(rgb_img, extent=extent, origin="upper", aspect="auto", alpha=0.45, zorder=2)
        if show_mask and mask_img is not None:
            axes[1].imshow(mask_img, extent=extent, origin="upper", aspect="auto", alpha=0.4, zorder=3)
        axes[1].set_xlim(-0.5, canvas_w - 0.5)
        axes[1].set_ylim(canvas_h - 0.5, -0.5)
        axes[1].set_title("Patches + RGB overlay", color="white", fontsize=11, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Patch visualizer over a WSI slide (2048×2048 px patches).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path",    default="/mnt/iam")
    p.add_argument("--hospital",    default=None)
    p.add_argument("--patient_id",  default=None)
    p.add_argument("--slide_id",    default=None)
    p.add_argument("--list",        action="store_true",
                   help="List available patients/slides and exit")
    p.add_argument("--mode",        default="simple_square",
                   choices=["simple_square", "image"])
    p.add_argument("--show_mask",   action="store_true")
    p.add_argument("--max_patches", type=int, default=500)
    p.add_argument("--output",      default="outputs/patch_vis.png")
    return p.parse_args()


def main():
    args     = parse_args()
    iam_path = Path(args.iam_path)

    patches_dir = find_patches_dir(iam_path)
    cls_dir     = iam_path / CLS_DIR_SUBPATH
    df_npz      = load_all_npz(cls_dir)

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
    patient_id = str(args.patient_id).strip() if args.patient_id else None
    slide_id   = str(args.slide_id).strip()   if args.slide_id   else None

    if patient_id and slide_id:
        df_pat = df_hosp[df_hosp["Patient_ID"] == patient_id]
        if df_pat.empty:
            df_any = df_npz[df_npz["Patient_ID"] == patient_id]
            if not df_any.empty:
                other = df_any["Hospital"].unique().tolist()
                sys.exit(
                    f"[ERROR] Patient '{patient_id}' not in hospital '{hospital}'.\n"
                    f"        Found in: {other}\n"
                    f"        Try: --hospital \"{other[0]}\""
                )
            available = sorted(df_hosp["Patient_ID"].unique())
            sys.exit(
                f"[ERROR] Patient '{patient_id}' not found in NPZ.\n"
                f"        Available ({len(available)}): {available}\n"
                f"        Tip: --list shows all patients/slides."
            )
        df_slide = df_pat[df_pat["Slide"] == slide_id]
        if df_slide.empty:
            available_slides = sorted(df_pat["Slide"].unique())
            sys.exit(
                f"[ERROR] Slide '{slide_id}' not found for patient '{patient_id}'.\n"
                f"        Available slides: {available_slides}"
            )
    elif patient_id:
        df_pat = df_hosp[df_hosp["Patient_ID"] == patient_id]
        if df_pat.empty:
            df_any = df_npz[df_npz["Patient_ID"] == patient_id]
            if not df_any.empty:
                other = df_any["Hospital"].unique().tolist()
                sys.exit(
                    f"[ERROR] Patient '{patient_id}' not in hospital '{hospital}'.\n"
                    f"        Found in: {other}"
                )
            sys.exit(
                f"[ERROR] Patient '{patient_id}' not found.\n"
                f"        Available: {sorted(df_hosp['Patient_ID'].unique())}"
            )
        slide_id = str(df_pat.groupby("Slide").size().idxmax())
        df_slide = df_pat[df_pat["Slide"] == slide_id]
    elif slide_id:
        df_sl = df_hosp[df_hosp["Slide"] == slide_id]
        if df_sl.empty:
            sys.exit(
                f"[ERROR] Slide '{slide_id}' not found (hospital='{hospital}').\n"
                f"        Tip: --list shows available slides."
            )
        patient_id = str(df_sl.groupby("Patient_ID").size().idxmax())
        df_slide   = df_sl[df_sl["Patient_ID"] == patient_id]
    else:
        counts     = df_hosp.groupby(["Patient_ID", "Slide"]).size()
        patient_id, slide_id = counts.idxmax()
        df_slide   = df_hosp[
            (df_hosp["Patient_ID"] == patient_id) & (df_hosp["Slide"] == slide_id)
        ]

    if len(df_slide) > args.max_patches:
        df_slide = df_slide.sample(n=args.max_patches, random_state=42).reset_index(drop=True)

    print(f"[INFO] Patient  : {patient_id}")
    print(f"[INFO] Slide    : {slide_id}")
    print(f"[INFO] Patches  : {len(df_slide)}")

    coord_arrays = np.stack(df_slide["coords_bag"].tolist(), axis=0)
    paths_list   = df_slide["paths_bag"].tolist()

    slide_meta = load_slide_meta(iam_path, hospital, patient_id, slide_id)
    rgb_img    = load_rgb_image(iam_path, hospital, patient_id, slide_id)
    mask_img   = load_mask_image(iam_path, hospital, patient_id, slide_id) if args.show_mask else None

    title    = (f"{hospital}  |  {patient_id}  |  {slide_id}  |  "
                f"{len(df_slide)} patches (2048 px)  |  mode: {args.mode}")
    out_path = Path(args.output)

    if args.mode == "simple_square":
        if rgb_img is None:
            print("[INFO] RGB image not available for this slide — using blank canvas.")
        if slide_meta is None:
            print("[INFO] Slide metadata not found — patch positions derived from coordinates.")
        render_simple_square(
            rgb_img=rgb_img, mask_img=mask_img, coord_arrays=coord_arrays,
            slide_meta=slide_meta, title=title, out_path=out_path, show_mask=args.show_mask,
        )
    elif args.mode == "image":
        slide_patches_dir = patches_dir / hospital / patient_id / slide_id / "patches"
        if not slide_patches_dir.exists():
            print(f"[WARN] patches/ subdir not found: {slide_patches_dir}")
            slide_patches_dir = patches_dir / hospital / patient_id / slide_id
        render_image(
            rgb_img=rgb_img, mask_img=mask_img, coord_arrays=coord_arrays,
            paths_list=paths_list, slide_patches_dir=slide_patches_dir,
            slide_meta=slide_meta, title=title, out_path=out_path, show_mask=args.show_mask,
        )


if __name__ == "__main__":
    main()
