#!/usr/bin/env python3
"""
WSI data I/O utilities: path resolution, metadata and image loading.

This module centralises all dataset paths and loading logic so that
delaunay_vis.py and build_dataset.py share a single source of truth.

Functions
---------
find_patches_dir    – locate the Patches directory
find_rgb_images_dir – locate the RGB_Images directory
find_masks_dir      – locate the Segmentation_Masks directory
load_metadata       – read the per-patch CSV for a hospital
load_slide_meta     – read per-slide WSI bounds from the global metadata CSV
load_mask_image     – load a segmentation mask as a (H, W, 3) uint8 array
load_rgb_image      – load a full slide RGB image as a (H, W, 3) uint8 array
load_patches        – load patch images with their WSI-level coordinates
load_all_npz        – load all *_CLS.npz files into a single DataFrame
load_labels         – load patient metastasis labels from the Excel file

Constants
---------
PATCHES_SUBPATH, RGB_IMAGES_SUBPATH, MASKS_SUBPATH,
SLIDE_META_SUBPATH, CLS_DIR_SUBPATH, LABELS_SUBPATH
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = 400_000_000

# ── dataset path constants ─────────────────────────────────────────────────────
PATCHES_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches"
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
CLS_DIR_SUBPATH = (
    "Experiments/MedImaging/ColonCancer/cls_info_plus_pipeline/cls_ALL"
)
LABELS_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON"
    "/xlsx_files/24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"
)


# ── path resolution ────────────────────────────────────────────────────────────

def find_patches_dir(iam_path: Path) -> Path:
    """Return the Patches directory (individual patch JPEGs), searching common locations under iam_path."""
    for candidate in [iam_path / "Patches", iam_path / PATCHES_SUBPATH]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find a 'Patches' directory under '{iam_path}'.\n"
        "Pass --iam_path pointing to the dataset root (/mnt/iam) or the Images/ folder."
    )


def find_rgb_images_dir(iam_path: Path) -> Optional[Path]:
    """Return the RGB_Images directory, or None if not found."""
    for candidate in [iam_path / "RGB_Images", iam_path / RGB_IMAGES_SUBPATH]:
        if candidate.is_dir():
            return candidate
    return None


def find_masks_dir(iam_path: Path) -> Optional[Path]:
    """Return the Segmentation_Masks directory, or None if not found."""
    for candidate in [iam_path / "Segmentation_Masks", iam_path / MASKS_SUBPATH]:
        if candidate.is_dir():
            return candidate
    return None


# ── metadata loading ───────────────────────────────────────────────────────────

def load_metadata(patches_dir: Path, hospital: str) -> pd.DataFrame:
    """
    Read the per-patch metadata CSV for a hospital.

    Returns a DataFrame with numeric columns i, j, blurriness,
    non_white_area, affected_percentage.
    """
    meta_path = patches_dir / hospital / f"metadata_{hospital}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    df = pd.read_csv(meta_path)
    for col in ["i", "j", "blurriness", "non_white_area", "affected_percentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_slide_meta(
    iam_path: Path, hospital: str, patient: str, slide: str
) -> Optional[dict]:
    """
    Read per-slide WSI bounds from Patient_Images_metadata.csv.

    Returns a dict with keys j_base, i_base, w, h (all WSI level-0 pixels),
    or None if the file or the matching row is not found.

      j_base / i_base : x/y origin of the slide crop (mrxs_cmin / mrxs_rmin)
      w / h           : width / height of the slide crop
    """
    meta_path = iam_path / SLIDE_META_SUBPATH
    if not meta_path.exists():
        print(f"[WARN] Slide metadata CSV not found: {meta_path}")
        print("       Background alignment will be approximate; edge mask-filter disabled.")
        return None
    df = pd.read_csv(meta_path, encoding="utf-8")
    row = df[
        (df["hospital"].astype(str)   == hospital) &
        (df["patient_ID"].astype(str) == patient)  &
        (df["slide_ID"].astype(str)   == slide)
    ]
    if row.empty:
        print(f"[WARN] {hospital}/{patient}/{slide} not found in slide metadata CSV.")
        return None
    r = row.iloc[-1]
    return {
        "j_base": float(r["j"]),
        "i_base": float(r["i"]),
        "w":      float(r["w"]),
        "h":      float(r["h"]),
    }


# ── image loading ──────────────────────────────────────────────────────────────

def load_mask_image(
    iam_path: Path, hospital: str, patient: str, slide: str
) -> Optional[np.ndarray]:
    """Load the segmentation mask as a (H, W, 3) uint8 RGB array."""
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
) -> Optional[np.ndarray]:
    """Load the full slide RGB image as a (H, W, 3) uint8 array."""
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

def _patch_filename(hospital: str, patient: str, slide: str, i: int, j: int) -> str:
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

    Parameters
    ----------
    patches_dir  : root directory returned by find_patches_dir
    df_slide     : metadata rows for the target (patient, slide)
    hospital     : hospital name (used to build file paths)
    patient      : patient ID
    slide        : slide ID
    max_patches  : cap on number of patches loaded (random sample if exceeded)

    Returns
    -------
    images    : list of (2048, 2048, 3) uint8 arrays
    coords    : (N, 2) float64 array — columns are (j, i) = (x, y) in WSI level-0 pixels
    non_white : (N,)  float64 array — non-white area fraction per patch
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
        fname     = _patch_filename(
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

    coords    = np.array(coords_list,    dtype=np.float64)
    non_white = np.array(non_white_list, dtype=np.float64)
    return images, coords, non_white


# ── NPZ / label loading ────────────────────────────────────────────────────────

def load_all_npz(cls_dir: Path) -> pd.DataFrame:
    """
    Load all *_CLS.npz files and return a single DataFrame with columns:
        Patient_ID, Slide, Section, Hospital, CLS, coords_bag

    CLS       : np.ndarray (1536,)    — real UNI2 CLS token embedding for the bag
    coords_bag: np.ndarray (256, 2)   — patch-level (x, y) coordinates for the bag;
                the centroid is used as the node position in the graph
    """
    npz_files = sorted(cls_dir.glob("*_CLS.npz"))
    if not npz_files:
        sys.exit(f"[ERROR] No *_CLS.npz files found in {cls_dir}")

    frames = []
    for npz_path in npz_files:
        try:
            npz = np.load(npz_path, allow_pickle=True)
            df  = pd.DataFrame({
                "Patient_ID": npz["patient_list"].astype(str),
                "Slide":      npz["slides"].astype(str),
                "Section":    npz["sections"].astype(str),
                "Hospital":   npz["hospitals"].astype(str),
                "CLS":        list(npz["embeddingCLS"]),   # list of (1536,) arrays
                "coords_bag": list(npz["coords"]),         # list of (256, 2) arrays
                "paths_bag":  list(npz["paths"]),          # list of (256,) Windows path strings
            })
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Could not load {npz_path.name}: {exc}")

    if not frames:
        sys.exit("[ERROR] No NPZ files could be loaded.")

    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] NPZ total bags: {len(df):,}  |  unique patients: {df['Patient_ID'].nunique()}")
    return df


def load_labels(iam_path: Path) -> pd.DataFrame:
    """
    Load patient metastasis labels from the Excel file.

    Returns DataFrame with columns [Patient_ID, Metastasis_score, label].
    NX rows are dropped. Labels: N0 → 0, N1/N2 subtypes → 1.
    """
    excel_path = iam_path / LABELS_SUBPATH
    if not excel_path.exists():
        sys.exit(f"[ERROR] Labels Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)

    score_col = next(
        (c for c in df.columns if str(c).startswith("PATHOLOGIST SCORE")), None
    )
    if score_col is None:
        print("[ERROR] Column 'PATHOLOGIST SCORE...' not found. Available columns:")
        for c in df.columns:
            print(f"    {c!r}")
        sys.exit(1)

    for col in ("CODE", "Data Access Group"):
        if col not in df.columns:
            print(f"[ERROR] Expected column '{col}' not found. Available: {list(df.columns)}")
            sys.exit(1)

    df = df.rename(columns={score_col: "Metastasis_score", "CODE": "Patient_ID"})
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    # Normalise N1 subtypes
    df["Metastasis_score"] = df["Metastasis_score"].replace(
        {"N1a": "N1", "N1b": "N1", "N1c": "N1", "N2a": "N1", "N2b": "N1"}
    )

    before = len(df)
    df = df[df["Metastasis_score"] != "NX"].copy()
    print(f"[INFO] Labels: dropped {before - len(df)} NX rows → {len(df)} patients retained")
    print(f"       Score distribution: {df['Metastasis_score'].value_counts().to_dict()}")

    df["label"] = (df["Metastasis_score"] != "N0").astype(int)
    return df[["Patient_ID", "Metastasis_score", "label"]].copy()
