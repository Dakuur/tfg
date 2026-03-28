#!/usr/bin/env python3
"""
Build a PyTorch Geometric graph dataset from WSI patches for GAT training.

Data sources
------------
- CLS embeddings : /mnt/iam/Experiments/MedImaging/ColonCancer/
                   cls_info_plus_pipeline/cls_ALL/{Hospital}_CLS.npz
  Each npz row is one **bag** of 256 patches with:
    patient_list   (N,)        patient ID
    slides         (N,)        slide ID
    sections       (N,)        section/bag ID
    hospitals      (N,)        hospital name
    embeddingCLS   (N, 1536)   real UNI2 CLS token for the bag
    coords         (N, 256, 2) patch-level (x, y) coordinates for the bag

- Labels          : .../xlsx_files/24_09_2025_..._fixed_N0s.xlsx
  Joined on patient CODE → Metastasis_score (N0 / N1, NX dropped)

Graph structure (per slide)
---------------------------
  Nodes   = bags belonging to that slide
  x       = CLS embedding   [N_bags, 1536]
  pos     = centroid of the bag's 256 patch coords   [N_bags, 2]
  edge_index = Delaunay triangulation on centroids (bidirectional)
  y       = patient-level label {0=N0, 1=N1}

Phases
------
1. Build patient/slide index (npz → label join, min-bag filter)
2. 80/20 stratified split at patient level
3. Construct graphs and save as .pt files
4. Verification summary

Usage
-----
    python scripts/build_dataset.py                   # full run
    python scripts/build_dataset.py --dry_run         # index + split, no .pt files
    python scripts/build_dataset.py --iam_path /mnt/iam

Outputs
-------
    outputs/graphs/train/{patient_id}_{slide_id}.pt
    outputs/graphs/val/{patient_id}_{slide_id}.pt
    outputs/graphs/train_index.csv
    outputs/graphs/val_index.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm import tqdm

# ── import Delaunay helper from delaunay_vis.py ───────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from delaunay_vis import build_delaunay_edges  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
CLS_DIR_SUBPATH = (
    "Experiments/MedImaging/ColonCancer/cls_info_plus_pipeline/cls_ALL"
)
LABELS_SUBPATH = (
    "Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON"
    "/xlsx_files/24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"
)
MIN_BAGS_PER_SLIDE = 3    # Delaunay requires ≥ 3 non-collinear nodes
DISTANCE_FACTOR    = 2.0  # same pruning threshold as delaunay_vis.py
RANDOM_STATE       = 42


# ── NPZ loading ───────────────────────────────────────────────────────────────

def load_all_npz(cls_dir: Path) -> pd.DataFrame:
    """
    Load all *_CLS.npz files and return a single DataFrame with columns:
        Patient_ID, Slide, Section, Hospital, CLS, coords_bag

    CLS       : np.ndarray of shape (1536,)  — real UNI2 bag embedding
    coords_bag: np.ndarray of shape (256, 2) — patch-level (x, y) coords for
                the bag; centroid is used as node position in the graph
    """
    npz_files = sorted(cls_dir.glob("*_CLS.npz"))
    if not npz_files:
        sys.exit(f"[ERROR] No *_CLS.npz files found in {cls_dir}")

    frames = []
    for npz_path in npz_files:
        try:
            npz = np.load(npz_path, allow_pickle=True)
            n   = len(npz["patient_list"])
            df  = pd.DataFrame({
                "Patient_ID": npz["patient_list"].astype(str),
                "Slide":      npz["slides"].astype(str),
                "Section":    npz["sections"].astype(str),
                "Hospital":   npz["hospitals"].astype(str),
                "CLS":        list(npz["embeddingCLS"]),        # list of (1536,) arrays
                "coords_bag": list(npz["coords"]),              # list of (256, 2) arrays
            })
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Could not load {npz_path.name}: {exc}")

    if not frames:
        sys.exit("[ERROR] No NPZ files could be loaded.")

    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] NPZ total bags: {len(df):,}  |  unique patients: {df['Patient_ID'].nunique()}")
    return df


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(iam_path: Path) -> pd.DataFrame:
    """
    Load the Excel and return DataFrame [Patient_ID, Metastasis_score, label].
    NX rows are dropped. label: N0 → 0, N1/N2 → 1.
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

    # Normalise subtypes → N1
    df["Metastasis_score"] = df["Metastasis_score"].replace(
        {"N1a": "N1", "N1b": "N1", "N1c": "N1", "N2a": "N1", "N2b": "N1"}
    )

    before = len(df)
    df = df[df["Metastasis_score"] != "NX"].copy()
    print(f"[INFO] Labels: dropped {before - len(df)} NX rows → {len(df)} patients retained")
    print(f"       Score distribution: {df['Metastasis_score'].value_counts().to_dict()}")

    df["label"] = (df["Metastasis_score"] != "N0").astype(int)
    return df[["Patient_ID", "Metastasis_score", "label"]].copy()


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def build_slide_index(df_npz: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: join npz bags with labels and build per-section index.

    The graph unit is (Patient_ID, Slide, Section): each section is a
    separate tissue piece on the glass slide and must be its own graph.
    Grouping at slide level would mix spatially disjoint sections and
    produce spurious Delaunay edges between unrelated tissue regions.

    Returns a DataFrame (one row per section) with columns:
        Patient_ID, Slide, Section, Hospital, Metastasis_score, label, n_bags
    Only sections with >= MIN_BAGS_PER_SLIDE bags are kept.
    """
    print("\n── Phase 1: Building patient/slide/section index ──────────────────")

    # Merge on Patient_ID (inner → keeps only patients with non-NX labels)
    merged = df_npz.merge(df_labels, on="Patient_ID", how="inner")
    print(f"[INFO] After label join: {len(merged):,} bags, {merged['Patient_ID'].nunique()} patients")

    # Count bags per (patient, slide, section) — the atomic graph unit
    section_counts = (
        merged.groupby(
            ["Patient_ID", "Slide", "Section", "Hospital", "Metastasis_score", "label"]
        )
        .size()
        .reset_index(name="n_bags")
    )

    before = len(section_counts)
    section_counts = section_counts[section_counts["n_bags"] >= MIN_BAGS_PER_SLIDE].copy()
    print(
        f"[INFO] Dropped {before - len(section_counts)} sections with "
        f"< {MIN_BAGS_PER_SLIDE} bags"
    )

    # Summary
    n_pat = section_counts["Patient_ID"].nunique()
    per_patient = section_counts.drop_duplicates("Patient_ID")
    n0 = (per_patient["label"] == 0).sum()
    n1 = (per_patient["label"] == 1).sum()
    n_slides   = section_counts.groupby(["Patient_ID", "Slide"]).ngroups
    n_sections = len(section_counts)
    print(f"\n  Total pacientes        : {n_pat}")
    print(f"  Con metástasis  (N1)   : {n1}")
    print(f"  Sin metástasis  (N0)   : {n0}")
    print(f"  Total slides únicos    : {n_slides}")
    print(f"  Total sections válidas : {n_sections}  (= grafos a construir)")
    print(
        f"  Bags por section       : "
        f"min={section_counts['n_bags'].min()} / "
        f"media={section_counts['n_bags'].mean():.1f} / "
        f"max={section_counts['n_bags'].max()}"
    )
    print(f"  Hospitales             : {sorted(section_counts['Hospital'].unique())}")

    return section_counts.reset_index(drop=True)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def split_patients(section_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 2: 80/20 stratified split at patient level (no data leakage across sections)."""
    print("\n── Phase 2: Stratified 80/20 split ────────────────────────────────")

    patient_labels = (
        section_index[["Patient_ID", "label"]]
        .drop_duplicates("Patient_ID")
        .reset_index(drop=True)
    )

    train_ids, val_ids = train_test_split(
        patient_labels["Patient_ID"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=patient_labels["label"],
    )

    train_idx = section_index[section_index["Patient_ID"].isin(set(train_ids))].copy()
    val_idx   = section_index[section_index["Patient_ID"].isin(set(val_ids))].copy()

    def _counts(df: pd.DataFrame) -> tuple[int, int]:
        vc = df.drop_duplicates("Patient_ID")["label"].value_counts()
        return int(vc.get(0, 0)), int(vc.get(1, 0))

    tr_n0, tr_n1 = _counts(train_idx)
    va_n0, va_n1 = _counts(val_idx)

    print(
        f"  Train: {train_idx['Patient_ID'].nunique()} pacientes, "
        f"{len(train_idx)} sections/grafos, "
        f"{train_idx['n_bags'].sum()} bags totales"
    )
    print(
        f"  Val  : {val_idx['Patient_ID'].nunique()} pacientes, "
        f"{len(val_idx)} sections/grafos, "
        f"{val_idx['n_bags'].sum()} bags totales"
    )
    print(f"  Ratio N0/N1 en train : {tr_n0}/{tr_n1}")
    print(f"  Ratio N0/N1 en val   : {va_n0}/{va_n1}")

    return train_idx, val_idx


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def build_graph_for_section(
    patient_id: str,
    slide_id: str,
    section_id: str,
    hospital: str,
    label: int,
    metastasis_score: str,
    df_npz: pd.DataFrame,
) -> Data | None:
    """
    Build one torch_geometric.data.Data for a single (slide, section).

    Each section is an independent tissue piece — building one graph per
    section prevents Delaunay from connecting spatially disjoint regions.

    Nodes      = bags in this section
    x          = CLS embedding per bag  [N, 1536]  — real UNI2 features
    pos        = centroid of the bag's 256 patch coords  [N, 2]
    edge_index = Delaunay on centroids (bidirectional, long edges pruned)
    y          = patient-level label {0=N0, 1=N1}
    """
    mask = (
        (df_npz["Patient_ID"] == patient_id) &
        (df_npz["Slide"]      == slide_id)   &
        (df_npz["Section"]    == section_id) &
        (df_npz["Hospital"]   == hospital)
    )
    df_section = df_npz[mask]

    if len(df_section) < MIN_BAGS_PER_SLIDE:
        return None

    # ── node features: real CLS embeddings ────────────────────────────────────
    cls_arrays = np.stack(df_section["CLS"].tolist(), axis=0)        # (N, 1536)
    x = torch.tensor(cls_arrays, dtype=torch.float32)

    # ── node positions: centroid of each bag's 256 patch coords ───────────────
    coord_arrays = np.stack(df_section["coords_bag"].tolist(), axis=0)  # (N, 256, 2)
    centroids    = coord_arrays.mean(axis=1)                             # (N, 2)

    # Delaunay requires ≥ 3 non-collinear points
    if len(centroids) < 3 or np.linalg.matrix_rank(centroids - centroids[0]) < 2:
        return None

    try:
        edges, _ = build_delaunay_edges(centroids, distance_factor=DISTANCE_FACTOR)
    except Exception as exc:
        print(
            f"[WARN] Delaunay failed for "
            f"{hospital}/{patient_id}/{slide_id}/sec{section_id}: {exc}"
        )
        return None

    if len(edges) > 0:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    pos = torch.tensor(centroids, dtype=torch.float32)
    y   = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    data.patient_id       = patient_id
    data.slide_id         = slide_id
    data.section_id       = section_id
    data.hospital         = hospital
    data.metastasis_score = metastasis_score

    return data


def build_and_save_graphs(
    split_index: pd.DataFrame,
    split_name: str,
    out_dir: Path,
    df_npz: pd.DataFrame,
    dry_run: bool,
) -> list[dict]:
    """Phase 3: build one graph per (patient, slide, section), optionally save."""
    records   = []
    split_dir = out_dir / split_name

    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        split_index.iterrows(),
        total=len(split_index),
        desc=f"Building {split_name} graphs",
        unit="section",
    ):
        patient_id       = str(row["Patient_ID"])
        slide_id         = str(row["Slide"])
        section_id       = str(row["Section"])
        hospital         = str(row["Hospital"])
        label            = int(row["label"])
        metastasis_score = str(row["Metastasis_score"])

        try:
            data = build_graph_for_section(
                patient_id=patient_id,
                slide_id=slide_id,
                section_id=section_id,
                hospital=hospital,
                label=label,
                metastasis_score=metastasis_score,
                df_npz=df_npz,
            )
        except Exception as exc:
            print(
                f"[WARN] Error {hospital}/{patient_id}/{slide_id}/sec{section_id}: {exc}"
            )
            continue

        if data is None:
            print(
                f"[WARN] Skipping {hospital}/{patient_id}/{slide_id}/sec{section_id}"
                " — could not build graph."
            )
            continue

        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]

        if not dry_run:
            safe_slide = slide_id.replace("/", "_")
            fname    = f"{patient_id}_{safe_slide}_sec{section_id}.pt"
            out_path = split_dir / fname
            torch.save(data, out_path)

        records.append({
            "patient_id":       patient_id,
            "slide_id":         slide_id,
            "section_id":       section_id,
            "hospital":         hospital,
            "metastasis_score": metastasis_score,
            "label":            label,
            "n_nodes":          n_nodes,
            "n_edges":          n_edges,
        })

    return records


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def print_verification(
    train_records: list[dict],
    val_records: list[dict],
    out_dir: Path,
    dry_run: bool,
) -> None:
    """Phase 4: summary statistics and one example graph."""
    print("\n── Phase 4: Verification ──────────────────────────────────────────")

    def _stats(records: list[dict], key: str) -> tuple[int, float, int]:
        vals = [r[key] for r in records]
        return int(np.min(vals)), float(np.mean(vals)), int(np.max(vals))

    for name, records in [("Train", train_records), ("Val", val_records)]:
        if not records:
            print(f"  {name}: no graphs built.")
            continue
        nd = _stats(records, "n_nodes")
        ne = _stats(records, "n_edges")
        print(f"  {name} grafos : {len(records)}")
        print(f"  {name} nodos  (min/med/max) : {nd[0]} / {nd[1]:.1f} / {nd[2]}")
        print(f"  {name} aristas(min/med/max) : {ne[0]} / {ne[1]:.1f} / {ne[2]}")

    if not dry_run and out_dir.exists():
        total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.pt"))
        print(f"\n  Tamaño en disco : {total_bytes / 1e9:.3f} GB")

        pt_files = sorted((out_dir / "train").glob("*.pt"))
        if pt_files:
            try:
                g = torch.load(pt_files[0], weights_only=False)
                print(f"\n  Ejemplo: {pt_files[0].name}")
                print(f"    {g}")
                print(f"    patient_id       = {g.patient_id}")
                print(f"    metastasis_score = {g.metastasis_score}")
            except Exception as exc:
                print(f"[WARN] Could not load example graph: {exc}")
    else:
        print("\n  (dry_run: no .pt files written)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PyTorch Geometric graphs from WSI CLS bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path",    default="/mnt/iam", help="Dataset root")
    p.add_argument("--output_dir",  default="outputs/graphs",
                   help="Output directory for .pt files and index CSVs")
    p.add_argument("--dry_run",     action="store_true",
                   help="Run all phases but skip writing .pt files")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    iam_path = Path(args.iam_path)
    out_dir  = Path(args.output_dir)
    dry_run  = args.dry_run

    if dry_run:
        print("[INFO] --dry_run: no .pt files will be written.")

    cls_dir = iam_path / CLS_DIR_SUBPATH
    if not cls_dir.is_dir():
        sys.exit(f"[ERROR] CLS directory not found: {cls_dir}")
    print(f"[INFO] CLS dir : {cls_dir}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    df_npz   = load_all_npz(cls_dir)
    df_labels = load_labels(iam_path)

    # Expose the final DataFrame for inspection (useful when run interactively)
    df_full = df_npz.merge(df_labels, on="Patient_ID", how="inner")
    print(
        f"\n[INFO] Master DataFrame shape : {df_full.shape}"
        f"  columns : {list(df_full.columns)}"
    )
    print(df_full[["Patient_ID", "Slide", "Section", "Hospital", "Metastasis_score"]].head(3).to_string())

    section_index = build_slide_index(df_npz, df_labels)
    if section_index.empty:
        sys.exit("[ERROR] No valid sections after filtering.")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    train_idx, val_idx = split_patients(section_index)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print("\n── Phase 3: Building graphs ────────────────────────────────────────")

    train_records = build_and_save_graphs(
        split_index=train_idx,
        split_name="train",
        out_dir=out_dir,
        df_npz=df_npz,
        dry_run=dry_run,
    )
    val_records = build_and_save_graphs(
        split_index=val_idx,
        split_name="val",
        out_dir=out_dir,
        df_npz=df_npz,
        dry_run=dry_run,
    )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_records).to_csv(out_dir / "train_index.csv", index=False)
        pd.DataFrame(val_records).to_csv(out_dir / "val_index.csv",   index=False)
        print(f"[INFO] Index CSVs → {out_dir.resolve()}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    print_verification(train_records, val_records, out_dir, dry_run)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
