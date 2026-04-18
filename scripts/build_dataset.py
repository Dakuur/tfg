#!/usr/bin/env python3
"""
Build a PyTorch Geometric graph dataset from WSI CLS bags for GAT training.

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

Graph structure (per section)
------------------------------
  Nodes      = bags belonging to that section
  x          = CLS embedding   [N_bags, 1536]
  pos        = centroid of the bag's 256 patch coords   [N_bags, 2]
  edge_index = Delaunay triangulation on centroids (bidirectional, pruned)
  y          = patient-level label {0=N0, 1=N1}

Phases
------
1. Build patient/slide/section index (npz → label join, min-bag filter)
2. 80/20 stratified split at patient level (no data leakage)
3. Construct graphs and save as .pt files
4. Verification summary

Usage
-----
    python scripts/build_dataset.py                   # full run
    python scripts/build_dataset.py --dry_run         # index + split, no .pt files
    python scripts/build_dataset.py --iam_path /mnt/iam

Outputs
-------
    outputs/graphs/train/{patient_id}_{slide_id}_sec{section_id}.pt
    outputs/graphs/val/{patient_id}_{slide_id}_sec{section_id}.pt
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

# ── local imports ──────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from graph_utils import (  # noqa: E402
    build_delaunay_edges,
    make_edge_index,
    MIN_BAGS_PER_SECTION,
    DISTANCE_FACTOR,
)
from wsi_io import (  # noqa: E402
    load_all_npz, load_labels, find_patches_dir,
    CLS_DIR_SUBPATH, LABELS_SUBPATH,
)

# ── constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42


# ── Informe de cobertura de dades ─────────────────────────────────────────────

def generate_coverage_report(
    df_npz:    pd.DataFrame,
    iam_path:  Path,
    out_dir:   Path,
) -> None:
    """
    Genera un CSV a `out_dir/coverage_report.csv` amb una fila per (pacient, hospital)
    i tres indicadors de presència de dades:

      in_excel        : apareix a l'Excel i el diagnòstic NO és NX
      excel_score     : N0, N1, NX o "-" (no trobat a l'Excel)
      in_cls          : apareix en algun fitxer *_CLS.npz
      in_patches      : existeix el directori Patches/{hospital}/{patient_id}

    L'informe cobreix la unió de tots els pacients coneguts (Excel + CLS NPZ).
    """
    excel_path = iam_path / LABELS_SUBPATH

    # ── carregar Excel (incloent NX) ──────────────────────────────────────────
    df_excel_raw: pd.DataFrame | None = None
    if excel_path.exists():
        try:
            df_excel_raw = pd.read_excel(excel_path)
            score_col = next(
                (c for c in df_excel_raw.columns if str(c).startswith("PATHOLOGIST SCORE")), None
            )
            if score_col and "CODE" in df_excel_raw.columns:
                df_excel_raw = df_excel_raw.rename(
                    columns={score_col: "Metastasis_score", "CODE": "Patient_ID"}
                )
                df_excel_raw["Patient_ID"] = df_excel_raw["Patient_ID"].astype(str).str.strip()
                # Normalitza subtipus N1
                df_excel_raw["Metastasis_score"] = df_excel_raw["Metastasis_score"].replace(
                    {"N1a": "N1", "N1b": "N1", "N1c": "N1", "N2a": "N1", "N2b": "N1"}
                )
                hospital_col = "Data Access Group" if "Data Access Group" in df_excel_raw.columns else None
                if hospital_col:
                    df_excel_raw["hospital_excel"] = df_excel_raw[hospital_col].astype(str).str.strip()
                else:
                    df_excel_raw["hospital_excel"] = "-"
            else:
                df_excel_raw = None
        except Exception as exc:
            print(f"[WARN] coverage_report: could not read Excel: {exc}")
            df_excel_raw = None
    else:
        print(f"[WARN] coverage_report: Excel not found at {excel_path}")

    # ── resum per pacient des dels NPZ ────────────────────────────────────────
    # Un pacient pot tenir dades en múltiples hospitals (rar però possible)
    npz_patient_hospital: dict[str, set[str]] = {}
    for _, row in df_npz[["Patient_ID", "Hospital"]].drop_duplicates().iterrows():
        pid  = str(row["Patient_ID"]).strip()
        hosp = str(row["Hospital"]).strip()
        npz_patient_hospital.setdefault(pid, set()).add(hosp)

    # ── construir registres per (pacient, hospital) ───────────────────────────
    excel_lookup: dict[str, dict] = {}
    if df_excel_raw is not None:
        for _, row in df_excel_raw.iterrows():
            pid = str(row["Patient_ID"]).strip()
            excel_lookup[pid] = {
                "score":           str(row.get("Metastasis_score", "-")),
                "hospital_excel":  str(row.get("hospital_excel", "-")),
            }

    # Union de tots els pacients coneguts
    all_patients: set[str] = set(npz_patient_hospital.keys()) | set(excel_lookup.keys())

    # Directori de patches
    try:
        patches_dir: Path | None = find_patches_dir(iam_path)
    except FileNotFoundError:
        patches_dir = None
        print("[WARN] coverage_report: Patches directory not found — in_patches will be False")

    records: list[dict] = []
    for pid in sorted(all_patients):
        exc = excel_lookup.get(pid, {})
        score         = exc.get("score", "-")
        hospital_excel = exc.get("hospital_excel", "-")
        in_excel_valid = score not in ("-", "NX")
        hospitals_cls  = sorted(npz_patient_hospital.get(pid, set()))

        # Si no surt als NPZ, usem hospital de l'Excel
        hospitals_to_check = hospitals_cls if hospitals_cls else (
            [hospital_excel] if hospital_excel != "-" else []
        )

        in_patches = False
        if patches_dir and hospitals_to_check:
            for hosp in hospitals_to_check:
                if (patches_dir / hosp / pid).is_dir():
                    in_patches = True
                    break

        # Generar una fila per cada hospital CLS; si no n'hi ha, una fila amb hospital de l'Excel
        if hospitals_cls:
            for hosp in hospitals_cls:
                records.append({
                    "patient_id":       pid,
                    "hospital":         hosp,
                    "excel_score":      score,
                    "in_excel":         in_excel_valid,
                    "in_cls":           True,
                    "in_patches":  in_patches,
                })
        else:
            records.append({
                "patient_id":       pid,
                "hospital":         hospital_excel,
                "excel_score":      score,
                "in_excel":         in_excel_valid,
                "in_cls":           False,
                "in_patches":       in_patches,
            })

    df_report = pd.DataFrame(records, columns=[
        "patient_id", "hospital", "excel_score",
        "in_excel", "in_cls", "in_patches",
    ])
    df_report.sort_values(["hospital", "patient_id"], inplace=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coverage_report.csv"
    df_report.to_csv(out_path, index=False)

    total = len(df_report)
    n_both    = ((df_report["in_excel"]) & (df_report["in_cls"])).sum()
    n_only_e  = ((df_report["in_excel"]) & (~df_report["in_cls"])).sum()
    n_only_c  = ((~df_report["in_excel"]) & (df_report["in_cls"])).sum()
    n_patches = df_report["in_patches"].sum()
    print(f"\n── Coverage report ─────────────────────────────────────────────────")
    print(f"  Pacients totals : {total}")
    print(f"  Excel ∩ CLS     : {n_both}   (usables per entrenament)")
    print(f"  Només Excel     : {n_only_e}  (sense embeddings UNI2)")
    print(f"  Només CLS       : {n_only_c}  (no a l'Excel o diagnòstic NX)")
    print(f"  Amb patches     : {n_patches}")
    print(f"  Desat a         : {out_path.resolve()}")


# ── Diagnòstic de cobertura de dades ──────────────────────────────────────────

def print_data_diagnostics(
    df_npz: pd.DataFrame,
    df_labels: pd.DataFrame,
    cls_dir: Path,
) -> None:
    """
    Imprimeix un resum detallat de la cobertura de dades i dels registres
    desemparellats entre els fitxers CLS (.npz) i l'Excel d'etiquetes.

    Útil per detectar pacients sense embeddings, fitxers mal ubicats o
    discrepàncies entre la base de dades i les dades reals.
    """
    print("\n── Diagnòstic de cobertura de dades ────────────────────────────────")

    # ── fitxers NPZ trobats al disc ────────────────────────────────────────────
    npz_files = sorted(cls_dir.glob("*_CLS.npz"))
    print(f"\n  Directori CLS : {cls_dir}")
    print(f"  Fitxers *_CLS.npz trobats : {len(npz_files)}")
    if npz_files:
        for f in npz_files:
            size_mb = f.stat().st_size / 1e6
            print(f"    [OK] {f.name}  ({size_mb:.1f} MB)")
    else:
        print("    [ERROR] Cap fitxer *_CLS.npz trobat. Comprova la ruta --iam_path.")

    # ── resum per hospital (dels NPZ carregats) ────────────────────────────────
    print(f"\n  Pacients per hospital (dels NPZ carregats):")
    for hosp, grp in df_npz.groupby("Hospital"):
        n_pats  = grp["Patient_ID"].nunique()
        n_bags  = len(grp)
        n_slides = grp.groupby(["Patient_ID", "Slide"]).ngroups
        print(f"    {hosp}: {n_pats} pacients, {n_slides} slides, {n_bags:,} bags")

    # ── resum global ───────────────────────────────────────────────────────────
    npz_patients   = set(df_npz["Patient_ID"].unique())
    excel_patients = set(df_labels["Patient_ID"].unique())
    common         = npz_patients & excel_patients

    n0_excel = (df_labels["label"] == 0).sum()
    n1_excel = (df_labels["label"] == 1).sum()

    print(f"\n  Resum global:")
    print(f"    Pacients únics als NPZ          : {len(npz_patients)}")
    print(f"    Pacients únics a l'Excel        : {len(excel_patients)}  "
          f"(N0={n0_excel}, N1={n1_excel})")
    print(f"    Pacients en comú (intersecció)  : {len(common)}")
    print(f"    Bags totals carregats           : {len(df_npz):,}")

    # ── als NPZ però NO a l'Excel ──────────────────────────────────────────────
    only_in_npz = sorted(npz_patients - excel_patients)
    if only_in_npz:
        print(f"\n  [WARN] {len(only_in_npz)} pacient(s) als NPZ però NO a l'Excel "
              f"(seran ignorats al join):")
        for pid in only_in_npz:
            row_npz  = df_npz[df_npz["Patient_ID"] == pid]
            hospitals = ", ".join(row_npz["Hospital"].unique())
            n_bags    = len(row_npz)
            n_slides  = row_npz["Slide"].nunique()
            print(f"    - {pid:>10}  hospital={hospitals}, {n_slides} slides, {n_bags} bags")
    else:
        print("\n  [OK] Tots els pacients dels NPZ estan a l'Excel.")

    # ── a l'Excel però sense dades CLS ────────────────────────────────────────
    only_in_excel = sorted(excel_patients - npz_patients)
    if only_in_excel:
        print(f"\n  [WARN] {len(only_in_excel)} pacient(s) a l'Excel però sense dades CLS als NPZ "
              f"(no es construirà cap graf):")
        for pid in only_in_excel:
            rows  = df_labels[df_labels["Patient_ID"] == pid]
            score = rows["Metastasis_score"].values[0] if len(rows) else "?"
            print(f"    - {pid:>10}  score={score}")
    else:
        print("\n  [OK] Tots els pacients de l'Excel tenen dades CLS als NPZ.")

    print()


# ── Verificació de cobertura de patches ──────────────────────────────────────

def verify_patch_coverage(df_npz: pd.DataFrame, patches_dir: Path, cls_dir: Path) -> None:
    """
    Inspect NPZ keys and check naming convention in Patches/ for one representative bag.
    """
    print("\n── Verificació cobertura Patches ───────────────────────────────────")
    print(f"  Directori patches : {patches_dir}")
    print(f"  Bags totals als NPZ (= nodes del graf) : {len(df_npz):,}")

    # Inspect all keys in first NPZ file
    npz_files = sorted(cls_dir.glob("*_CLS.npz"))
    if npz_files:
        npz = np.load(npz_files[0], allow_pickle=True)
        print(f"\n  Camps del NPZ ({npz_files[0].name}): {list(npz.keys())}")
        for key in npz.keys():
            arr = npz[key]
            shape = arr.shape if hasattr(arr, "shape") else f"len={len(arr)}"
            dtype = arr.dtype if hasattr(arr, "dtype") else type(arr)
            sample = str(arr[0])[:80] if len(arr) > 0 else ""
            print(f"    {key:20s} shape={shape}  dtype={dtype}  ex: {sample}")

    # Pick first bag from NPZ
    row        = df_npz.iloc[0]
    hospital   = str(row["Hospital"])
    patient_id = str(row["Patient_ID"])
    slide_id   = str(row["Slide"])
    section_id = str(row["Section"])
    bag_coords = np.array(row["coords_bag"])   # (256, 2)
    centroid   = bag_coords.mean(axis=0)
    dists      = np.linalg.norm(bag_coords - centroid, axis=1)
    central_in_bag = int(dists.argmin())
    j_c, i_c   = bag_coords[central_in_bag]

    print(f"\n  Primer bag: hospital={hospital!r}  patient={patient_id!r}  "
          f"slide={slide_id!r}  section={section_id!r}")
    print(f"  Coord central (j,i) = ({j_c:.0f}, {i_c:.0f})  "
          f"  (índex dins bag: {central_in_bag}/255)")

    slide_dir = patches_dir / hospital / patient_id / slide_id
    print(f"\n  Directori slide: {slide_dir}  exists={slide_dir.exists()}")

    if slide_dir.exists():
        all_files = sorted(slide_dir.iterdir())
        print(f"  Total fitxers al directori: {len(all_files)}")
        print(f"  Primeres 8 mostres:")
        for f in all_files[:8]:
            print(f"    {f.name}")
        extensions = {f.suffix.lower() for f in all_files if f.is_file()}
        print(f"  Extensions trobades: {extensions}")

        # Try to infer naming: {hospital}_{patient}_{slide}_{section}_{index}.ext
        prefix = f"{hospital}_{patient_id}_{slide_id}_{section_id}_"
        section_files = sorted(
            [f for f in all_files if f.name.startswith(prefix)],
            key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else 0
        )
        print(f"\n  Fitxers de la secció {section_id!r} (prefix={prefix!r}): {len(section_files)}")
        if section_files:
            print(f"  Rang d'índexs: 0 … {len(section_files)-1}")
            # Estimate which bag index (0-based) this row is within the section
            df_section = df_npz[
                (df_npz["Patient_ID"] == patient_id) &
                (df_npz["Slide"]      == slide_id)   &
                (df_npz["Section"]    == section_id) &
                (df_npz["Hospital"]   == hospital)
            ]
            bag_idx_in_section = int(df_section.index.get_loc(df_npz.index[0]))
            patch_global_idx   = bag_idx_in_section * 256 + central_in_bag
            print(f"  Bag index dins secció: {bag_idx_in_section}  "
                  f"→ patch global estimat: {patch_global_idx}")
            if patch_global_idx < len(section_files):
                guessed = section_files[patch_global_idx]
                print(f"  Fitxer estimat pel patch central: {guessed.name}  exists={guessed.exists()}")
            else:
                print(f"  [WARN] Índex estimat {patch_global_idx} fora de rang ({len(section_files)} fitxers)")
    print()


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def build_slide_index(df_npz: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: join npz bags with labels and build a per-section index.

    The graph unit is (Patient_ID, Slide, Section): each section is a
    separate tissue piece on the glass slide and must be its own graph.
    Grouping at slide level would mix spatially disjoint sections and
    produce spurious Delaunay edges between unrelated tissue regions.

    Returns a DataFrame (one row per section) with columns:
        Patient_ID, Slide, Section, Hospital, Metastasis_score, label, n_bags
    Only sections with >= MIN_BAGS_PER_SECTION bags are kept.
    """
    print("\n── Phase 1: Building patient/slide/section index ──────────────────")

    merged = df_npz.merge(df_labels, on="Patient_ID", how="inner")
    print(f"[INFO] After label join: {len(merged):,} bags, {merged['Patient_ID'].nunique()} patients")

    section_counts = (
        merged.groupby(
            ["Patient_ID", "Slide", "Section", "Hospital", "Metastasis_score", "label"]
        )
        .size()
        .reset_index(name="n_bags")
    )

    before = len(section_counts)
    section_counts = section_counts[section_counts["n_bags"] >= MIN_BAGS_PER_SECTION].copy()
    print(
        f"[INFO] Dropped {before - len(section_counts)} sections with "
        f"< {MIN_BAGS_PER_SECTION} bags"
    )

    n_pat      = section_counts["Patient_ID"].nunique()
    per_patient = section_counts.drop_duplicates("Patient_ID")
    n0         = (per_patient["label"] == 0).sum()
    n1         = (per_patient["label"] == 1).sum()
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
) -> "Data | None":
    """
    Build one torch_geometric.data.Data object for a single (slide, section).

    Each section is an independent tissue piece — one graph per section
    prevents Delaunay from connecting spatially disjoint regions.

    Nodes      = bags in this section
    x          = CLS embedding per bag  [N, 1536]  — real UNI2 features
    pos        = centroid of the bag's 256 patch coords  [N, 2]
    edge_index = Delaunay on centroids (bidirectional, long edges pruned)
    y          = patient-level label {0=N0, 1=N1}

    Returns None if the section has fewer than MIN_BAGS_PER_SECTION bags
    or if Delaunay cannot be computed (e.g. collinear points).
    """
    mask = (
        (df_npz["Patient_ID"] == patient_id) &
        (df_npz["Slide"]      == slide_id)   &
        (df_npz["Section"]    == section_id) &
        (df_npz["Hospital"]   == hospital)
    )
    df_section = df_npz[mask]

    if len(df_section) < MIN_BAGS_PER_SECTION:
        return None

    # ── node features: real CLS embeddings ────────────────────────────────────
    cls_arrays = np.stack(df_section["CLS"].tolist(), axis=0)          # (N, 1536)
    x          = torch.tensor(cls_arrays, dtype=torch.float32)

    # ── node positions: centroid of each bag's 256 patch coords ───────────────
    coord_arrays = np.stack(df_section["coords_bag"].tolist(), axis=0)  # (N, 256, 2)
    centroids    = coord_arrays.mean(axis=1)                             # (N, 2)

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

    edge_index = make_edge_index(edges)
    pos        = torch.tensor(centroids, dtype=torch.float32)
    y          = torch.tensor([label],   dtype=torch.long)

    # For each bag (node), find the most central patch in its 256-patch bag.
    # Stored as (j, i) WSI level-0 pixel coordinates so the frontend can
    # reconstruct the patch filename: {hospital}_{patient}_{slide}_{j}_{i}.jpg
    patch_j_list: list[int] = []
    patch_i_list: list[int] = []
    for n, bag_coords in enumerate(coord_arrays):      # (256, 2) each
        centroid    = centroids[n]                      # (j_c, i_c)
        dists       = np.linalg.norm(bag_coords - centroid, axis=1)
        central_idx = int(dists.argmin())
        j_c, i_c   = bag_coords[central_idx]
        patch_j_list.append(int(round(j_c)))
        patch_i_list.append(int(round(i_c)))

    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    data.patient_id       = patient_id
    data.slide_id         = slide_id
    data.section_id       = section_id
    data.hospital         = hospital
    data.metastasis_score = metastasis_score
    data.patch_j          = torch.tensor(patch_j_list, dtype=torch.int32)
    data.patch_i          = torch.tensor(patch_i_list, dtype=torch.int32)

    return data


def build_and_save_graphs(
    split_index: pd.DataFrame,
    split_name: str,
    out_dir: Path,
    df_npz: pd.DataFrame,
    dry_run: bool,
) -> list[dict]:
    """Phase 3: build one graph per (patient, slide, section), optionally save to disk."""
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
            print(f"[WARN] Error {hospital}/{patient_id}/{slide_id}/sec{section_id}: {exc}")
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
            fname      = f"{patient_id}_{safe_slide}_sec{section_id}.pt"
            torch.save(data, split_dir / fname)

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
    """Phase 4: summary statistics and one example graph load."""
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
    p.add_argument("--iam_path",   default="/mnt/iam", help="Dataset root")
    p.add_argument("--output_dir", default="outputs/graphs",
                   help="Output directory for .pt files and index CSVs")
    p.add_argument("--dry_run",    action="store_true",
                   help="Run all phases but skip writing .pt files")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
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
    df_npz    = load_all_npz(cls_dir)
    df_labels = load_labels(iam_path)

    generate_coverage_report(df_npz, iam_path, out_dir)
    print_data_diagnostics(df_npz, df_labels, cls_dir)

    try:
        patches_dir = find_patches_dir(iam_path)
        verify_patch_coverage(df_npz, patches_dir, cls_dir)
    except FileNotFoundError as exc:
        print(f"[WARN] Patches dir not found, skipping coverage check: {exc}")

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
        pd.DataFrame(val_records).to_csv(out_dir / "val_index.csv",     index=False)
        print(f"[INFO] Index CSVs → {out_dir.resolve()}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    print_verification(train_records, val_records, out_dir, dry_run)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
