#!/usr/bin/env python3
"""
Build a PyTorch Geometric graph dataset from WSI CLS patches for GAT training.

Data sources
------------
- CLS embeddings : /mnt/iam/Experiments/MedImaging/ColonCancer/
                   CLS_datasets/NEW_DATASET_cls_2048/{Hospital}_CLS_2048.npz
- Labels          : .../xlsx_files/18_03_2026_..._fixed_N0s.xlsx

Graph structure (per section)
------------------------------
  Nodes = 2048×2048 patches; x = CLS embedding [N, 1536]; edge_index = Delaunay; y = patient label.

All graphs are written to a single flat directory (no train/val split).
The split is handled by k-fold CV in train.py at training time.

Usage
-----
    python scripts/build_dataset.py                   # one graph per section
    python scripts/build_dataset.py --dry_run
    python scripts/build_dataset.py --mega            # one mega-graph per patient
    python scripts/build_dataset.py --check           # list N0/N1 patients without CLS

Outputs
-------
Standard mode:
    ~/outputs/graphs/per-slide/{patient_id}_{slide_id}_sec{section_id}.pt

Mega-graph mode (--mega):
    ~/outputs/graphs/per-pacient/{patient_id}.pt

Both modes write:
    ~/outputs/graphs/{per-slide|per-pacient}/index.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
      in_cls          : apareix en algun fitxer *_CLS_2048.npz
      in_patches      : existeix el directori Patches2048/{hospital}/{patient_id}

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
    npz_files = sorted(cls_dir.glob("*_CLS_2048.npz"))
    print(f"\n  Directori CLS : {cls_dir}")
    print(f"  Fitxers *_CLS_2048.npz trobats : {len(npz_files)}")
    if npz_files:
        for f in npz_files:
            size_mb = f.stat().st_size / 1e6
            print(f"    [OK] {f.name}  ({size_mb:.1f} MB)")
    else:
        print("    [ERROR] Cap fitxer *_CLS_2048.npz trobat. Comprova la ruta --iam_path.")

    # ── resum per hospital (dels NPZ carregats) ────────────────────────────────
    print(f"\n  Pacients per hospital (dels NPZ carregats):")
    for hosp, grp in df_npz.groupby("Hospital"):
        n_pats   = grp["Patient_ID"].nunique()
        n_nodes  = len(grp)
        n_slides = grp.groupby(["Patient_ID", "Slide"]).ngroups
        print(f"    {hosp}: {n_pats} pacients, {n_slides} slides, {n_nodes:,} patches")

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
    print(f"    Patches totals carregats        : {len(df_npz):,}")

    # ── als NPZ però NO a l'Excel ──────────────────────────────────────────────
    only_in_npz = sorted(npz_patients - excel_patients)
    if only_in_npz:
        print(f"\n  [WARN] {len(only_in_npz)} pacient(s) als NPZ però NO a l'Excel "
              f"(seran ignorats al join):")
        for pid in only_in_npz:
            row_npz   = df_npz[df_npz["Patient_ID"] == pid]
            hospitals = ", ".join(row_npz["Hospital"].unique())
            n_nodes   = len(row_npz)
            n_slides  = row_npz["Slide"].nunique()
            print(f"    - {pid:>10}  hospital={hospitals}, {n_slides} slides, {n_nodes} patches")
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
    Inspect NPZ keys and verify that patch files on disk match the new 2048-px format.
    Each NPZ row corresponds to one 2048×2048 patch (one graph node).
    """
    print("\n── Verificació cobertura Patches2048 ───────────────────────────────")
    print(f"  Directori patches : {patches_dir}")
    print(f"  Patches totals als NPZ (= nodes del graf) : {len(df_npz):,}")

    npz_files = sorted(cls_dir.glob("*_CLS_2048.npz"))
    if npz_files:
        npz = np.load(npz_files[0], allow_pickle=True)
        print(f"\n  Camps del NPZ ({npz_files[0].name}): {list(npz.files)}")
        for key in npz.files:
            arr    = npz[key]
            shape  = arr.shape if hasattr(arr, "shape") else f"len={len(arr)}"
            dtype  = arr.dtype if hasattr(arr, "dtype") else type(arr)
            sample = str(arr.flat[0])[:80] if arr.size > 0 else ""
            print(f"    {key:20s} shape={shape}  dtype={dtype}  ex: {sample}")

    # Pick first patch from NPZ
    row        = df_npz.iloc[0]
    hospital   = str(row["Hospital"])
    patient_id = str(row["Patient_ID"])
    slide_id   = str(row["Slide"])
    section_id = str(row["Section"])
    j_c, i_c  = np.array(row["coords_bag"])  # single (x, y) coordinate

    print(f"\n  Primer patch: hospital={hospital!r}  patient={patient_id!r}  "
          f"slide={slide_id!r}  section={section_id!r}")
    print(f"  Coord (j, i) = ({j_c}, {i_c})")

    # Path is stored directly in NPZ; derive on-disk path via Linux convention
    path_str = str(row["paths_bag"])
    basename = path_str.replace("\\", "/").split("/")[-1]

    # Structure: patches_dir/{hospital}/{patient}/{slide}/patches/{basename}
    patch_file = patches_dir / hospital / patient_id / slide_id / "patches" / basename
    print(f"\n  Fitxer esperat : {patch_file}")
    print(f"  Existeix       : {patch_file.exists()}")
    if patch_file.exists():
        print("  [OK] Camp 'paths' del NPZ apunta correctament als fitxers JPEG.")
    else:
        # Show what's actually in the slide/patches dir
        patch_dir = patches_dir / hospital / patient_id / slide_id / "patches"
        if patch_dir.exists():
            samples = sorted(patch_dir.iterdir())[:5]
            print(f"  [WARN] Fitxer no trobat. Mostres al directori patches/:")
            for s in samples:
                print(f"    {s.name}")
        else:
            print(f"  [WARN] Directori patches/ no existeix: {patch_dir}")
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
        Patient_ID, Slide, Section, Hospital, Metastasis_score, label, n_nodes
    Only sections with >= MIN_BAGS_PER_SECTION nodes are kept.
    """
    print("\n── Phase 1: Building patient/slide/section index ──────────────────")

    merged = df_npz.merge(df_labels, on="Patient_ID", how="inner")
    print(f"[INFO] After label join: {len(merged):,} patches, {merged['Patient_ID'].nunique()} patients")

    section_counts = (
        merged.groupby(
            ["Patient_ID", "Slide", "Section", "Hospital", "Metastasis_score", "label"]
        )
        .size()
        .reset_index(name="n_nodes")
    )

    before = len(section_counts)
    section_counts = section_counts[section_counts["n_nodes"] >= MIN_BAGS_PER_SECTION].copy()
    print(
        f"[INFO] Dropped {before - len(section_counts)} sections with "
        f"< {MIN_BAGS_PER_SECTION} nodes"
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
        f"  Nodes per section      : "
        f"min={section_counts['n_nodes'].min()} / "
        f"media={section_counts['n_nodes'].mean():.1f} / "
        f"max={section_counts['n_nodes'].max()}"
    )
    print(f"  Hospitales             : {sorted(section_counts['Hospital'].unique())}")

    return section_counts.reset_index(drop=True)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

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

    Nodes      = 2048×2048 patches in this section
    x          = CLS embedding per patch  [N, 1536]  — real UNI2 features
    pos        = (x, y) WSI-level coordinate of each patch  [N, 2]
    edge_index = Delaunay on patch positions (bidirectional, long edges pruned)
    y          = patient-level label {0=N0, 1=N1}

    Returns None if the section has fewer than MIN_BAGS_PER_SECTION patches
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

    # ── node positions: direct (x, y) coord of each 2048×2048 patch ──────────
    coord_arrays = np.stack(df_section["coords_bag"].tolist(), axis=0)  # (N, 2)
    centroids    = coord_arrays.astype(np.float64)                       # (N, 2)

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

    # patch_j/patch_i: WSI level-0 pixel coords of each patch (for visualisation)
    # patch_idx: node index within this section (for downstream lookups)
    paths_list  = df_section["paths_bag"].tolist()      # N strings
    patch_j_list:   list[int] = []
    patch_i_list:   list[int] = []
    patch_idx_list: list[int] = []
    for n in range(len(coord_arrays)):
        j_c, i_c = coord_arrays[n]
        patch_j_list.append(int(j_c))
        patch_i_list.append(int(i_c))
        patch_idx_list.append(n)     # node index within this section

    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    data.patient_id       = patient_id
    data.slide_id         = slide_id
    data.section_id       = section_id
    data.hospital         = hospital
    data.metastasis_score = metastasis_score
    data.patch_j          = torch.tensor(patch_j_list,   dtype=torch.int32)
    data.patch_i          = torch.tensor(patch_i_list,   dtype=torch.int32)
    data.patch_idx        = torch.tensor(patch_idx_list, dtype=torch.int32)

    return data


def build_and_save_graphs(
    section_index: pd.DataFrame,
    out_dir: Path,
    df_npz: pd.DataFrame,
    dry_run: bool,
) -> list[dict]:
    """Phase 2: build one graph per (patient, slide, section), optionally save to disk."""
    records = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        section_index.iterrows(),
        total=len(section_index),
        desc="Building graphs",
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

        if not dry_run:
            safe_slide = slide_id.replace("/", "_")
            fname      = f"{patient_id}_{safe_slide}_sec{section_id}.pt"
            torch.save(data, out_dir / fname)

        records.append({
            "patient_id":       patient_id,
            "slide_id":         slide_id,
            "section_id":       section_id,
            "hospital":         hospital,
            "metastasis_score": metastasis_score,
            "label":            label,
            "n_nodes":          data.x.shape[0],
            "n_edges":          data.edge_index.shape[1],
        })

    return records


# ── Mega-graph (Model 2) ──────────────────────────────────────────────────────

def build_patient_mega_graph(
    patient_id: str,
    patient_rows: pd.DataFrame,
    df_npz: pd.DataFrame,
) -> "Data | None":
    """Build one mega-graph for all sections of a patient (Model 2 / --mega mode).

    Concatenates feature matrices vertically and places each section's adjacency
    matrix on the block diagonal, so no edges cross section boundaries:

        X_pac = [X_1; ...; X_S]   ∈ R^(ΣN_i × 1536)
        A_pac = block_diag(A_1, ..., A_S)   (zero off-diagonal blocks)

    This allows the GAT to attend over all sections simultaneously while a
    single global pooling produces the patient-level prediction directly,
    without a separate MIL aggregation stage.

    Returns None if no valid section graphs could be built.
    """
    x_list, pos_list, ei_list = [], [], []
    patch_j_list, patch_i_list, patch_idx_list = [], [], []
    node_offset = 0

    for _, row in patient_rows.iterrows():
        section_data = build_graph_for_section(
            patient_id=patient_id,
            slide_id=str(row["Slide"]),
            section_id=str(row["Section"]),
            hospital=str(row["Hospital"]),
            label=int(row["label"]),
            metastasis_score=str(row["Metastasis_score"]),
            df_npz=df_npz,
        )
        if section_data is None:
            continue
        x_list.append(section_data.x)
        pos_list.append(section_data.pos)
        # Offset edge indices so each section's nodes are addressed correctly
        ei_list.append(section_data.edge_index + node_offset)
        patch_j_list.append(section_data.patch_j)
        patch_i_list.append(section_data.patch_i)
        patch_idx_list.append(section_data.patch_idx)
        node_offset += section_data.x.shape[0]

    if not x_list:
        return None

    first_row = patient_rows.iloc[0]
    data = Data(
        x          = torch.cat(x_list,   dim=0),
        edge_index = torch.cat(ei_list,  dim=1),
        pos        = torch.cat(pos_list, dim=0),
        y          = torch.tensor([int(first_row["label"])], dtype=torch.long),
    )
    data.patient_id       = patient_id
    data.hospital         = str(first_row["Hospital"])
    data.metastasis_score = str(first_row["Metastasis_score"])
    data.patch_j          = torch.cat(patch_j_list)
    data.patch_i          = torch.cat(patch_i_list)
    data.patch_idx        = torch.cat(patch_idx_list)
    return data


def build_and_save_mega_graphs(
    section_index: pd.DataFrame,
    out_dir: Path,
    df_npz: pd.DataFrame,
    dry_run: bool,
) -> list[dict]:
    """Phase 2 (mega mode): build one graph per patient from all its sections."""
    records = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = section_index["Patient_ID"].unique()
    for patient_id in tqdm(patient_ids, desc="Building mega-graphs", unit="patient"):
        patient_rows = section_index[section_index["Patient_ID"] == patient_id]
        try:
            data = build_patient_mega_graph(patient_id, patient_rows, df_npz)
        except Exception as exc:
            print(f"[WARN] Error building mega-graph for {patient_id}: {exc}")
            continue

        if data is None:
            print(f"[WARN] Skipping {patient_id} — could not build any section graph.")
            continue

        if not dry_run:
            torch.save(data, out_dir / f"{patient_id}.pt")

        records.append({
            "patient_id":       patient_id,
            "hospital":         data.hospital,
            "metastasis_score": data.metastasis_score,
            "label":            data.y.item(),
            "n_sections":       len(patient_rows),
            "n_nodes":          data.x.shape[0],
            "n_edges":          data.edge_index.shape[1],
        })

    return records


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def print_verification(records: list[dict], out_dir: Path, dry_run: bool) -> None:
    """Phase 3: summary statistics and one example graph load."""
    print("\n── Phase 3: Verification ──────────────────────────────────────────")

    if not records:
        print("  No graphs built.")
        return

    def _stats(key: str) -> tuple[int, float, int]:
        vals = [r[key] for r in records]
        return int(np.min(vals)), float(np.mean(vals)), int(np.max(vals))

    nd = _stats("n_nodes")
    ne = _stats("n_edges")
    n_patients = len({r["patient_id"] for r in records})
    n0 = sum(1 for r in records if r["label"] == 0)
    n1 = sum(1 for r in records if r["label"] == 1)

    print(f"  Grafos totals  : {len(records)}")
    print(f"  Pacients únics : {n_patients}  (N0 grafos={n0}, N1 grafos={n1})")
    print(f"  Nodes  (min/med/max) : {nd[0]} / {nd[1]:.1f} / {nd[2]}")
    print(f"  Arestes(min/med/max) : {ne[0]} / {ne[1]:.1f} / {ne[2]}")

    if not dry_run and out_dir.exists():
        total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
        print(f"\n  Tamaño en disco : {total_bytes / 1e9:.3f} GB")

        pt_files = sorted(out_dir.glob("*.pt"))
        if pt_files:
            try:
                g = torch.load(pt_files[0], weights_only=False)
                print(f"\n  Exemple: {pt_files[0].name}")
                print(f"    {g}")
                print(f"    patient_id       = {g.patient_id}")
                print(f"    metastasis_score = {g.metastasis_score}")
            except Exception as exc:
                print(f"[WARN] Could not load example graph: {exc}")
    else:
        print("\n  (dry_run: no .pt files written)")


# ── Pacients N0/N1 sense CLS ──────────────────────────────────────────────────

def print_missing_cls_patients(df_labels: pd.DataFrame, cls_dir: Path) -> None:
    """
    Llista els pacients amb diagnòstic N0 o N1 (NX ja exclosos de df_labels)
    que no tenen cap fitxer CLS generat al directori indicat.
    """
    print("\n── Pacients N0/N1 sense CLS ────────────────────────────────────────")
    print(f"  Directori CLS : {cls_dir}")

    npz_patients: set[str] = set()
    for npz_path in sorted(cls_dir.glob("*_CLS_2048.npz")):
        try:
            npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
            npz_patients.update(str(p) for p in npz["patient_list"])
        except Exception as exc:
            print(f"  [WARN] No s'ha pogut llegir {npz_path.name}: {exc}")

    excel_patients = set(df_labels["Patient_ID"].unique())
    missing = sorted(excel_patients - npz_patients)

    n0_miss = sum(
        1 for pid in missing
        if df_labels.loc[df_labels["Patient_ID"] == pid, "label"].values[0] == 0
    )
    n1_miss = len(missing) - n0_miss

    print(f"  Total N0/N1 a l'Excel : {len(excel_patients)}")
    print(f"  Amb CLS disponible    : {len(excel_patients & npz_patients)}")
    print(f"  Sense CLS             : {len(missing)}  (N0={n0_miss}, N1={n1_miss})")

    if missing:
        print()
        for pid in missing:
            rows  = df_labels[df_labels["Patient_ID"] == pid]
            score = rows["Metastasis_score"].values[0] if len(rows) else "?"
            print(f"    {pid:>12s}  {score}")
    else:
        print("  [OK] Tots els pacients N0/N1 tenen CLS.")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PyTorch Geometric graphs from WSI CLS patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iam_path",   default="/mnt/iam", help="Dataset root")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for .pt files and index CSVs "
                        "(default: ~/outputs/graphs/per-slide or ~/outputs/graphs/per-pacient with --mega)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Run all phases but skip writing .pt files")
    p.add_argument("--mega",       action="store_true",
                   help="Build one mega-graph per patient (block-diagonal adjacency "
                        "across all sections). Train with patient_level=false.")
    p.add_argument("--check",      action="store_true",
                   help="Only report which N0/N1 patients lack CLS embeddings, then exit.")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    iam_path = Path(args.iam_path)
    dry_run  = args.dry_run
    mega     = args.mega

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
    else:
        base = Path.home() / "outputs" / "graphs"
        out_dir = base / ("per-pacient" if mega else "per-slide")

    if dry_run:
        print("[INFO] --dry_run: no .pt files will be written.")
    if mega:
        print("[INFO] --mega: building one graph per patient (block-diagonal adjacency).")

    cls_dir = iam_path / CLS_DIR_SUBPATH
    if not cls_dir.is_dir():
        sys.exit(f"[ERROR] CLS directory not found: {cls_dir}")
    print(f"[INFO] CLS dir    : {cls_dir}")
    print(f"[INFO] Output dir : {out_dir}")

    # ── Mode --check ──────────────────────────────────────────────────────────
    if args.check:
        df_labels = load_labels(iam_path)
        print_missing_cls_patients(df_labels, cls_dir)
        return

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

    # ── Phase 2: Build graphs (all in one flat directory) ─────────────────────
    print("\n── Phase 2: Building graphs ────────────────────────────────────────")

    if mega:
        records = build_and_save_mega_graphs(
            section_index=section_index,
            out_dir=out_dir,
            df_npz=df_npz,
            dry_run=dry_run,
        )
    else:
        records = build_and_save_graphs(
            section_index=section_index,
            out_dir=out_dir,
            df_npz=df_npz,
            dry_run=dry_run,
        )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_dir / "index.csv", index=False)
        print(f"[INFO] Index CSV → {out_dir.resolve()}")

    # ── Phase 3: Verification ─────────────────────────────────────────────────
    print_verification(records, out_dir, dry_run)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
