"""
data_loaders.py  –  Universal Data Loaders for Drug-ADR Link Prediction
========================================================================
Converts every supported dataset format into the DrugADRDataset API
defined in graph_kernel_adr_prediction.py.

Expected folder layout
----------------------
  your_project/
  ├── graph_kernel_adr_prediction.py
  ├── data_loaders.py
  └── Dataset/                        ← default data_dir
      ├── Dataset 1_Drug ADR Relation.csv
      ├── Dataset 2_SIDER 4.1_drug_adr_associations.csv
      ├── Dataset 3_Neonatal_Drug_ADR_Expanded.csv
      ├── Dataset 4_WebMD.csv
      ├── TWOSIDES.csv
      ├── OFFSIDES.csv                (download separately)
      ├── bio-decagon-mono.csv        (download separately)
      ├── bio-decagon-combo.csv       (download separately)
      └── webmd.csv                   (download separately)

Supported sources
-----------------
YOUR DATASETS:
  DS1  – Drug_ADR_Relation           (single-drug,  5 730 rows)
  DS2  – SIDER_4_1                   (single-drug, 158 559 rows)
  DS3  – Neonatal_Drug_ADR_Expanded  (single + multi-drug, 203 616 rows)

PUBLIC BENCHMARK DATASETS:
  OFFSIDES   – single-drug ADR, ~1 332 drugs, ~10 k ADRs
  TWOSIDES   – drug-pair ADR,   ~645 drugs, 63 k pairs, 1 318 ADRs
  SNAP Mono  – Stanford SNAP Decagon monopharmacy graph
  SNAP Poly  – Stanford SNAP Decagon polypharmacy graph
  WebMD      – patient drug-review ADR mentions (Kaggle)

Each loader returns a DrugADRDataset ready for run_full_pipeline().

Author : Generated for Manuscript_Draft_1 experiments
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Import the dataset class from the kernel pipeline ───────────────────────
# If running standalone, define a thin stub so loaders work independently.
try:
    from graph_kernel_adr_prediction import DrugADRDataset
except ImportError:
    class DrugADRDataset:          # type: ignore[no-redef]
        """Minimal stub – replace with the real class from graph_kernel_adr_prediction.py"""
        def __init__(self, pairs, triples=None):
            self.drug_adr_pairs          = pairs
            self.drug_pair_adr_triples   = triples or []
            drugs = sorted({d for d, _ in pairs})
            adrs  = sorted({a for _, a in pairs})
            self.drug2idx = {d: i for i, d in enumerate(drugs)}
            self.adr2idx  = {a: i for i, a in enumerate(adrs)}
            self.n_drugs  = len(drugs)
            self.n_adrs   = len(adrs)
        def __repr__(self):
            return (f"DrugADRDataset(drugs={self.n_drugs}, adrs={self.n_adrs}, "
                    f"edges={len(self.drug_adr_pairs)}, "
                    f"poly_triples={len(self.drug_pair_adr_triples)})")

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FILE READER  – handles both .xlsx and .csv transparently
# ════════════════════════════════════════════════════════════════════════════

def _read_file(
    filepath: str,
    usecols=None,
    nrows: Optional[int] = None,
    sep: str = ",",
    dtype=None,
    on_bad_lines: str = "skip",
    sheet_name: int = 0,
) -> pd.DataFrame:
    """
    Read a dataset file regardless of whether it is .xlsx/.xls or .csv/.tsv.

    Strategy
    --------
    1. Try the format implied by the file extension first.
    2. If that fails (e.g. a CSV saved with an .xlsx name, or vice versa),
       automatically fall back to the other format.
    This means the caller never needs to know which format the file is in.

    Parameters
    ----------
    filepath     : full path to the file
    usecols      : list of column names to load (None = all)
    nrows        : maximum rows to read (None = all)
    sep          : CSV delimiter (ignored for Excel files)
    dtype        : dtype dict or single dtype
    on_bad_lines : {"skip","error","warn"} – CSV bad-line policy
    sheet_name   : Excel sheet index or name (default 0 = first sheet)
    """
    ext = Path(filepath).suffix.lower()
    is_excel_ext = ext in (".xlsx", ".xls", ".xlsm", ".xlsb")

    def _excel_kw():
        kw: dict = dict(sheet_name=sheet_name)
        if usecols  is not None: kw["usecols"]  = usecols
        if nrows    is not None: kw["nrows"]     = nrows
        if dtype    is not None: kw["dtype"]     = dtype
        return kw

    def _csv_kw():
        kw: dict = dict(sep=sep, on_bad_lines=on_bad_lines)
        if usecols  is not None: kw["usecols"]  = usecols
        if nrows    is not None: kw["nrows"]     = nrows
        if dtype    is not None: kw["dtype"]     = dtype
        return kw

    if is_excel_ext:
        # Try Excel first; fall back to CSV if the file is actually plain-text
        try:
            return pd.read_excel(filepath, engine="openpyxl", **_excel_kw())
        except Exception:
            try:
                return pd.read_excel(filepath, engine="xlrd", **_excel_kw())
            except Exception:
                return pd.read_csv(filepath, **_csv_kw())
    else:
        # Try CSV first; fall back to Excel if the file is actually a workbook
        try:
            return pd.read_csv(filepath, **_csv_kw())
        except Exception:
            return pd.read_excel(filepath, engine="openpyxl", **_excel_kw())


# ════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _normalise(s: str) -> str:
    """Lowercase + strip + collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _pairs_to_triples(
    drug_adr_pairs: List[Tuple[str, str]],
    max_triples: int = 50_000,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, str, str]]:
    """
    Synthesise polypharmacy triples from single-drug pairs via shared-ADR inference:
      If d1 -> a  AND  d2 -> a  (d1 ≠ d2)  →  candidate triple (d1, d2, a)
    Returns up to max_triples shuffled triples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    from collections import defaultdict
    adr_to_drugs: Dict[str, List[str]] = defaultdict(list)
    for drug, adr in drug_adr_pairs:
        adr_to_drugs[adr].append(drug)

    triples: List[Tuple[str, str, str]] = []
    for adr, drugs in adr_to_drugs.items():
        drugs = list(set(drugs))
        if len(drugs) < 2:
            continue
        # Sample pairs to avoid combinatorial explosion
        pairs = [(drugs[i], drugs[j])
                 for i in range(len(drugs))
                 for j in range(i + 1, len(drugs))]
        if len(pairs) > 500:
            idx = rng.choice(len(pairs), 500, replace=False)
            pairs = [pairs[k] for k in idx]
        for d1, d2 in pairs:
            triples.append((d1, d2, adr))

    rng.shuffle(triples)
    return triples[:max_triples]


def _split_train_test(
    dataset: DrugADRDataset,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DrugADRDataset, DrugADRDataset]:
    """
    Edge-level train / test split on single-drug pairs.
    Returns (train_dataset, test_dataset).
    """
    from sklearn.model_selection import train_test_split
    pairs = dataset.drug_adr_pairs
    idx   = list(range(len(pairs)))
    tr_idx, te_idx = train_test_split(idx, test_size=test_ratio,
                                       random_state=seed)
    train_pairs = [pairs[i] for i in tr_idx]
    test_pairs  = [pairs[i] for i in te_idx]

    train_ds = DrugADRDataset(train_pairs)
    test_ds  = DrugADRDataset(test_pairs)
    return train_ds, test_ds


# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────
# SECTION A :  YOUR OWN DATASETS  (DS1 – DS3)
# ──────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def load_ds1_drug_adr_relation(
    filepath: str = "Dataset 1_Drug ADR Relation.csv",
    infer_triples: bool = True,
    max_triples: int = 30_000,
) -> DrugADRDataset:
    """
    Dataset 1 – Drug ADR Relation
    File    : Dataset 1_Drug ADR Relation.csv
    Columns : drug | adr
    Size    : 5 730 rows, 1 319 drugs, 3 341 ADRs
    Use     : Single-drug bipartite graph; base graph for Model 2.
    """
    df = _read_file(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["drug", "adr"])
    df["drug"] = df["drug"].apply(_normalise)
    df["adr"]  = df["adr"].apply(_normalise)

    pairs = list(zip(df["drug"], df["adr"]))
    triples = _pairs_to_triples(pairs, max_triples) if infer_triples else []
    ds = DrugADRDataset(pairs, triples)
    print(f"[DS1] Loaded: {ds.n_drugs} drugs | {ds.n_adrs} ADRs | "
          f"{len(pairs)} edges | {len(triples)} poly-triples")
    return ds


def load_ds2_sider(
    filepath: str = "Dataset 2_SIDER 4.1_drug_adr_associations.csv",
    infer_triples: bool = True,
    max_triples: int = 50_000,
) -> DrugADRDataset:
    """
    Dataset 2 – SIDER 4.1 Drug-ADR Associations
    File    : Dataset 2_SIDER 4.1_drug_adr_associations.csv
    Columns : drug | adr
    Size    : 158 559 rows, 1 347 drugs, 6 123 ADRs
    Use     : Primary single-drug benchmark graph (largest of your datasets).
    """
    df = _read_file(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["drug", "adr"])
    df["drug"] = df["drug"].apply(_normalise)
    df["adr"]  = df["adr"].apply(_normalise)
    df = df.drop_duplicates()

    pairs = list(zip(df["drug"], df["adr"]))
    triples = _pairs_to_triples(pairs, max_triples) if infer_triples else []
    ds = DrugADRDataset(pairs, triples)
    print(f"[DS2-SIDER] Loaded: {ds.n_drugs} drugs | {ds.n_adrs} ADRs | "
          f"{len(pairs)} edges | {len(triples)} poly-triples")
    return ds


def load_ds3_neonatal(
    filepath: str = "Dataset 3_Neonatal_Drug_ADR_Expanded.csv",
    use_multidrug: bool = True,
    separator: str = "\\",
    min_combo_size: int = 2,
    max_combo_size: int = 5,
    max_triples: int = 50_000,
) -> DrugADRDataset:
    """
    Dataset 3 – Neonatal Drug-ADR Expanded (FAERS-derived)
    File    : Dataset 3_Neonatal_Drug_ADR_Expanded.csv
    Columns : drug | adr
    Size    : 203 616 rows, 13 598 unique drug strings, 4 993 ADRs
    Multi-drug entries use '\\' as separator (73 879 rows are multi-drug combos).

    Parameters
    ----------
    use_multidrug : bool
        Parse backslash-separated combos as real polypharmacy triples.
    min/max_combo_size : int
        Only include combinations within this drug-count range.
    """
    df = _read_file(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["drug", "adr"])
    df["adr"] = df["adr"].apply(_normalise)

    is_multi  = df["drug"].str.contains(re.escape(separator), na=False)
    single_df = df[~is_multi].copy()
    multi_df  = df[is_multi].copy()

    single_df["drug"] = single_df["drug"].apply(_normalise)
    pairs = list(zip(single_df["drug"], single_df["adr"]))

    triples: List[Tuple[str, str, str]] = []
    if use_multidrug:
        for _, row in multi_df.iterrows():
            drugs = [_normalise(d) for d in row["drug"].split(separator) if d.strip()]
            if not (min_combo_size <= len(drugs) <= max_combo_size):
                continue
            adr = row["adr"]
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    triples.append((drugs[i], drugs[j], adr))
                pairs.append((drugs[i], adr))

        rng = np.random.default_rng(42)
        rng.shuffle(triples)
        triples = triples[:max_triples]

    pairs = list(set(pairs))
    ds = DrugADRDataset(pairs, triples)
    print(f"[DS3-Neonatal] Loaded: {ds.n_drugs} drugs | {ds.n_adrs} ADRs | "
          f"{len(pairs)} edges | {len(triples)} poly-triples")
    return ds


# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────
# SECTION B :  PUBLIC BENCHMARK DATASETS
# ──────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# ─── OFFSIDES ───────────────────────────────────────────────────────────────
# Download: http://tatonettilab.org/offsides/  (OFFSIDES.csv.gz, ~800 MB)
#           or via PharmGKB / Zenodo mirrors
# File     : 3003.tsv  (stitch_id | drug_name | condition_meddra_id | condition_name
#                        | A | B | prr | prr_error | mean_reporting_freq)
# Columns that matter: drug_name, condition_name, prr (proportional reporting ratio)

def load_offsides(
    filepath: str,
    prr_threshold: float = 1.0,
    max_rows: int = 500_000,
    infer_triples: bool = True,
    max_triples: int = 50_000,
) -> DrugADRDataset:
    """
    OFFSIDES – Off-label single-drug side effects
    Source : http://tatonettilab.org/offsides/
    Format : TSV, columns include drug_name and condition_name (+ PRR score)

    Parameters
    ----------
    prr_threshold : float
        Minimum Proportional Reporting Ratio to include an association.
        PRR > 2 is a common pharmacovigilance signal threshold.
    """
    # Try both known column layouts
    try:
        df = _read_file(filepath, sep="\t", nrows=max_rows,
                        usecols=["drug_name", "condition_name", "prr"])
        df = df[df["prr"] >= prr_threshold]
        drug_col, adr_col = "drug_name", "condition_name"
    except (ValueError, KeyError):
        df = _read_file(filepath, sep="\t", nrows=max_rows)
        cols = df.columns.str.lower().tolist()
        drug_col = next(c for c in cols if "drug" in c)
        adr_col  = next(c for c in cols if "condition" in c or "side" in c or "adr" in c)
        df.columns = cols

    df[drug_col] = df[drug_col].apply(_normalise)
    df[adr_col]  = df[adr_col].apply(_normalise)
    df = df.dropna(subset=[drug_col, adr_col]).drop_duplicates([drug_col, adr_col])

    pairs = list(zip(df[drug_col], df[adr_col]))
    triples = _pairs_to_triples(pairs, max_triples) if infer_triples else []
    ds = DrugADRDataset(pairs, triples)
    print(f"[OFFSIDES] Loaded: {ds}")
    return ds


# ─── TWOSIDES ────────────────────────────────────────────────────────────────
# Download : https://doi.org/10.6084/m9.figshare.22495666.v1
# File     : twosides.csv
#
# Confirmed column schema (from dataset screenshot):
#   drug_1_rxnorm_id   | drug_1_concept_name  | drug_2_rxnorm_id
#   drug_2_concept_name| condition_meddra_id  | condition_concept_name
#   A | B | C | D      | PRR | PRR_error | mean_reporting_frequency
#
# Columns used:
#   drug_1_concept_name   → Drug 1 name
#   drug_2_concept_name   → Drug 2 name
#   condition_concept_name → ADR / side-effect name
#   PRR                    → Proportional Reporting Ratio (signal strength filter)

def load_twosides(
    filepath: str,
    prr_threshold: float = 1.0,
    max_rows: int = 200_000,
    build_single_edges: bool = True,
) -> DrugADRDataset:
    """
    TWOSIDES – Drug-pair polypharmacy side effects.
    Source : https://doi.org/10.6084/m9.figshare.22495666.v1
    Size   : ~645 drugs, ~63 473 unique pairs, ~1 318 ADRs, ~4.65 M triples

    Column schema (confirmed from dataset):
      drug_1_concept_name, drug_2_concept_name, condition_concept_name, PRR, …

    Parameters
    ----------
    prr_threshold : float
        Keep only associations where PRR >= threshold.
        PRR > 1.0 means the combination is more likely to cause the ADR
        than either drug alone.  Use 2.0 for a stricter signal threshold.
    build_single_edges : bool
        Derive individual drug→ADR edges from each triple and add them to
        the bipartite base graph. Required for kernel matrix computation.
    """
    # ── 1. Load with confirmed column names ─────────────────────────────────
    df = _read_file(filepath, nrows=max_rows)
    df.columns = df.columns.str.strip()          # preserve original case first

    # Column name map: handle slight naming variations across file versions
    _rename = {
        # confirmed schema (screenshot)
        "drug_1_concept_name":    "d1",
        "drug_2_concept_name":    "d2",
        "condition_concept_name": "adr",
        "PRR":                    "prr",
        # older / alternative layouts
        "drug1_name":             "d1",
        "drug2_name":             "d2",
        "condition_name":         "adr",
        "side_effect_name":       "adr",
    }
    df = df.rename(columns={k: v for k, v in _rename.items() if k in df.columns})

    # Verify required columns exist after renaming
    missing = [c for c in ("d1", "d2", "adr") if c not in df.columns]
    if missing:
        raise ValueError(
            f"[TWOSIDES] Could not locate drug/ADR columns. "
            f"Available columns: {df.columns.tolist()}"
        )

    # ── 2. PRR filter ────────────────────────────────────────────────────────
    if "prr" in df.columns and prr_threshold > 0:
        df["prr"] = pd.to_numeric(df["prr"], errors="coerce")
        df = df[df["prr"] >= prr_threshold]

    # ── 3. Normalise & drop nulls ────────────────────────────────────────────
    df["d1"]  = df["d1"].apply(_normalise)
    df["d2"]  = df["d2"].apply(_normalise)
    df["adr"] = df["adr"].apply(_normalise)
    df = df.dropna(subset=["d1", "d2", "adr"])
    df = df[df["d1"].str.len() > 0]
    df = df[df["d2"].str.len() > 0]
    df = df[df["adr"].str.len() > 0]

    # ── 4. Build triples and single-drug pairs ───────────────────────────────
    triples = list(set(zip(df["d1"], df["d2"], df["adr"])))

    pairs: List[Tuple[str, str]] = []
    if build_single_edges:
        pairs = list(set(
            [(d1, a) for d1, _,  a in triples] +
            [(d2, a) for _,  d2, a in triples]
        ))

    ds = DrugADRDataset(pairs, triples)
    n_drugs = len({d for d, _ in pairs}) if pairs else 0
    n_adrs  = len({a for _, a in pairs}) if pairs else 0
    print(f"[TWOSIDES] Loaded: {len(triples):,} triples | "
          f"{n_drugs:,} drugs | {n_adrs:,} ADRs | PRR≥{prr_threshold}")
    return ds


# ─── SNAP Monopharmacy ───────────────────────────────────────────────────────
# Download: http://snap.stanford.edu/biodata/datasets/10013/10013-ChSe-Decagon_monopharmacy.html
# Files:
#   bio-decagon-mono.csv  → stitch_id, side_effect_id, side_effect_name
#   bio-decagon-targets.csv → stitch_id, gene_id
# The side_effect column maps stitch_id (drug) → Decagon side-effect code

def load_snap_mono(
    mono_filepath: str,
    combo_filepath: Optional[str] = None,
    infer_triples: bool = True,
    max_triples: int = 50_000,
) -> DrugADRDataset:
    """
    SNAP Decagon Monopharmacy side effects
    Source : http://snap.stanford.edu/biodata/datasets/10013/
    File   : bio-decagon-mono.csv
    Columns: Individual Stitch ID, Individual Side Effect Name (+ UMLS ID)

    Parameters
    ----------
    combo_filepath : str, optional
        Path to bio-decagon-combo.csv for polypharmacy triples.
    """
    df = _read_file(mono_filepath)
    df.columns = df.columns.str.strip()

    # Two known header layouts
    if "Individual Stitch ID" in df.columns:
        drug_col = "Individual Stitch ID"
        adr_col  = "Individual Side Effect Name"
    elif "STITCH 1" in df.columns:
        drug_col, adr_col = "STITCH 1", "Side Effect Name"
    else:
        cols = df.columns.tolist()
        drug_col, adr_col = cols[0], cols[-1]

    df[drug_col] = df[drug_col].apply(_normalise)
    df[adr_col]  = df[adr_col].apply(_normalise)
    pairs = list(set(zip(df[drug_col], df[adr_col])))

    # Optionally load polypharmacy combos
    triples: List[Tuple[str, str, str]] = []
    if combo_filepath:
        triples = _load_snap_combo_triples(combo_filepath)
    elif infer_triples:
        triples = _pairs_to_triples(pairs, max_triples)

    ds = DrugADRDataset(pairs, triples)
    print(f"[SNAP-Mono] Loaded: {ds}")
    return ds


def _load_snap_combo_triples(combo_filepath: str) -> List[Tuple[str, str, str]]:
    """
    Internal helper: load SNAP Decagon combo (polypharmacy) file.
    File: bio-decagon-combo.csv
    Columns: STITCH 1 | STITCH 2 | Polypharmacy Side Effect | Side Effect Name
    """
    df = _read_file(combo_filepath)
    df.columns = df.columns.str.strip()

    if "STITCH 1" in df.columns:
        d1_col, d2_col = "STITCH 1", "STITCH 2"
        adr_col = "Side Effect Name"
    else:
        cols = df.columns.tolist()
        d1_col, d2_col, adr_col = cols[0], cols[1], cols[-1]

    df[d1_col] = df[d1_col].apply(_normalise)
    df[d2_col] = df[d2_col].apply(_normalise)
    df[adr_col] = df[adr_col].apply(_normalise)
    return list(zip(df[d1_col], df[d2_col], df[adr_col]))


def load_snap_poly(
    combo_filepath: str,
    mono_filepath: Optional[str] = None,
) -> DrugADRDataset:
    """
    SNAP Decagon Polypharmacy (combo only, or combo + mono).
    Source : http://snap.stanford.edu/biodata/datasets/10013/
    File   : bio-decagon-combo.csv
    Columns: STITCH 1 | STITCH 2 | Polypharmacy Side Effect | Side Effect Name
    Size   : ~645 drugs, ~63 473 pairs, ~1 317 side effects, ~4.65 M rows

    This is effectively the same source as TWOSIDES but distributed via SNAP.
    """
    triples = _load_snap_combo_triples(combo_filepath)

    pairs: List[Tuple[str, str]] = []
    if mono_filepath:
        ds_mono = load_snap_mono(mono_filepath, infer_triples=False)
        pairs   = ds_mono.drug_adr_pairs
    else:
        # Derive pairs from combo: each drug in any combo → its side effects
        pairs = list(set(
            [(d1, a) for d1, _, a in triples] +
            [(d2, a) for _, d2, a in triples]
        ))

    ds = DrugADRDataset(pairs, triples)
    print(f"[SNAP-Poly] Loaded: {ds}")
    return ds


# ─── WebMD Drug Reviews ──────────────────────────────────────────────────────
# Download : https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset
# File     : webmd.csv  (also available as Dataset_4_WebMD.csv in this project)
#
# Confirmed column schema (from dataset screenshot):
#   Age | Condition | Date | Drug | DrugId | EaseofUse | Effectiveness
#   Reviews | Satisfaction | Sex | Sides | UsefulCount
#
# Columns used:
#   Drug   → drug name  (must be non-empty)
#   Sides  → free-text, comma-separated patient-reported side effects
#             (must be non-empty)
#
# Extraction strategy:
#   Split each Sides entry on commas and semicolons → individual ADR tokens.
#   Each unique (Drug, adr_token) pair becomes one bipartite edge.
#   Single-word tokens with length ≥ 3 chars are kept (e.g. "dizziness").
#   Tokens that are purely numeric or < 3 chars are discarded as noise.

def load_webmd(
    filepath: str,
    drug_col: str = "Drug",
    sides_col: str = "Sides",
    min_chars: int = 3,
    max_tokens_per_entry: int = 15,
    max_rows: Optional[int] = None,
    infer_triples: bool = True,
    max_triples: int = 30_000,
) -> DrugADRDataset:
    """
    WebMD Drug Reviews Dataset – Drug + Sides (ADR) columns only.
    Source  : https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset

    Confirmed schema
    ----------------
    Age | Condition | Date | Drug | DrugId | EaseofUse | Effectiveness
    Reviews | Satisfaction | Sex | Sides | UsefulCount

    Only the ``Drug`` and ``Sides`` columns are used; all other columns
    are ignored.  Rows where either column is empty / NaN are dropped.

    Parameters
    ----------
    drug_col : str
        Column name for the drug name (default ``"Drug"``).
    sides_col : str
        Column name for the side-effect text (default ``"Sides"``).
    min_chars : int
        Minimum character length for an individual ADR token to be kept.
        Filters out very short noise tokens (default 3).
    max_tokens_per_entry : int
        Maximum number of ADR tokens extracted per row. Caps unusually
        long comma-lists that are likely data-entry artefacts (default 15).
    max_rows : int or None
        If set, read only this many rows from the CSV (useful for testing).
    infer_triples : bool
        Build polypharmacy triples via shared-ADR heuristic.

    Notes
    -----
    • Patient-reported – noisy, not clinically verified.
    • Treat as a complementary evidence source alongside SIDER / TWOSIDES.
    """
    # ── 1. Load only the two required columns ────────────────────────────────
    # Try loading with usecols first (faster); fall back to full load if the
    # column names differ slightly (e.g. extra whitespace in Excel headers).
    try:
        df = _read_file(filepath, usecols=[drug_col, sides_col],
                        nrows=max_rows, dtype=str)
    except (ValueError, KeyError):
        df = _read_file(filepath, nrows=max_rows, dtype=str)
        # Case-insensitive column match
        col_map = {c.lower(): c for c in df.columns}
        drug_col  = col_map.get(drug_col.lower(),  drug_col)
        sides_col = col_map.get(sides_col.lower(), sides_col)

    # ── 2. Keep only Drug + Sides; drop rows where either is empty ───────────
    df = df[[drug_col, sides_col]].copy()
    df = df.dropna(subset=[drug_col, sides_col])
    df = df[df[drug_col].str.strip().str.len() > 0]
    df = df[df[sides_col].str.strip().str.len() > 0]

    # ── 3. Normalise drug names ──────────────────────────────────────────────
    df[drug_col] = df[drug_col].apply(_normalise)

    # ── 4. Parse Sides: split on commas / semicolons / newlines ─────────────
    _split_re = re.compile(r"[,;\n]+")

    pairs: List[Tuple[str, str]] = []
    for drug, sides in zip(df[drug_col], df[sides_col]):
        tokens = _split_re.split(str(sides))
        n = 0
        for tok in tokens:
            tok = _normalise(tok)
            # Keep tokens that are ≥ min_chars and not purely numeric
            if len(tok) >= min_chars and not tok.replace(" ", "").isdigit():
                pairs.append((drug, tok))
                n += 1
                if n >= max_tokens_per_entry:
                    break

    pairs = list(set(pairs))
    triples = _pairs_to_triples(pairs, max_triples) if infer_triples else []
    ds = DrugADRDataset(pairs, triples)
    print(f"[WebMD] Loaded: {len(pairs):,} (drug, ADR) pairs from "
          f"{len(df):,} non-empty rows | {ds.n_drugs:,} drugs | "
          f"{ds.n_adrs:,} unique ADR tokens")
    return ds


# ════════════════════════════════════════════════════════════════════════════
# SECTION C :  COMBINED / MERGED DATASET
# ════════════════════════════════════════════════════════════════════════════

def merge_datasets(*datasets: DrugADRDataset) -> DrugADRDataset:
    """
    Merge multiple DrugADRDataset instances into one unified dataset.
    Duplicates in pairs and triples are automatically removed.
    """
    all_pairs:   Set[Tuple[str, str]]          = set()
    all_triples: Set[Tuple[str, str, str]]     = set()

    for ds in datasets:
        all_pairs.update(ds.drug_adr_pairs)
        all_triples.update(ds.drug_pair_adr_triples)

    merged = DrugADRDataset(list(all_pairs), list(all_triples))
    print(f"[MERGED] {len(datasets)} datasets → {merged}")
    return merged


# ════════════════════════════════════════════════════════════════════════════
# SECTION D :  DATASET SUMMARY & STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def dataset_summary(ds: DrugADRDataset, name: str = "") -> pd.DataFrame:
    """Print a compact statistics summary of a DrugADRDataset."""
    import scipy.sparse as sp

    n_edges = len(ds.drug_adr_pairs)
    n_drugs = ds.n_drugs
    n_adrs  = ds.n_adrs
    density = n_edges / (n_drugs * n_adrs + 1e-9)
    n_poly  = len(ds.drug_pair_adr_triples)

    # Degree distribution (from B matrix)
    try:
        B = np.zeros((n_drugs, n_adrs))
        for d, a in ds.drug_adr_pairs:
            if d in ds.drug2idx and a in ds.adr2idx:
                B[ds.drug2idx[d], ds.adr2idx[a]] = 1
        drug_deg = B.sum(axis=1)
        adr_deg  = B.sum(axis=0)
        stats = {
            "Dataset":              name,
            "# Drugs":              n_drugs,
            "# ADRs":               n_adrs,
            "# Drug-ADR edges":     n_edges,
            "Graph density":        f"{density:.4f}",
            "Avg drug degree":      f"{drug_deg.mean():.1f}",
            "Max drug degree":      f"{drug_deg.max():.0f}",
            "Avg ADR degree":       f"{adr_deg.mean():.1f}",
            "Max ADR degree":       f"{adr_deg.max():.0f}",
            "# Poly triples":       n_poly,
        }
    except Exception:
        stats = {"Dataset": name, "# Drugs": n_drugs,
                 "# ADRs": n_adrs, "# Edges": n_edges,
                 "# Poly triples": n_poly}

    df = pd.DataFrame([stats]).T
    df.columns = ["Value"]
    print(f"\n{'─'*45}")
    print(f"  Statistics: {name}")
    print(f"{'─'*45}")
    print(df.to_string())
    print(f"{'─'*45}\n")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION E :  READY-MADE EXPERIMENT CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    # ── Single-drug experiments (for Model 2 unsupervised link prediction) ──
    "exp1_single_drug_small": {
        "description": "DS1 alone – fast iteration / unit testing",
        "loader":      "load_ds1_drug_adr_relation",
        "args":        {"infer_triples": True, "max_triples": 10_000},
    },
    "exp2_single_drug_sider": {
        "description": "SIDER 4.1 – standard medium-scale single-drug benchmark",
        "loader":      "load_ds2_sider",
        "args":        {"infer_triples": True, "max_triples": 50_000},
    },
    "exp3_single_drug_offsides": {
        "description": "OFFSIDES – large-scale off-label single-drug benchmark",
        "loader":      "load_offsides",
        "args":        {"prr_threshold": 2.0, "max_rows": 300_000,
                        "infer_triples": True, "max_triples": 50_000},
    },
    # ── Polypharmacy experiments (for Model 1 Kernel SVM) ──
    "exp4_poly_neonatal": {
        "description": "DS3 Neonatal – domain-specific polypharmacy",
        "loader":      "load_ds3_neonatal",
        "args":        {"use_multidrug": True, "max_combo_size": 3,
                        "max_triples": 30_000},
    },
    "exp5_poly_twosides": {
        "description": "TWOSIDES – gold-standard polypharmacy benchmark",
        "loader":      "load_twosides",
        "args":        {"max_rows": 200_000, "prr_threshold": 1.0},
    },
    "exp6_poly_snap": {
        "description": "SNAP Decagon Poly – alternative polypharmacy source",
        "loader":      "load_snap_poly",
        "args":        {},                        # provide paths at runtime
    },
    "exp8_single_webmd": {
        "description": "WebMD Drug Reviews – Drug + Sides columns only",
        "loader":      "load_webmd",
        "args":        {"min_chars": 3, "infer_triples": True,
                        "max_triples": 30_000},
    },
    # ── Merged / multi-source ──
    "exp7_merged_all_yours": {
        "description": "DS1 + DS2 + DS3 merged",
        "loader":      "merge",
        "sources":     ["exp1_single_drug_small",
                        "exp2_single_drug_sider",
                        "exp4_poly_neonatal"],
    },
    "exp9_merged_with_webmd": {
        "description": "DS1 + DS2 + DS3 + WebMD merged",
        "loader":      "merge",
        "sources":     ["exp1_single_drug_small",
                        "exp2_single_drug_sider",
                        "exp4_poly_neonatal",
                        "exp8_single_webmd"],
    },
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION F :  QUICK-START  run_experiment()
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    config_name: str,
    data_dir: str = "./Dataset",
    **path_overrides,
) -> DrugADRDataset:
    """
    Load a named experiment configuration.

    Parameters
    ----------
    config_name : str
        One of the keys in EXPERIMENT_CONFIGS.
    data_dir : str
        Directory where data files are located.
        Defaults to  ./Dataset  – the subfolder name used in this project.
    path_overrides : dict
        Pass filepath=... to override the default filename for any loader.

    Examples
    --------
    # Using DS1 from ./Dataset/
    ds = run_experiment("exp1_single_drug_small")

    # Using DS3 from a custom path
    ds = run_experiment("exp4_poly_neonatal",
                        data_dir="/home/user/mydata")

    # Using TWOSIDES from a specific file
    ds = run_experiment("exp5_poly_twosides",
                        filepath="/data/twosides.csv")
    """
    cfg = EXPERIMENT_CONFIGS[config_name]
    print(f"\n[run_experiment] {config_name}: {cfg['description']}")

    if cfg["loader"] == "merge":
        sub_datasets = [
            run_experiment(src, data_dir=data_dir, **path_overrides)
            for src in cfg["sources"]
        ]
        return merge_datasets(*sub_datasets)

    loader_fn = {
        "load_ds1_drug_adr_relation":  load_ds1_drug_adr_relation,
        "load_ds2_sider":              load_ds2_sider,
        "load_ds3_neonatal":           load_ds3_neonatal,
        "load_offsides":               load_offsides,
        "load_twosides":               load_twosides,
        "load_snap_mono":              load_snap_mono,
        "load_snap_poly":              load_snap_poly,
        "load_webmd":                  load_webmd,
    }[cfg["loader"]]

    args = dict(cfg.get("args", {}))

    # Default filenames – exact names as they appear in the Dataset folder (all .csv)
    default_files = {
        "load_ds1_drug_adr_relation": "Dataset 1_Drug ADR Relation.csv",
        "load_ds2_sider":             "Dataset 2_SIDER 4.1_drug_adr_associations.csv",
        "load_ds3_neonatal":          "Dataset 3_Neonatal_Drug_ADR_Expanded.csv",
        "load_offsides":              "OFFSIDES.csv",
        "load_twosides":              "TWOSIDES.csv",
        "load_snap_mono":             "bio-decagon-mono.csv",
        "load_snap_poly":             "bio-decagon-combo.csv",
        "load_webmd":                 "Dataset 4_WebMD.csv",
    }

    if "filepath" not in args and "filepath" not in path_overrides:
        fname = default_files.get(cfg["loader"], "")
        if fname:
            args["filepath"] = str(Path(data_dir) / fname)

    args.update(path_overrides)
    return loader_fn(**args)


# ════════════════════════════════════════════════════════════════════════════
# SECTION G :  ENTRY POINT  –  demo run on your four datasets
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Set this to the folder that contains your .xlsx files ────────────────
    DATA_DIR = "./Dataset"
    # If you placed files directly next to this script, use DATA_DIR = "."

    print("\n" + "="*65)
    print("  LOADING YOUR THREE DATASETS  (auto-detects .xlsx / .csv)")
    print("="*65)

    ds1 = load_ds1_drug_adr_relation(
        filepath=f"{DATA_DIR}/Dataset 1_Drug ADR Relation.csv")

    ds2 = load_ds2_sider(
        filepath=f"{DATA_DIR}/Dataset 2_SIDER 4.1_drug_adr_associations.csv",
        max_triples=30_000)

    ds3 = load_ds3_neonatal(
        filepath=f"{DATA_DIR}/Dataset 3_Neonatal_Drug_ADR_Expanded.csv",
        use_multidrug=True,
        max_combo_size=3,
        max_triples=30_000)

    for ds, name in [(ds1, "DS1  Drug-ADR Relation"),
                     (ds2, "DS2  SIDER 4.1"),
                     (ds3, "DS3  Neonatal FAERS")]:
        dataset_summary(ds, name)

    print("\n" + "="*65)
    print("  MERGING DS1 + DS2 + DS3")
    print("="*65)
    merged = merge_datasets(ds1, ds2, ds3)
    dataset_summary(merged, "DS1 + DS2 + DS3  (merged)")

    print("\n" + "="*65)
    print("  LOADING WebMD (Drug + Sides columns only)")
    print("="*65)
    ds_web = load_webmd(
        filepath=f"{DATA_DIR}/Dataset 4_WebMD.csv")
    dataset_summary(ds_web, "DS4  WebMD")

    print("\n" + "="*65)
    print("  LOADING TWOSIDES (polypharmacy benchmark)")
    print("="*65)
    ds_two = load_twosides(
        filepath=f"{DATA_DIR}/TWOSIDES.csv",
        prr_threshold=1.0)
    dataset_summary(ds_two, "TWOSIDES")

    print("\n" + "="*65)
    print("  EXTERNAL DATASETS  (download separately)")
    print("="*65)
    print(f"""
  ┌─ OFFSIDES (single-drug, off-label) ──────────────────────────────────┐
  │  ds_off = load_offsides("{DATA_DIR}/OFFSIDES.csv", prr_threshold=2.0) │
  └───────────────────────────────────────────────────────────────────────┘
  Download: http://tatonettilab.org/offsides/

  ┌─ SNAP Decagon Monopharmacy ───────────────────────────────────────────┐
  │  ds_mono = load_snap_mono("{DATA_DIR}/bio-decagon-mono.csv")           │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─ SNAP Decagon Polypharmacy ───────────────────────────────────────────┐
  │  ds_poly = load_snap_poly(                                            │
  │      combo_filepath="{DATA_DIR}/bio-decagon-combo.csv",               │
  │      mono_filepath ="{DATA_DIR}/bio-decagon-mono.csv")                │
  └───────────────────────────────────────────────────────────────────────┘
  Download: http://snap.stanford.edu/biodata/datasets/10013/

  ── Run the kernel pipeline on any dataset ──────────────────────────────
  from graph_kernel_adr_prediction import run_full_pipeline
  run_full_pipeline(dataset=ds_two)   # or ds3, ds_web, merged …
    """)

    print("\n" + "="*65)
    print("  RUNNING KERNEL PIPELINE ON DS3 (Neonatal)")
    print("="*65)
    try:
        from graph_kernel_adr_prediction import run_full_pipeline
        run_full_pipeline(dataset=ds3)
    except ImportError:
        print("  graph_kernel_adr_prediction.py not found in Python path.")
        print("  Ensure both .py files are in the same folder, then run again.")

