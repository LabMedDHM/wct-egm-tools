"""
egm_common.py
=============
Shared core for the WCT EVI MAP figure and table generator. It turns one
merged Blue Book JSON (the output of 02_merge_json_files.py after LoE coding)
into tidy data, and holds every styling and ordering decision so that all
figures and tables come out identical across thedifferent Blue Books.

This module produces no final files. It is imported, not run, which is why it
carries no numeric filename prefix.

Data model and counting units
------------------------------
The merged JSON stores one row per *raw entry* in the References list. A raw
entry is one coded combination of (one characteristic, one study design, one
level of evidence, and one or more tumour-type leaves). One physical study
(one OldItemId) can produce several raw entries.

entries_df expands every raw entry to one row per tumour-type leaf it carries,
and keeps an `entry_id` so the three counting units used in the publications
can all be recovered from the same frame:

  - per-leaf record  : use entries_df directly. One row per (raw entry, leaf).
                       Drives Figure 7, Figure S2, Table S1, Table S2 and the
                       entries-per-tumour-type counts.
  - raw entry        : entries_df.drop_duplicates("entry_id"). One row per raw
                       entry. Drives Figure 5, Figure S1, the study-design
                       figure and the "Entries in the EGM" count.
  - unique study     : dedupe on `study_id` (OldItemId). Drives Figure 3 and
                       the study-level rows of Table 1.

For per-group (per source map) views, an entry is counted once per source map
it touches, i.e. drop_duplicates(["entry_id", "source_map"]). In every Blue
Book here each entry sits in exactly one source map, so this only matters as a
safeguard.

Source maps versus codeset tumour groups
----------------------------------------
The split unit for per-group outputs is the *source map*, the search and map
that produced the records, not the codeset tumour group (the immediate child
of the Blue Book root). For Endo and CNS the source map equals the codeset
group, except that the CNS "Embryonal tumours" group is the Medulloblastoma
map. For Haema the three AML codeset groups all belong to the single AML map.
SOURCE_MAP_OF_GROUP below encodes the only departures from "source map equals
codeset group"; everything else falls through to the group name.

British English and verbatim WHO nomenclature
---------------------------------------------
Tumour type and tumour group names are written verbatim as they appear in the
data, treated as WHO Classification of Tumours nomenclature (proper nouns with
an external source authority) and covered by a single methods footnote. Study
design labels inherited from the HETP hierarchy may be anglicised at the
display layer only, never by editing the data; STUDY_DESIGN_DISPLAY holds the
one required change, with the American spelling as the key because that is the
string that can appear in the source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before pyplot is imported
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Ordering and display constants
# ---------------------------------------------------------------------------

# The six framework characteristics in fixed display order, following the
# clinical care pathway (pathogenesis -> diagnosis -> outcome). The keys are
# the exact AttributeName strings as they appear in the codeset, so matching is
# exact and needs no fuzzy canonicalisation.
CHARACTERISTIC_ORDER: List[str] = [
    "Pathogenesis",
    "Diagnostic imaging",
    "Histopathology (incl. cytopathology)",
    "Diagnostic molecular pathology",
    "Prognosis",
    "Prediction",
]

# Optional shorter axis labels. Only Histopathology is shortened, to keep the
# heatmap and bar axes readable; everything else renders verbatim. Set this to
# an empty dict to print the full codeset names everywhere.
CHARACTERISTIC_DISPLAY: Dict[str, str] = {
    "Histopathology (incl. cytopathology)": "Histopathology",
}

# The six Level-of-Evidence bands, best to worst, with the Unclassifiable
# bucket last. The merged JSON already carries the HETP P-wording, so these are
# both the canonical keys and the display labels.
LOE_ORDER: List[str] = [
    "Level P1",
    "Level P2",
    "Level P3",
    "Level P4",
    "Level P5",
    "Unclassifiable",
]

HIGH_LOE = {"Level P1", "Level P2"}   # high-level evidence
LOW_LOE = {"Level P4", "Level P5"}    # low-level evidence

# LoE colours, viridis best-to-worst plus a neutral grey for Unclassifiable.
LOE_COLOURS: Dict[str, str] = {
    "Level P1":       "#FDE725",
    "Level P2":       "#5EC962",
    "Level P3":       "#21918C",
    "Level P4":       "#3B528B",
    "Level P5":       "#440154",
    "Unclassifiable": "#999999",
}

# Gap categories, worst to best, mapped onto four viridis stops.
GAP_CATEGORIES = ["Absolute gap", "Relative gap", "Synthesis gap", "Solid evidence"]
GAP_PALETTE: Dict[str, str] = {
    "Absolute gap":   "#440154",  # dark purple, viridis 0.00
    "Relative gap":   "#3A528B",  # blue,        viridis 0.25
    "Synthesis gap":  "#20908C",  # teal,        viridis 0.50
    "Solid evidence": "#5EC961",  # green,       viridis 0.75
}
GAP_TEXT_COLOUR: Dict[str, str] = {
    "Absolute gap":   "#FFFFFF",
    "Relative gap":   "#FFFFFF",
    "Synthesis gap":  "#FFFFFF",
    "Solid evidence": "#000000",
}
RELATIVE_GAP_MAX_N = 20  # < 20 records and no high LoE = relative, else synthesis

# Display-only anglicisation of study design labels. American key because that
# is the string that can appear in the data; the merged files used here already
# carry the British spelling, so this acts as a harmless safeguard.
STUDY_DESIGN_DISPLAY: Dict[str, str] = {
    "Randomized Controlled Trial": "Randomised Controlled Trial",
}

# The field used to identify a unique study for deduplication.
STUDY_ID_FIELD = "OldItemId"

# Departures from "source map equals codeset tumour group". Keyed by the exact
# codeset group AttributeName. Any group not listed here keeps its own name as
# the source map.
SOURCE_MAP_OF_GROUP: Dict[str, str] = {
    # Haema: the three AML codeset groups were one search and one map.
    "Acute myeloid leukaemia with defining genetic abnormalities": "Acute myeloid leukaemia",
    "Acute myeloid leukaemia defined by differentiation":          "Acute myeloid leukaemia",
    "Myeloid sarcoma":                                             "Acute myeloid leukaemia",
    # CNS: the Embryonal tumours codeset group is the Medulloblastoma map.
    "Embryonal tumours": "Medulloblastoma",
}


def display_char(name: str) -> str:
    """Characteristic display label (short form where defined)."""
    return CHARACTERISTIC_DISPLAY.get(name, name)


def display_design(label: Optional[str]) -> Optional[str]:
    """Study design display label (anglicised where defined)."""
    if label is None:
        return None
    return STUDY_DESIGN_DISPLAY.get(label, label)


def source_map_of_group(group_name: str) -> str:
    """Source map for a codeset tumour group, falling back to the group name."""
    return SOURCE_MAP_OF_GROUP.get(group_name, group_name)


def apply_style() -> None:
    """Matplotlib rcParams so every figure shares one look."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------

def load_merged(path: str) -> dict:
    """Load one merged Blue Book JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _children(node: dict) -> List[dict]:
    sub = node.get("Attributes", {})
    return (sub.get("AttributesList", []) if isinstance(sub, dict) else []) or []


def book_name(data: dict) -> str:
    """Blue Book name, read from the top tumour node of the codeset."""
    top = data["CodeSets"][0]["Attributes"]["AttributesList"]
    for a in top:
        lower = a["AttributeName"].lower()
        if ("characteristic" in lower or "study design" in lower
                or "level of evidence" in lower or "loe" in lower):
            continue
        return a["AttributeName"]
    return ""


def build_code_index(data: dict) -> dict:
    """Resolve the codeset hierarchy into lookups used downstream.

    Returns a dict with:
      char_id_to_name : AttributeId -> characteristic name (codeset string)
      loe_id_to_name  : AttributeId -> LoE band name
      design_id_to_name : AttributeId -> study-design name
      leaf_id_to_name : AttributeId -> tumour-type leaf name
      leaf_id_to_group : AttributeId -> codeset tumour group name
      leaf_id_to_map   : AttributeId -> source map
      leaves_in_order  : list of (leaf_id, leaf_name) in JSON tree order
      group_order      : codeset tumour groups in tree order
      map_order        : source maps in first-appearance (tree) order
      map_to_leaves    : source map -> list of (leaf_id, leaf_name) in order
      book             : Blue Book name
    """
    top = data["CodeSets"][0]["Attributes"]["AttributesList"]

    char_id_to_name: Dict[int, str] = {}
    loe_id_to_name: Dict[int, str] = {}
    design_id_to_name: Dict[int, str] = {}
    leaf_id_to_name: Dict[int, str] = {}
    leaf_id_to_group: Dict[int, str] = {}
    leaf_id_to_map: Dict[int, str] = {}
    leaves_in_order: List[Tuple[int, str]] = []
    group_order: List[str] = []
    map_order: List[str] = []
    map_to_leaves: Dict[str, List[Tuple[int, str]]] = {}
    book = ""

    def collect_leaves(node, into):
        kids = _children(node)
        if not kids:
            into.append((node["AttributeId"], node["AttributeName"]))
            return
        for c in kids:
            collect_leaves(c, into)

    for a in top:
        name = a["AttributeName"]
        lower = name.lower()
        if "characteristic" in lower:
            for ch in _children(a):
                char_id_to_name[ch["AttributeId"]] = ch["AttributeName"]
        elif "level of evidence" in lower or "loe" in lower:
            for ch in _children(a):
                loe_id_to_name[ch["AttributeId"]] = ch["AttributeName"]
        elif "study design" in lower:
            for ch in _children(a):
                design_id_to_name[ch["AttributeId"]] = ch["AttributeName"]
        else:
            book = name
            for group in _children(a):
                gname = group["AttributeName"]
                group_order.append(gname)
                smap = source_map_of_group(gname)
                gleaves: List[Tuple[int, str]] = []
                collect_leaves(group, gleaves)
                if smap not in map_to_leaves:
                    map_to_leaves[smap] = []
                    map_order.append(smap)
                for lid, lname in gleaves:
                    leaf_id_to_name[lid] = lname
                    leaf_id_to_group[lid] = gname
                    leaf_id_to_map[lid] = smap
                    leaves_in_order.append((lid, lname))
                    map_to_leaves[smap].append((lid, lname))

    return {
        "char_id_to_name": char_id_to_name,
        "loe_id_to_name": loe_id_to_name,
        "design_id_to_name": design_id_to_name,
        "leaf_id_to_name": leaf_id_to_name,
        "leaf_id_to_group": leaf_id_to_group,
        "leaf_id_to_map": leaf_id_to_map,
        "leaves_in_order": leaves_in_order,
        "group_order": group_order,
        "map_order": map_order,
        "map_to_leaves": map_to_leaves,
        "book": book,
    }


def entries_df(data: dict, idx: Optional[dict] = None) -> pd.DataFrame:
    """One row per (raw entry, tumour-type leaf).

    Columns: entry_id, study_id, doi, year, source_map, tumour_group,
    tumour_type, characteristic, study_design, loe.

    Raw entries with no tumour-type leaf, no characteristic or no LoE are
    dropped (they cannot be placed in any cell). The `entry_id` column lets
    callers recover raw-entry-level counts via drop_duplicates("entry_id").
    """
    if idx is None:
        idx = build_code_index(data)
    char_ids = idx["char_id_to_name"]
    loe_ids = idx["loe_id_to_name"]
    design_ids = idx["design_id_to_name"]
    leaf_name = idx["leaf_id_to_name"]
    leaf_group = idx["leaf_id_to_group"]
    leaf_map = idx["leaf_id_to_map"]

    rows = []
    dropped = 0
    for entry_id, r in enumerate(data.get("References", [])):
        code_ids = [c.get("AttributeId") for c in r.get("Codes", [])]
        leaves = [c for c in code_ids if c in leaf_name]
        characteristic = next((char_ids[c] for c in code_ids if c in char_ids), None)
        loe = next((loe_ids[c] for c in code_ids if c in loe_ids), None)
        design_raw = next((design_ids[c] for c in code_ids if c in design_ids), None)

        if not leaves or characteristic is None or loe is None:
            dropped += 1
            continue

        try:
            year = int(str(r.get("Year", "")).strip())
        except (ValueError, TypeError):
            year = np.nan

        study_id = r.get(STUDY_ID_FIELD)
        doi = str(r.get("DOI", "") or "").strip()
        design = display_design(design_raw)

        for lid in leaves:
            rows.append({
                "entry_id": entry_id,
                "study_id": study_id,
                "doi": doi,
                "year": year,
                "source_map": leaf_map[lid],
                "tumour_group": leaf_group[lid],
                "tumour_type": leaf_name[lid],
                "characteristic": characteristic,
                "study_design": design,
                "loe": loe,
            })

    df = pd.DataFrame(rows)
    df.attrs["dropped_entries"] = dropped
    return df


def unique_studies_df(entries: pd.DataFrame, by: str = "study_id") -> pd.DataFrame:
    """One row per unique study, keeping the first occurrence."""
    return entries.drop_duplicates(subset=[by]).reset_index(drop=True)


def raw_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """One row per raw entry (collapsing the per-leaf fan-out)."""
    return entries.drop_duplicates(subset=["entry_id"]).reset_index(drop=True)


def raw_entries_by_map(entries: pd.DataFrame) -> pd.DataFrame:
    """One row per (raw entry, source map). An entry is counted once per map."""
    return entries.drop_duplicates(subset=["entry_id", "source_map"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Filesystem-friendly slug for a source map name."""
    s = text.lower()
    s = s.replace("/", "-")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def map_slug(source_map: str) -> str:
    return _slug(source_map)


def figures_dir(base: str) -> Path:
    p = Path(base) / "4_Figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir(base: str) -> Path:
    p = Path(base) / "5_Tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def supplements_dir(base: str) -> Path:
    p = Path(base) / "supplements"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, outdir: Path, name: str) -> None:
    """Save a figure as PNG (300 dpi) and PDF under outdir/name.*"""
    base = Path(outdir) / name
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    """Save a DataFrame as a tab-separated file under outdir/name.tsv"""
    path = Path(outdir) / f"{name}.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path