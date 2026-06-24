"""
03_egm_figures_and_tables.py
============================
Single generator for every WCT EVI MAP publication figure and table, from one
merged Blue Book JSON. The same script serves Haema, Endo and CNS by changing
only the path on the command line.

Usage
-----
    python3 03_egm_figures_and_tables.py path/to/Report_<book>_merged_loe.json
    python3 03_egm_figures_and_tables.py path/to/file.json --only fig7_gap_matrix
    python3 03_egm_figures_and_tables.py path/to/file.json --output-dir /some/dir

Outputs (created next to the merged file, or under --output-dir):
    4_Figures/     Figure 3 to Figure 7 (main text)
    5_Tables/      Table 1 (main text)
    6_Supplemental_material/   Figure S1, Figure S2, Table S1, Table S2, Table S3, HETP

Author-made Figures 1 (map screenshot) and 2 (PRISMA flowchart) are not
produced here.

Counting units (see egm_common for the full rationale):
    Figure 3   unique studies per year
    Figure 4   unique studies per study design
    Figure 5   raw entries (characteristic x LoE, pooled)
    Figure 6   raw entries, once per source map (high-LoE share by characteristic)
    Figure 7   per-leaf records (one per matching tumour-type leaf)
    Table 1    unique studies where study-level, raw entries where entry-level
    Figure S1  raw entries per source map
    Figure S2  per-leaf records per source map
    Table S1   per-leaf records
    Table S2   per-leaf records
    Table S3   raw entries per source map
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb

import egm_common as ec


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.2f}%)" if total else f"{n:,} (0.00%)"


def _map_colours(map_order: List[str]):
    """Viridis 0.10 to 0.90 sampled across the source maps, smallest stop to
    largest, returned as a {source_map: rgba} dict. Stable per map order."""
    cmap = plt.get_cmap("viridis")
    n = len(map_order)
    if n == 1:
        stops = [0.5]
    else:
        stops = list(np.linspace(0.10, 0.90, n))
    return {m: cmap(s) for m, s in zip(map_order, stops)}


# ---------------------------------------------------------------------------
# Table 1: details of the evidence and gap maps
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    "Studies in EGM (n)",
    "Systematic reviews (n (%))",
    "Primary studies (n (%))",
    "Multi-coded tumour type studies (n (%))",
    "Multi-coded characteristic studies (n (%))",
    "Multi-coded studies in total (n (%))",
    "Entries in the EGM (n)",
    "Entries per tumour type (median (range))",
    "High-level evidence, P1 to P2 (n (%))",
    "Medium-level evidence, P3 (n (%))",
    "Low-level evidence, P4 to P5 (n (%))",
]


def _map_metrics(entries: pd.DataFrame, idx: dict, source_map: str) -> dict:
    """Raw metrics for one source map. Studies are unique study_id within the
    map; entries are raw entries within the map; entries per tumour type uses
    per-leaf fan-out across the map's leaves."""
    em = entries[entries["source_map"] == source_map]
    raw = em.drop_duplicates("entry_id")          # one row per raw entry
    n_entries = len(raw)
    n_studies = raw["study_id"].nunique()

    by_study = raw.groupby("study_id")
    is_sr = by_study["study_design"].apply(
        lambda s: s.dropna().astype(str).str.lower().str.contains("systematic review").any()
    )
    n_sr = int(is_sr.sum())
    n_primary = n_studies - n_sr

    # Multi-coding uses the full leaf-level frame for tumour types.
    leaves_per_study = em.groupby("study_id")["tumour_type"].nunique()
    multi_tum = set(leaves_per_study[leaves_per_study > 1].index)
    chars_per_study = raw.groupby("study_id")["characteristic"].nunique()
    multi_char = set(chars_per_study[chars_per_study > 1].index)
    multi_total = multi_tum | multi_char

    # Entries per tumour type: per-leaf fan-out across this map's leaves.
    leaf_names = [name for _, name in idx["map_to_leaves"][source_map]]
    leaf_counts = em["tumour_type"].value_counts()
    ept = [int(leaf_counts.get(name, 0)) for name in leaf_names]

    n_high = int(raw["loe"].isin(ec.HIGH_LOE).sum())
    n_medium = int((raw["loe"] == "Level P3").sum())
    n_low = int(raw["loe"].isin(ec.LOW_LOE).sum())

    return {
        "n_studies": n_studies, "n_sr": n_sr, "n_primary": n_primary,
        "n_multi_tum": len(multi_tum), "n_multi_char": len(multi_char),
        "n_multi_total": len(multi_total), "n_entries": n_entries,
        "ept": ept, "n_high": n_high, "n_medium": n_medium, "n_low": n_low,
    }


def _format_metric_column(raw: dict) -> dict:
    ns, ne, ept = raw["n_studies"], raw["n_entries"], raw["ept"]
    if ept:
        ept_str = f"{int(np.median(ept)):,} ({min(ept):,} to {max(ept):,})"
    else:
        ept_str = "n/a"
    return {
        "Studies in EGM (n)": f"{ns:,}",
        "Systematic reviews (n (%))": _pct(raw["n_sr"], ns),
        "Primary studies (n (%))": _pct(raw["n_primary"], ns),
        "Multi-coded tumour type studies (n (%))": _pct(raw["n_multi_tum"], ns),
        "Multi-coded characteristic studies (n (%))": _pct(raw["n_multi_char"], ns),
        "Multi-coded studies in total (n (%))": _pct(raw["n_multi_total"], ns),
        "Entries in the EGM (n)": f"{ne:,}",
        "Entries per tumour type (median (range))": ept_str,
        "High-level evidence, P1 to P2 (n (%))": _pct(raw["n_high"], ne),
        "Medium-level evidence, P3 (n (%))": _pct(raw["n_medium"], ne),
        "Low-level evidence, P4 to P5 (n (%))": _pct(raw["n_low"], ne),
    }


def table1_details_of_egm(data, entries, idx, outdir) -> None:
    maps = idx["map_order"]
    raws = {m: _map_metrics(entries, idx, m) for m in maps}

    # Total column: sum across maps (one study counted once per map it appears
    # in, no cross-map deduplication), matching the published Uro Table 2.
    total_raw = {
        "n_studies": sum(r["n_studies"] for r in raws.values()),
        "n_sr": sum(r["n_sr"] for r in raws.values()),
        "n_primary": sum(r["n_primary"] for r in raws.values()),
        "n_multi_tum": sum(r["n_multi_tum"] for r in raws.values()),
        "n_multi_char": sum(r["n_multi_char"] for r in raws.values()),
        "n_multi_total": sum(r["n_multi_total"] for r in raws.values()),
        "n_entries": sum(r["n_entries"] for r in raws.values()),
        "ept": [n for r in raws.values() for n in r["ept"]],
        "n_high": sum(r["n_high"] for r in raws.values()),
        "n_medium": sum(r["n_medium"] for r in raws.values()),
        "n_low": sum(r["n_low"] for r in raws.values()),
    }

    formatted = {m: _format_metric_column(raws[m]) for m in maps}
    total_col = _format_metric_column(total_raw)

    rows = []
    for metric in METRIC_ORDER:
        row = {"Metric": metric}
        for m in maps:
            row[m] = formatted[m][metric]
        row["Total"] = total_col[metric]
        rows.append(row)
    out_df = pd.DataFrame(rows)

    path = ec.save_table(out_df, ec.tables_dir(outdir), "Table1_details_of_EGM")

    # Transparency: the DOI-based grand-unique study count, which dedupes the
    # same paper coded under more than one map. Reported to console only; the
    # table keeps the summed convention above.
    doi_known = entries[entries["doi"] != ""]
    grand_unique_doi = doi_known["doi"].nunique()
    grand_unique_oldid = entries["study_id"].nunique()
    print(f"[Table1] saved to {path.name}")
    print(f"[Table1] grand-total studies, summed per map: {total_raw['n_studies']:,}")
    print(f"[Table1] grand-unique studies, DOI-based: {grand_unique_doi:,} "
          f"(OldItemId-based: {grand_unique_oldid:,})")


# ---------------------------------------------------------------------------
# Table 2: evidence-level summary by tumour group
# ---------------------------------------------------------------------------

def table2_evidence_level_by_tumour_group(data, entries, idx, outdir) -> None:
    """Evidence-level summary: tumour groups in the rows, evidence-level bands
    in the columns (high P1 to P2, moderate P3, low P4 to P5, unclassifiable),
    with a total column on the right and a total row at the bottom. Counting
    unit: raw EGM entries, each counted once for its source map."""
    maps = idx["map_order"]
    by_map = ec.raw_entries_by_map(entries)

    bands = [
        ("High-level evidence, P1 to P2, n (%)", ec.HIGH_LOE),
        ("Moderate-level evidence, P3, n (%)", {"Level P3"}),
        ("Low-level evidence, P4 to P5, n (%)", ec.LOW_LOE),
        ("Unclassifiable, n (%)", {"Unclassifiable"}),
    ]

    counts = {}
    for m in maps:
        seg = by_map[by_map["source_map"] == m]
        counts[m] = {col: int(seg["loe"].isin(levels).sum()) for col, levels in bands}
        counts[m]["total"] = sum(counts[m][col] for col, _ in bands)

    col_tot = {col: sum(counts[m][col] for m in maps) for col, _ in bands}
    grand = sum(counts[m]["total"] for m in maps)

    rows = []
    for m in maps:
        row = {"Tumour group": ec.display_map(m)}
        for col, _ in bands:
            row[col] = _pct(counts[m][col], counts[m]["total"])
        row["Total, n (%)"] = _pct(counts[m]["total"], grand)
        rows.append(row)
    trow = {"Tumour group": "Total"}
    for col, _ in bands:
        trow[col] = _pct(col_tot[col], grand)
    trow["Total, n (%)"] = _pct(grand, grand)
    rows.append(trow)

    cols = ["Tumour group"] + [c for c, _ in bands] + ["Total, n (%)"]
    out_df = pd.DataFrame(rows)[cols]
    path = ec.save_table(out_df, ec.tables_dir(outdir),
                         "Table2_evidence_level_by_tumour_group")
    print(f"[Table2] saved to {path.name}. Grand total entries: {grand:,}")


# ---------------------------------------------------------------------------
# Table S4: level of evidence by characteristic and tumour group (Uro format)
# ---------------------------------------------------------------------------

def tableS4_loe_by_characteristic_and_group(data, entries, idx, outdir) -> None:
    """LoE distribution across characteristics by tumour group, one combined
    table with a subtotal per characteristic, in the Uro publication format.
    Counting unit: raw EGM entries, each counted once for its source map."""
    maps = idx["map_order"]
    by_map = ec.raw_entries_by_map(entries)
    loe_cols = [f"{loe}, n (%)" for loe in ec.LOE_ORDER]

    rows = []
    grand = 0
    for ch in ec.CHARACTERISTIC_ORDER:
        sub_per_loe = {loe: 0 for loe in ec.LOE_ORDER}
        for i, m in enumerate(maps):
            seg = by_map[(by_map["source_map"] == m) &
                         (by_map["characteristic"] == ch)]
            per_loe = {loe: int((seg["loe"] == loe).sum()) for loe in ec.LOE_ORDER}
            row_total = sum(per_loe.values())
            row = {"Tumour characteristic": ec.display_char(ch) if i == 0 else "",
                   "Tumour group": ec.display_map(m),
                   "Total n": f"{row_total:,}"}
            for loe in ec.LOE_ORDER:
                row[f"{loe}, n (%)"] = _pct(per_loe[loe], row_total)
                sub_per_loe[loe] += per_loe[loe]
            rows.append(row)
        sub_total = sum(sub_per_loe.values())
        grand += sub_total
        srow = {"Tumour characteristic": "", "Tumour group": "Subtotal",
                "Total n": f"{sub_total:,}"}
        for loe in ec.LOE_ORDER:
            srow[f"{loe}, n (%)"] = _pct(sub_per_loe[loe], sub_total)
        rows.append(srow)

    cols = ["Tumour characteristic", "Tumour group", "Total n"] + loe_cols
    out_df = pd.DataFrame(rows)[cols]
    path = ec.save_table(out_df, ec.supplements_dir(outdir),
                         "TableS4_LoE_by_characteristic_and_group")
    print(f"[TableS4] saved to {path.name}. Grand total entries: {grand:,}")


# ---------------------------------------------------------------------------
# Figure 3: unique studies per publication year, stacked by source map
# ---------------------------------------------------------------------------

def fig3_studies_per_year(data, entries, idx, outdir) -> None:
    maps = idx["map_order"]
    # One row per (study, source map), so a study counts once per map per year.
    per = (entries.drop_duplicates(["study_id", "source_map"])
                  .dropna(subset=["year"]))
    if per.empty:
        print("[Figure3] no years available, skipping")
        return
    per = per.copy()
    per["year"] = per["year"].astype(int)
    years = list(range(per["year"].min(), per["year"].max() + 1))

    counts = {m: [int(((per["source_map"] == m) & (per["year"] == y)).sum())
                  for y in years] for m in maps}
    stack_order = sorted(maps, key=lambda m: sum(counts[m]))
    colours = _map_colours(stack_order)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(years))
    bar_width = 0.62
    bottoms = np.zeros(len(years))
    totals = np.sum([counts[m] for m in stack_order], axis=0)

    # Adaptive inline-label threshold: a segment is labelled inside when it is
    # at least ~2.5% of the tallest bar (small absolute floor for low-volume
    # plots). Anything smaller gets a callout with a leader line so every
    # number appears.
    max_total = int(totals.max()) if totals.max() > 0 else 1
    inline_threshold = max(3, int(round(0.025 * max_total)))
    segment_bottoms = {}

    for m in stack_order:
        vals = np.array(counts[m])
        segment_bottoms[m] = bottoms.copy()
        bars = ax.bar(x, vals, bar_width, bottom=bottoms, color=colours[m],
                      edgecolor="white", linewidth=0.6, label=ec.display_map(m))
        r, g_c, b, _ = colours[m]
        lum = 0.299 * r + 0.587 * g_c + 0.114 * b
        text_colour = "white" if lum < 0.55 else "black"
        for i, (bar, value) in enumerate(zip(bars, vals)):
            seg_height = bar.get_height()
            if seg_height < inline_threshold:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bottoms[i] + seg_height / 2,
                    f"{int(value)}", ha="center", va="center", fontsize=9.5,
                    color=text_colour, fontweight="medium")
        bottoms += vals

    # Grand totals on top of each bar.
    for xi, total in zip(x, totals):
        if total > 0:
            ax.text(xi, total + totals.max() * 0.012, f"{int(total)}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="black")

    # Callouts for small segments, placed just right of the bar with a thin
    # grey leader line and a white text outline for legibility.
    for m in stack_order:
        vals = np.array(counts[m])
        for i, value in enumerate(vals):
            if value <= 0 or value >= inline_threshold:
                continue
            seg_centre_y = float(segment_bottoms[m][i]) + value / 2
            bar_right_edge = float(x[i]) + bar_width / 2
            label_x = bar_right_edge + 0.08
            ax.plot([bar_right_edge + 0.005, label_x - 0.015],
                    [seg_centre_y, seg_centre_y], color="dimgray", lw=0.6, zorder=4)
            txt = ax.text(label_x, seg_centre_y, f"{int(value)}", fontsize=9,
                          color="black", ha="left", va="center", zorder=5)
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_xlabel("Publication year", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of studies", fontsize=12, labelpad=8)
    ax.set_title(f"Studies per year by tumour group ({years[0]} to {years[-1]})",
                 fontsize=13, pad=14)

    # Headroom for the total labels; matplotlib auto-picks the tick step,
    # which gives 50-unit ticks at this scale as in the reports.
    ceiling = totals.max() * 1.12
    ax.set_ylim(0, ceiling)
    ax.tick_params(axis="y", labelsize=10)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend outside on the right, items sorted alphabetically top to bottom.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].lower())
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True,
              edgecolor="lightgray", fontsize=10, title="Tumour group",
              title_fontsize=10.5)
    fig.tight_layout()

    ec.save_figure(fig, ec.figures_dir(outdir), "Figure3_studies_per_year")
    src = pd.DataFrame({"Year": years, **{m: counts[m] for m in maps}})
    src["Total"] = src[maps].sum(axis=1)
    ec.save_table(src, ec.figures_dir(outdir), "Figure3_studies_per_year_data")
    print(f"[Figure3] saved. Years {years[0]} to {years[-1]}, "
          f"{int(totals.sum()):,} study-by-map counts.")


# ---------------------------------------------------------------------------
# Figure 4: studies per study design, stacked by source map
# ---------------------------------------------------------------------------

def fig4_study_designs(data, entries, idx, outdir) -> None:
    maps = idx["map_order"]
    # Counting unit: unique studies per map (study_id), one count per study
    # design within a map. A study coded under several characteristics or LoE
    # still counts once for its design in a given map. A cross-map study counts
    # once in each map, so a bar length is the sum across maps, not the
    # distinct union; the TSV Total column uses the same sum convention.
    per = entries.dropna(subset=["study_design"]).drop_duplicates(
        ["study_id", "study_design", "source_map"])
    designs = sorted(per["study_design"].unique(), key=str.lower)
    if not designs:
        print("[Figure4] no study designs found, skipping")
        return
    if len(maps) < 2:
        print("[Figure4] only one map, skipping all-maps chart")
        return

    counts = {m: [int(((per["study_design"] == d) & (per["source_map"] == m)).sum())
                  for d in designs] for m in maps}
    stack_order = sorted(maps, key=lambda m: sum(counts[m]))
    colours = _map_colours(stack_order)

    # Designs alphabetical top to bottom; reverse for matplotlib's bottom-up
    # y-axis so alphabetical order reads top down on screen.
    designs_plot = list(reversed(designs))
    counts_plot = {m: list(reversed(counts[m])) for m in maps}

    fig_h = max(5.0, 0.36 * len(designs) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = np.arange(len(designs_plot))
    bar_height = 0.62
    lefts = np.zeros(len(designs_plot))

    per_group_values = {}
    for m in stack_order:
        vals = np.array(counts_plot[m], dtype=float)
        per_group_values[m] = vals
        ax.barh(y, vals, bar_height, left=lefts, color=colours[m],
                edgecolor="white", linewidth=0.6, label=ec.display_map(m))
        lefts += vals

    totals = lefts.copy()
    bar_max = float(totals.max()) if len(totals) else 1.0
    inline_threshold = max(3, int(round(0.04 * bar_max)))

    # Inline labels for segments at or above the threshold.
    accum = np.zeros(len(designs_plot))
    for m in stack_order:
        vals = per_group_values[m]
        r, g_c, b, _ = colours[m]
        lum = 0.299 * r + 0.587 * g_c + 0.114 * b
        text_colour = "white" if lum < 0.55 else "black"
        for yi, v in zip(y, vals):
            if v >= inline_threshold:
                ax.text(accum[int(yi)] + v / 2, yi, f"{int(v)}", ha="center",
                        va="center", fontsize=8.5, color=text_colour,
                        fontweight="medium")
        accum += vals

    # Bold per-design totals at the right end of each bar.
    for yi, total in zip(y, totals):
        if total > 0:
            ax.text(total + bar_max * 0.01, yi, f"{int(total):,}", ha="left",
                    va="center", fontsize=9.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(designs_plot, fontsize=9)
    ax.set_ylabel("Study design", labelpad=8)
    ax.set_xlabel("Number of studies", fontsize=11, labelpad=8)
    ax.set_title("Studies per study design", fontsize=12.5, pad=12)
    ax.set_xlim(0, bar_max * 1.10 if bar_max > 0 else 1)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend alphabetical top to bottom, inside the lower-right corner where
    # the shortest bars leave empty space.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].casefold())
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", frameon=True, edgecolor="lightgray",
              fontsize=10, title="Tumour group", title_fontsize=10.5)
    fig.tight_layout()

    ec.save_figure(fig, ec.figures_dir(outdir), "Figure4_study_designs")
    src = pd.DataFrame({"Study design": designs,
                        **{m: counts[m] for m in maps}})
    src["Total"] = src[maps].sum(axis=1)
    ec.save_table(src, ec.figures_dir(outdir), "Figure4_study_designs_data")
    print(f"[Figure4] saved. {len(designs)} distinct study designs.")


# ---------------------------------------------------------------------------
# Characteristic x LoE stacked bars (shared core for Figure 5 and Figure S1)
# ---------------------------------------------------------------------------

def _char_loe_counts(raw: pd.DataFrame) -> np.ndarray:
    """counts[characteristic, LoE] over raw entries (one per entry)."""
    counts = np.zeros((len(ec.CHARACTERISTIC_ORDER), len(ec.LOE_ORDER)), dtype=int)
    for i, ch in enumerate(ec.CHARACTERISTIC_ORDER):
        sub = raw[raw["characteristic"] == ch]
        for j, loe in enumerate(ec.LOE_ORDER):
            counts[i, j] = int((sub["loe"] == loe).sum())
    return counts


def _plot_char_loe(ax, counts: np.ndarray, title: str,
                   inline_labels: bool = True, callouts: bool = True,
                   ylabel: bool = True) -> None:
    x = np.arange(len(ec.CHARACTERISTIC_ORDER))
    bar_width = 0.7
    bottoms = np.zeros(len(ec.CHARACTERISTIC_ORDER))
    totals = counts.sum(axis=1)
    max_total = int(totals.max()) if totals.max() > 0 else 1
    inline_threshold = max(3, int(round(0.025 * max_total)))

    # Draw order: Unclassifiable at the bottom up to Level P1 on top, so the
    # highest level of evidence sits at the top of every bar.
    draw_order = list(reversed(ec.LOE_ORDER))
    for loe in draw_order:
        j = ec.LOE_ORDER.index(loe)
        vals = counts[:, j].astype(float)
        ax.bar(x, vals, bar_width, bottom=bottoms, color=ec.LOE_COLOURS[loe],
               edgecolor="white", linewidth=0.6, label=loe)
        if inline_labels:
            r, g_c, b = to_rgb(ec.LOE_COLOURS[loe])
            lum = 0.299 * r + 0.587 * g_c + 0.114 * b
            text_colour = "white" if lum < 0.55 else "black"
            for i, v in enumerate(vals):
                if v >= inline_threshold:
                    ax.text(x[i], bottoms[i] + v / 2, f"{int(v)}", ha="center",
                            va="center", fontsize=9.5, color=text_colour,
                            fontweight="medium")
        bottoms += vals

    # Grand totals on top of each bar.
    for xi, total in zip(x, totals):
        if total > 0:
            ax.text(xi, total + max_total * 0.012, f"{int(total):,}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="black")

    # Callouts for small segments below the inline threshold, placed just right
    # of the bar with a thin grey leader line and a white text outline.
    if callouts:
        cumulative = np.zeros(len(ec.CHARACTERISTIC_ORDER))
        for loe in draw_order:
            j = ec.LOE_ORDER.index(loe)
            vals = counts[:, j]
            for i, v in enumerate(vals):
                if 0 < v < inline_threshold:
                    bar_right = float(x[i]) + bar_width / 2
                    label_x = bar_right + 0.08
                    seg_centre = cumulative[i] + v / 2
                    ax.plot([bar_right + 0.005, label_x - 0.015],
                            [seg_centre, seg_centre], color="dimgray", lw=0.6, zorder=4)
                    txt = ax.text(label_x, seg_centre, f"{int(v)}", fontsize=9,
                                  color="black", ha="left", va="center", zorder=5)
                    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
                cumulative[i] += v

    ax.set_xticks(x)
    ax.set_xticklabels([ec.display_char(c) for c in ec.CHARACTERISTIC_ORDER],
                       rotation=30, ha="right")
    ax.set_xlabel("Characteristics", labelpad=8)
    if ylabel:
        ax.set_ylabel("Number of EGM entries")
    ax.set_title(title)
    ax.set_ylim(0, max_total * 1.12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig5_characteristics_by_loe(data, entries, idx, outdir) -> None:
    """Pooled across all tumour types: characteristic x LoE, raw entries."""
    raw = ec.raw_entries(entries)
    counts = _char_loe_counts(raw)
    fig, ax = plt.subplots(figsize=(11, 7))
    _plot_char_loe(ax, counts,
                   f"{ec.book_name(data)}: Characteristics by level of evidence",
                   inline_labels=True, callouts=False)
    # Legend in canonical LoE order (P1 first) outside on the right with a grey
    # border, matching the Uro paper Figure 4.
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered = [(label_to_handle[l], l) for l in ec.LOE_ORDER if l in label_to_handle]
    leg = ax.legend([h for h, _ in ordered], [l for _, l in ordered],
                    loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True,
                    edgecolor="lightgray", fontsize=10, title="Level of Evidence",
                    title_fontsize=10.5)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    ec.save_figure(fig, ec.figures_dir(outdir),
                   "Figure5_characteristics_by_loe_pooled")

    src = pd.DataFrame(counts, index=ec.CHARACTERISTIC_ORDER, columns=ec.LOE_ORDER)
    src.insert(0, "Characteristic", src.index)
    ec.save_table(src.reset_index(drop=True), ec.figures_dir(outdir),
                  "Figure5_characteristics_by_loe_pooled_data")
    print(f"[Figure5] saved. {int(counts.sum()):,} entries pooled.")


def figS1_characteristics_by_loe_by_group(data, entries, idx, outdir) -> None:
    """Panels two per row: characteristic x LoE, raw entries per map. For an
    odd number of groups the legend sits in the empty grid cell beside the
    last panel; for an even number it sits to the right of the grid."""
    maps = idx["map_order"]
    by_map = ec.raw_entries_by_map(entries)
    n = len(maps)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.6 * nrows),
                             squeeze=False)
    src_rows = []
    for k, m in enumerate(maps):
        r, c = divmod(k, ncols)
        raw = by_map[by_map["source_map"] == m]
        counts = _char_loe_counts(raw)
        _plot_char_loe(axes[r][c], counts, ec.display_map(m),
                       callouts=False, ylabel=False)
        for i, ch in enumerate(ec.CHARACTERISTIC_ORDER):
            row = {"Source map": m, "Characteristic": ch}
            for j, loe in enumerate(ec.LOE_ORDER):
                row[loe] = int(counts[i, j])
            src_rows.append(row)

    handles, labels = axes[0][0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered = [(label_to_handle[l], l) for l in ec.LOE_ORDER if l in label_to_handle]
    leg_handles = [h for h, _ in ordered]
    leg_labels = [l for _, l in ordered]

    empty = nrows * ncols - n
    if empty:
        # Legend in the first empty cell, beside the last panel.
        r, c = divmod(n, ncols)
        axes[r][c].axis("off")
        axes[r][c].legend(leg_handles, leg_labels, title="Level of Evidence",
                          loc="center", frameon=True, edgecolor="lightgray")
        for k in range(n + 1, nrows * ncols):  # any further empty cells
            r2, c2 = divmod(k, ncols)
            axes[r2][c2].axis("off")
    else:
        fig.legend(leg_handles, leg_labels, title="Level of Evidence",
                   loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True,
                   edgecolor="lightgray")

    fig.supylabel("Number of EGM entries")
    fig.suptitle(f"{ec.book_name(data)}: Characteristics by level of evidence "
                 f"per tumour group", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ec.save_figure(fig, ec.supplements_dir(outdir),
                   "FigureS1_characteristics_by_loe_by_group")
    ec.save_table(pd.DataFrame(src_rows), ec.supplements_dir(outdir),
                  "FigureS1_characteristics_by_loe_by_group_data")
    print(f"[FigureS1] saved. {n} panel(s) in {nrows}x{ncols} grid.")


# ---------------------------------------------------------------------------
# Tumour type x LoE per source map (Figure S2)
# ---------------------------------------------------------------------------

def figS2_tumour_type_by_loe_by_group(data, entries, idx, outdir) -> None:
    """One panel per source map: tumour-type leaves x LoE, per-leaf records."""
    maps = idx["map_order"]
    n = len(maps)
    heights = [max(3.4, 0.42 * len(idx["map_to_leaves"][m]) + 1.6) for m in maps]
    fig, axes = plt.subplots(n, 1, figsize=(11, sum(heights)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    src_rows = []
    for k, m in enumerate(maps):
        ax = axes[k][0]
        em = entries[entries["source_map"] == m]
        leaves = [name for _, name in idx["map_to_leaves"][m]]
        y = np.arange(len(leaves))
        lefts = np.zeros(len(leaves))
        for loe in ec.LOE_ORDER:
            vals = np.array([int(((em["tumour_type"] == t) & (em["loe"] == loe)).sum())
                             for t in leaves])
            ax.barh(y, vals, left=lefts, height=0.7, color=ec.LOE_COLOURS[loe],
                    edgecolor="white", linewidth=0.4, label=loe)
            lefts += vals
        ax.set_yticks(y)
        ax.set_yticklabels(leaves, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Number of records")
        ax.set_title(ec.display_map(m))
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for t in leaves:
            row = {"Source map": m, "Tumour type": t}
            sub = em[em["tumour_type"] == t]
            for loe in ec.LOE_ORDER:
                row[loe] = int((sub["loe"] == loe).sum())
            row["Total"] = int(len(sub))
            src_rows.append(row)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Level of evidence",
               loc="upper right", frameon=False)
    fig.suptitle(f"Tumour types by level of evidence per source map, "
                 f"{ec.book_name(data)}", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    ec.save_figure(fig, ec.supplements_dir(outdir),
                   "FigureS2_tumour_type_by_loe_by_group")
    ec.save_table(pd.DataFrame(src_rows), ec.supplements_dir(outdir),
                  "FigureS2_tumour_type_by_loe_by_group_data")
    print(f"[FigureS2] saved. {n} panel(s).")


# ---------------------------------------------------------------------------
# Figure 6: high-level evidence heatmap, source map x characteristic
# (ported from the Uro figure 5 script)
# ---------------------------------------------------------------------------

def fig6_high_loe_heatmap(data, entries, idx, outdir) -> None:
    maps = idx["map_order"]
    chars = ec.CHARACTERISTIC_ORDER
    # Entry once per source map, so within-group multi-leaf coding does not
    # inflate the denominator (matches the Uro figure 5 counting rule).
    by_map = ec.raw_entries_by_map(entries)

    num = np.zeros((len(maps), len(chars)), dtype=int)
    den = np.zeros((len(maps), len(chars)), dtype=int)
    for i, m in enumerate(maps):
        em = by_map[by_map["source_map"] == m]
        for j, ch in enumerate(chars):
            sub = em[em["characteristic"] == ch]
            den[i, j] = len(sub)
            num[i, j] = int(sub["loe"].isin(ec.HIGH_LOE).sum())

    pct = np.divide(num * 100.0, den, out=np.zeros_like(num, dtype=float),
                    where=den > 0)

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 1.1 * len(maps) + 2)))
    im = ax.imshow(pct, cmap=plt.get_cmap("Blues"), vmin=0, vmax=100,
                   aspect="auto")
    for i in range(len(maps)):
        for j in range(len(chars)):
            d, nval, value = den[i, j], num[i, j], pct[i, j]
            tcol = "white" if value > 50 else "black"
            top = "n/a" if d == 0 else f"{value:.2f}%"
            ax.text(j, i - 0.12, top, ha="center", va="center", fontsize=11,
                    color=tcol)
            ax.text(j, i + 0.18, f"{nval:,}/{d:,}", ha="center", va="center",
                    fontsize=8.5, color=tcol)

    ax.set_xticks(np.arange(len(chars)))
    ax.set_xticklabels([ec.display_char(c) for c in chars], rotation=30,
                       ha="right")
    ax.set_yticks(np.arange(len(maps)))
    ax.set_yticklabels([ec.display_map(m) for m in maps])
    ax.set_xlabel("Characteristic", labelpad=10)
    ax.set_ylabel("Tumour group", labelpad=10)
    ax.set_title("High-level evidence (P1 plus P2) by tumour group and "
                 "characteristic", pad=14)
    ax.set_xticks(np.arange(len(chars) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(maps) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Percent high-level evidence", labelpad=8)
    fig.tight_layout()
    ec.save_figure(fig, ec.figures_dir(outdir), "Figure6_high_loe_heatmap")

    src = pd.DataFrame(
        [{"Source map": m,
          **{ec.display_char(chars[j]):
             (f"{pct[i, j]:.2f}" if den[i, j] else "n/a")
             for j in range(len(chars))}}
         for i, m in enumerate(maps)])
    ec.save_table(src, ec.figures_dir(outdir), "Figure6_high_loe_heatmap_data")
    print(f"[Figure6] saved. {len(maps)} x {len(chars)} cells.")


# ---------------------------------------------------------------------------
# Figure 7: combined gap classification matrix, tumour type x characteristic
# (ported from the Uro figure 6 script, combined across all maps)
# ---------------------------------------------------------------------------

def _classify_cell(n_total: int, n_high: int) -> str:
    if n_total == 0:
        return "Absolute gap"
    if n_high == 0:
        return "Relative gap" if n_total < ec.RELATIVE_GAP_MAX_N else "Synthesis gap"
    return "Solid evidence"


def fig7_gap_matrix(data, entries, idx, outdir) -> None:
    maps = idx["map_order"]
    chars = ec.CHARACTERISTIC_ORDER

    # Row order: leaves in JSON tree order, grouped by source map.
    rows = []  # (source_map, leaf_name)
    for m in maps:
        for _, name in idx["map_to_leaves"][m]:
            rows.append((m, name))
    n_rows, n_cols = len(rows), len(chars)

    # Per-leaf records: count entries rows by (leaf, characteristic).
    def cell_counts(leaf, ch):
        sub = entries[(entries["tumour_type"] == leaf) &
                      (entries["characteristic"] == ch)]
        total = len(sub)
        high = int(sub["loe"].isin(ec.HIGH_LOE).sum())
        low = int(sub["loe"].isin(ec.LOW_LOE).sum())
        return total, high, low

    colour = np.zeros((n_rows, n_cols, 3))
    annot = np.empty((n_rows, n_cols), dtype=object)
    tcol = np.empty((n_rows, n_cols), dtype=object)
    counter = Counter()
    counts_cache = {}
    for ri, (_m, leaf) in enumerate(rows):
        for ci, ch in enumerate(chars):
            total, high, low = cell_counts(leaf, ch)
            counts_cache[(ri, ci)] = (total, high, low)
            cat = _classify_cell(total, high)
            counter[cat] += 1
            hexc = ec.GAP_PALETTE[cat]
            colour[ri, ci] = [int(hexc[1:3], 16) / 255,
                              int(hexc[3:5], 16) / 255,
                              int(hexc[5:7], 16) / 255]
            annot[ri, ci] = str(total)
            tcol[ri, ci] = ec.GAP_TEXT_COLOUR[cat]

    fig_h = max(8.0, 0.34 * n_rows + 3.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.imshow(colour, aspect="auto", interpolation="nearest")
    for ri in range(n_rows):
        for ci in range(n_cols):
            ax.text(ci, ri, annot[ri, ci], ha="center", va="center",
                    fontsize=9, color=tcol[ri, ci], fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([ec.display_char(c) for c in chars], rotation=35,
                       ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([leaf for _m, leaf in rows], fontsize=8.5)
    ax.set_xlabel("Characteristic", labelpad=10)
    ax.set_ylabel("Tumour type", labelpad=10)
    ax.set_title("Gap classification by tumour type and characteristic", pad=14)

    # Right-side brackets per source map. The bracket opens to the left so it
    # reads as embracing the tumour-type rows that belong to the group.
    bracket_x = n_cols - 0.5 + 0.25
    label_x = bracket_x + 0.45
    cumulative = 0
    for m in maps:
        n = len(idx["map_to_leaves"][m])
        start, end = cumulative, cumulative + n - 1
        cumulative += n
        top, bot = start - 0.45, end + 0.45
        ax.annotate("", xy=(bracket_x, top), xytext=(bracket_x, bot),
                    xycoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#333333",
                                    linewidth=1.4))
        for yb in (top, bot):
            ax.plot([bracket_x, bracket_x - 0.15], [yb, yb],
                    color="#333333", linewidth=1.4, clip_on=False)
        ax.annotate(ec.display_map(m), xy=(label_x, (start + end) / 2), xycoords="data",
                    ha="left", va="center", fontsize=9, color="#333333",
                    fontweight="bold", annotation_clip=False)

    legend_handles = [mpatches.Patch(facecolor=ec.GAP_PALETTE[c],
                                     edgecolor="#444444", label=c)
                      for c in ec.GAP_CATEGORIES]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(0.98, 0.02), bbox_transform=fig.transFigure,
              ncol=2, frameon=True, edgecolor="lightgray", fontsize=10,
              title="Gap category", title_fontsize=11)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    ec.save_figure(fig, ec.figures_dir(outdir), "Figure7_gap_matrix")

    # Companion data: labels and counts per cell.
    out_rows = []
    for ri, (m, leaf) in enumerate(rows):
        row = {"Source map": m, "Tumour type": leaf}
        for ci, ch in enumerate(chars):
            total, high, low = counts_cache[(ri, ci)]
            cat = _classify_cell(total, high)
            row[ec.display_char(ch)] = f"{cat} (n = {total:,})"
        out_rows.append(row)
    ec.save_table(pd.DataFrame(out_rows), ec.figures_dir(outdir),
                  "Figure7_gap_matrix_data")

    total = sum(counter.values())
    summary = ", ".join(f"{c} {counter[c]}" for c in ec.GAP_CATEGORIES)
    print(f"[Figure7] saved. {n_rows} x {n_cols} cells. {summary} "
          f"(of {total}).")


# ---------------------------------------------------------------------------
# Supplementary tables, one TSV per source map
# ---------------------------------------------------------------------------

def _letter(pos: int) -> str:
    return chr(ord("a") + pos)


def _per_map_axis_table(entries, idx, outdir, axis_values, axis_col,
                        base_name, descriptor, with_pct=False):
    """Shared core for Table S1 (characteristic), S2 (LoE) and S3 (char x LoE
    is handled separately). Here axis is a single categorical column counted
    against tumour-type leaves, one TSV per source map."""
    maps = idx["map_order"]
    supp = ec.supplements_dir(outdir)
    saved = []
    for pos, m in enumerate(maps):
        em = entries[entries["source_map"] == m]
        leaves = [name for _, name in idx["map_to_leaves"][m]]
        rows = []
        for leaf in leaves:
            sub = em[em["tumour_type"] == leaf]
            row = {"Tumour group": m, "Tumour type": leaf}
            for v in axis_values:
                row[v] = f"{int((sub[axis_col] == v).sum()):,}"
            row["Total n"] = f"{len(sub):,}"
            rows.append(row)
        # Group total uses entry-once-per-map (raw entries within the map).
        raw_m = em.drop_duplicates("entry_id")
        total_row = {"Tumour group": "", "Tumour type": f"Total: {m}"}
        for v in axis_values:
            total_row[v] = f"{int((raw_m[axis_col] == v).sum()):,}"
        total_row["Total n"] = f"{len(raw_m):,}"
        rows.append(total_row)

        name = f"{base_name}{_letter(pos)}_{ec.map_slug(m)}_{descriptor}"
        path = ec.save_table(pd.DataFrame(rows), supp, name)
        saved.append(path.name)
    return saved


def tableS1_type_by_characteristic(data, entries, idx, outdir) -> None:
    saved = _per_map_axis_table(
        entries, idx, outdir, ec.CHARACTERISTIC_ORDER, "characteristic",
        "TableS1", "tumour_type_by_characteristic")
    print(f"[TableS1] saved {len(saved)} per-map TSV(s): {', '.join(saved)}")


def tableS2_type_by_loe(data, entries, idx, outdir) -> None:
    saved = _per_map_axis_table(
        entries, idx, outdir, ec.LOE_ORDER, "loe",
        "TableS2", "tumour_type_by_loe")
    print(f"[TableS2] saved {len(saved)} per-map TSV(s): {', '.join(saved)}")


def tableS3_characteristics_by_loe(data, entries, idx, outdir) -> None:
    """Characteristic x LoE, one TSV per source map. Counts and within-map
    percentages over raw entries (one per entry)."""
    maps = idx["map_order"]
    supp = ec.supplements_dir(outdir)
    by_map = ec.raw_entries_by_map(entries)
    saved = []
    for pos, m in enumerate(maps):
        raw = by_map[by_map["source_map"] == m]
        n_entries = len(raw)
        counts = _char_loe_counts(raw)

        def cell(n):
            return _pct(int(n), n_entries)

        rows = []
        for i, ch in enumerate(ec.CHARACTERISTIC_ORDER):
            row = {"Tumour characteristic": ch}
            for j, loe in enumerate(ec.LOE_ORDER):
                row[loe] = cell(counts[i, j])
            row["Total"] = cell(counts[i, :].sum())
            rows.append(row)
        total_row = {"Tumour characteristic": "Total"}
        for j, loe in enumerate(ec.LOE_ORDER):
            total_row[loe] = cell(counts[:, j].sum())
        total_row["Total"] = cell(counts.sum())
        rows.append(total_row)

        name = f"TableS3{_letter(pos)}_{ec.map_slug(m)}_characteristics_by_loe"
        path = ec.save_table(pd.DataFrame(rows), supp, name)
        saved.append(path.name)
    print(f"[TableS3] saved {len(saved)} per-map TSV(s): {', '.join(saved)}")


# ---------------------------------------------------------------------------
# Supplementary HETP table (anglicised study-design to LoE lookup)
# ---------------------------------------------------------------------------

HETP_LOOKUP_CANDIDATES = [
    "01_study_design_to_loe_HETP.json",
    "study_design_to_loe_HETP.json",
    "2_study_design_to_loe_HETP.json",
]


def supp_hetp(data, entries, idx, outdir) -> None:
    """Render the HETP study-design to LoE lookup as a supplementary table,
    anglicised at the display layer. The lookup file is searched for next to
    this script and next to the merged JSON. If not found, the step is skipped
    with a note rather than failing the run."""
    search = []
    here = Path(__file__).resolve().parent
    base = Path(outdir)
    for cand in HETP_LOOKUP_CANDIDATES:
        search.append(here / cand)
        search.append(base / cand)
    path = next((p for p in search if p.exists()), None)
    if path is None:
        print("[HETP] lookup JSON not found next to script or output dir, "
              "skipping. Looked for: " + ", ".join(HETP_LOOKUP_CANDIDATES))
        return

    lookup = ec.load_merged(str(path))
    rows = []
    if isinstance(lookup, dict):
        items = lookup.items()
    else:
        items = []
    for design, loe in items:
        rows.append({"Study design": ec.display_design(design),
                     "Level of evidence": loe})
    rows.sort(key=lambda r: str(r["Study design"]).lower())
    ec.save_table(pd.DataFrame(rows), ec.supplements_dir(outdir),
                  "SupplementaryTable_HETP_study_design_to_loe")
    print(f"[HETP] saved {len(rows)} study-design rows from {path.name}")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

BUILDERS = {
    # main text
    "table1_details_of_egm":                 table1_details_of_egm,
    "table2_evidence_level_by_tumour_group": table2_evidence_level_by_tumour_group,
    "fig3_studies_per_year":                 fig3_studies_per_year,
    "fig4_study_designs":                    fig4_study_designs,
    "fig5_characteristics_by_loe":           fig5_characteristics_by_loe,
    "fig6_high_loe_heatmap":                 fig6_high_loe_heatmap,
    "fig7_gap_matrix":                       fig7_gap_matrix,
    # supplements
    "figS1_characteristics_by_loe_by_group": figS1_characteristics_by_loe_by_group,
    "figS2_tumour_type_by_loe_by_group":     figS2_tumour_type_by_loe_by_group,
    "tableS1_type_by_characteristic":        tableS1_type_by_characteristic,
    "tableS2_type_by_loe":                   tableS2_type_by_loe,
    "tableS3_characteristics_by_loe":        tableS3_characteristics_by_loe,
    "tableS4_loe_by_characteristic_and_group": tableS4_loe_by_characteristic_and_group,
    "supp_hetp":                             supp_hetp,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all WCT EVI MAP figures and tables for one "
                    "merged Blue Book JSON.")
    parser.add_argument("merged_json", help="path to Report_<book>_merged_loe.json")
    parser.add_argument("--output-dir", default=None,
                        help="base output dir (default: next to the JSON)")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=list(BUILDERS),
                        help="build only the named output(s)")
    args = parser.parse_args()

    data = ec.load_merged(args.merged_json)
    idx = ec.build_code_index(data)
    entries = ec.entries_df(data, idx)
    outdir = args.output_dir or os.path.dirname(os.path.abspath(args.merged_json))
    ec.apply_style()

    print(f"Blue Book: {ec.book_name(data)}")
    print(f"Source maps: {', '.join(idx['map_order'])}")
    print(f"Raw References: {len(data.get('References', [])):,}  "
          f"dropped (incomplete): {entries.attrs.get('dropped_entries', 0):,}")
    print(f"Leaf-level rows: {len(entries):,}  "
          f"raw entries: {entries['entry_id'].nunique():,}  "
          f"unique studies: {entries['study_id'].nunique():,}")
    print(f"Output base: {outdir}\n")

    for name in (args.only or BUILDERS):
        BUILDERS[name](data, entries, idx, outdir)


if __name__ == "__main__":
    main()