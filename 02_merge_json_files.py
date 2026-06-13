"""
merge_haem_json.py
==================
Merges two WCT EVI MAP coding JSON files (AML and B-ALL/LBL) into a single
Haematolymphoid Tumours JSON.

Merge logic
-----------
- The top-level CodeSets wrapper and the shared attribute axes
  (Characteristics, Study design, Level of Evidence) are taken from the
  AML file (primary source).
- The Haematolymphoid Tumours tumour-group/tumour-type hierarchy from B-ALL
  is appended after the AML hierarchy under the same parent node.
- References from both files are combined and de-duplicated on ReferenceId.
- All other metadata (SetName, SetId, etc.) is preserved from the AML file.

Usage
-----
    python merge_haem_json.py [AML_JSON] [BALL_JSON] [OUTPUT_JSON]

Defaults
--------
    AML_JSON  : Report_AML_loe.json
    BALL_JSON : Report_B-ALL_loe.json
    OUTPUT_JSON: Report_Haem_merged_loe.json
"""

import json
import sys
import copy
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"Saved merged JSON to: {path}")


def get_tumour_groups(codeset: dict) -> list:
    """Return the list of tumour-group dicts nested inside the
    Haematolymphoid Tumours attribute of a given CodeSet entry."""
    attrs = codeset["Attributes"]["AttributesList"]
    # First attribute is always "Haematolymphoid Tumours"
    haem_attr = attrs[0]
    return haem_attr["Attributes"]["AttributesList"]


def ref_key(ref: dict):
    """Deduplication key for a reference entry.
    Uses DOI (preferred), falling back to Title, to identify duplicates
    across the two maps. The EPPI-Reviewer ItemId is map-scoped and
    therefore not suitable as a cross-map key."""
    doi = (ref.get("DOI") or "").strip()
    if doi:
        return ("doi", doi.lower())
    title = (ref.get("Title") or "").strip().lower()
    return ("title", title)


def merge_references(refs_a: list, refs_b: list) -> list:
    """Combine two reference lists, de-duplicating on DOI (then Title).
    AML references take precedence for duplicate entries."""
    seen: dict = {}
    for ref in refs_a:
        k = ref_key(ref)
        seen[k] = ref
    for ref in refs_b:
        k = ref_key(ref)
        if k not in seen:
            seen[k] = ref
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def merge(aml_path: str, ball_path: str, output_path: str) -> None:
    print(f"Loading AML  : {aml_path}")
    aml = load_json(aml_path)

    print(f"Loading B-ALL: {ball_path}")
    ball = load_json(ball_path)

    # Work on a deep copy so the originals are not mutated
    merged = copy.deepcopy(aml)
    merged_cs = merged["CodeSets"][0]

    # Retrieve tumour groups from both files
    aml_groups = get_tumour_groups(aml["CodeSets"][0])
    ball_groups = get_tumour_groups(ball["CodeSets"][0])

    print(f"  AML  tumour groups : {[g['AttributeName'] for g in aml_groups]}")
    print(f"  B-ALL tumour groups: {[g['AttributeName'] for g in ball_groups]}")

    # Append B-ALL tumour groups (deep copies) after AML groups
    merged_haem_attr = merged_cs["Attributes"]["AttributesList"][0]
    for group in ball_groups:
        merged_haem_attr["Attributes"]["AttributesList"].append(copy.deepcopy(group))

    merged_groups = get_tumour_groups(merged_cs)
    print(f"  Merged tumour groups ({len(merged_groups)}):")
    for g in merged_groups:
        types = g.get("Attributes", {}).get("AttributesList", [])
        print(f"    - {g['AttributeName']}  ({len(types)} tumour types)")

    # Merge References
    aml_refs  = aml.get("References", [])
    ball_refs = ball.get("References", [])
    merged["References"] = merge_references(aml_refs, ball_refs)
    print(f"  References: AML={len(aml_refs)}, B-ALL={len(ball_refs)}, "
          f"merged={len(merged['References'])} (de-duplicated on ReferenceId)")

    save_json(merged, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    aml_path    = args[0] if len(args) > 0 else "Report_AML_loe.json"
    ball_path   = args[1] if len(args) > 1 else "Report_B-ALL_loe.json"
    output_path = args[2] if len(args) > 2 else "Report_Haem_merged_loe.json"

    merge(aml_path, ball_path, output_path)