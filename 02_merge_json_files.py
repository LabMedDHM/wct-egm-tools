"""
02_merge_json_files.py
==================
Merges two WCT EVI MAP coding JSON files (AML and B-ALL/LBL) into a single
Haematolymphoid Tumours JSON.

Merge logic
-----------
- The top-level CodeSets wrapper and the shared attribute axes
  (Characteristics, Study design, Level of Evidence) are taken from the
  AML file (primary source).
- The Haematolymphoid Tumours tumour-group & tumour-type hierarchy from B-ALL
  is appended after the AML hierarchy under the same parent node.
- B-ALL references are remapped before merging: each code AttributeId that
  corresponds to a shared axis (Characteristics, Study design, LoE) is
  translated to its AML equivalent by matching on AttributeName. B-ALL
  tumour type codes are left untouched because those AttributeIds are
  present in the merged codeset (appended from the B-ALL hierarchy).
- References from both files are concatenated without de-duplication. A
  study coded for both an AML and a B-ALL tumour type legitimately appears
  in both files and must be retained as two separate records, consistent
  with how 01_study_design_to_loe.py splits studies coded for multiple 
  characteristics or tumour types into one record each.
- All other metadata (SetName, SetId, etc.) is preserved from the AML file.

Usage
-----
    python 02_merge_json_files.py [AML_JSON] [BALL_JSON] [OUTPUT_JSON]

Defaults
--------
    AML_JSON  : Report_AML_loe.json
    BALL_JSON : Report_B-ALL_loe.json
    OUTPUT_JSON: Report_Haem_merged_loe.json
"""

import json
import sys
import copy
 
 
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
 
 
def collect_attr_ids(data, result: dict) -> None:
    """Recursively collect all AttributeId -> AttributeName pairs
    found anywhere in data (dict or list)."""
    if isinstance(data, dict):
        if "AttributeId" in data and "AttributeName" in data:
            result[data["AttributeId"]] = data["AttributeName"]
        for v in data.values():
            collect_attr_ids(v, result)
    elif isinstance(data, list):
        for item in data:
            collect_attr_ids(item, result)
 
 
def build_remap_table(ball_codeset: dict, aml_codeset: dict) -> dict:
    """Build a mapping from B-ALL AttributeId to AML AttributeId for all
    codes that exist in both codesets under the same AttributeName (i.e.
    the shared axes: Characteristics, Study design, LoE).
 
    B-ALL tumour type codes that have no AML counterpart are omitted from
    the table; they are handled correctly already because the merged codeset
    retains their original B-ALL AttributeIds.
 
    Args:
        ball_codeset: The B-ALL CodeSets[0] dict.
        aml_codeset:  The AML CodeSets[0] dict.
 
    Returns:
        dict mapping B-ALL AttributeId (int) -> AML AttributeId (int).
    """
    ball_ids: dict = {}
    aml_ids: dict = {}
    collect_attr_ids(ball_codeset, ball_ids)
    collect_attr_ids(aml_codeset, aml_ids)
 
    # Reverse lookup: AttributeName -> AttributeId for AML
    aml_name_to_id = {name: aid for aid, name in aml_ids.items()}
 
    remap: dict = {}
    unmapped = []
    for ball_id, name in ball_ids.items():
        if name in aml_name_to_id:
            aml_id = aml_name_to_id[name]
            if ball_id != aml_id:
                remap[ball_id] = aml_id
        else:
            unmapped.append((ball_id, name))
 
    print(f"  ID remap table: {len(remap)} shared codes remapped")
    print(f"  B-ALL-only codes (kept as-is): {len(unmapped)}")
    for bid, name in unmapped:
        print(f"    {bid} -> {name}")
 
    return remap
 
 
def remap_reference_codes(refs: list, remap: dict) -> list:
    """Return a deep copy of refs with each code's AttributeId
    translated via remap where applicable.
 
    Args:
        refs:  List of reference dicts from the B-ALL JSON.
        remap: Mapping from B-ALL AttributeId to AML AttributeId.
 
    Returns:
        New list of reference dicts with remapped AttributeIds.
    """
    remapped_refs = []
    for ref in refs:
        new_ref = copy.deepcopy(ref)
        for code in new_ref.get("Codes", []):
            old_id = code["AttributeId"]
            if old_id in remap:
                code["AttributeId"] = remap[old_id]
        remapped_refs.append(new_ref)
    return remapped_refs
 
 
# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------
 
def merge(aml_path: str, ball_path: str, output_path: str) -> None:
    print(f"Loading AML  : {aml_path}")
    aml = load_json(aml_path)
 
    print(f"Loading B-ALL: {ball_path}")
    ball = load_json(ball_path)
 
    # Work on a deep copy of AML as the base for the merged output
    merged = copy.deepcopy(aml)
    merged_cs = merged["CodeSets"][0]
 
    # Retrieve tumour groups from both files
    aml_groups  = get_tumour_groups(aml["CodeSets"][0])
    ball_groups = get_tumour_groups(ball["CodeSets"][0])
 
    print(f"  AML  tumour groups : {[g['AttributeName'] for g in aml_groups]}")
    print(f"  B-ALL tumour groups: {[g['AttributeName'] for g in ball_groups]}")
 
    # Append B-ALL tumour groups (deep copies) after the AML groups
    merged_haem_attr = merged_cs["Attributes"]["AttributesList"][0]
    for group in ball_groups:
        merged_haem_attr["Attributes"]["AttributesList"].append(copy.deepcopy(group))
 
    merged_groups = get_tumour_groups(merged_cs)
    print(f"  Merged tumour groups ({len(merged_groups)}):")
    for g in merged_groups:
        types = g.get("Attributes", {}).get("AttributesList", [])
        print(f"    - {g['AttributeName']}  ({len(types)} tumour types)")
 
    # Build the ID remap table and translate B-ALL reference codes
    print()
    print("Building AttributeId remap table ...")
    remap = build_remap_table(ball["CodeSets"][0], aml["CodeSets"][0])
 
    print()
    print("Remapping B-ALL reference codes ...")
    ball_refs_remapped = remap_reference_codes(ball.get("References", []), remap)
 
    # Concatenate reference lists without de-duplication: a study coded for
    # both an AML and a B-ALL tumour type legitimately appears in both files
    # and must be retained as two separate records.
    aml_refs = aml.get("References", [])
    merged["References"] = aml_refs + ball_refs_remapped
    print()
    print(f"  References: AML={len(aml_refs)}, B-ALL={len(ball_refs_remapped)}, "
          f"merged total={len(merged['References'])}")
 
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
 