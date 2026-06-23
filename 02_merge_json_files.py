"""
02_merge_json_files.py
======================
Merges all WCT EVI MAP coding JSON files belonging to a single WHO Blue Book
(WHO Classification of Tumours) into one combined JSON.

Merge logic
-----------
- The first input file is the primary. The merged output is built as a deep
  copy of it, so its CodeSets wrapper and shared attribute axes
  (Characteristics, Study design, Level of Evidence) become the canonical axes.
- Every other (secondary) file contributes its tumour-group and tumour-type
  hierarchy, which is appended under the primary's Blue-Book tumour node.
- Secondary reference codes are remapped before merging: each code
  AttributeId that corresponds to a shared axis is translated to its primary
  equivalent by matching on AttributeName. Tumour-type codes have no primary
  counterpart and are kept as-is, since the merged codeset retains their
  original AttributeIds via the appended hierarchy.
- References from all files are concatenated without de-duplication. A study
  coded for tumour types in more than one file legitimately appears in each
  file and must be retained as separate records, consistent with how
  01_automated_coding_loe.py splits studies coded for multiple characteristics
  or tumour types into one record each.

Safety checks
-------------
- The Blue-Book tumour attribute is located by name (not by position), so the
  script does not rely on a fixed ordering of the attribute axes.
- An AttributeId collision guard warns if a kept (tumour-type) ID from a
  secondary file clashes with a different code already present in the merge.
- A post-merge integrity check confirms every reference code resolves to a
  name in the merged codeset, catching any unmapped shared-axis code.

Usage
-----
    # Folder mode: merge every *_loe.json in a folder (first alphabetically
    # is the primary). This matches the one-folder-per-Blue-Book layout.
    python 02_merge_json_files.py <folder> [--output OUTPUT_JSON]

    # Explicit mode: list the files in the exact order you want, primary first.
    python 02_merge_json_files.py --files Report_tumour_group1_loe.json Report_tumour_group2_loe.json [--output OUTPUT_JSON]

If --output is omitted, the merged file is written next to the inputs as
Report_merged_loe.json.
"""

import argparse
import copy
import json
import os
import sys

# Attribute axes that are shared across all files of a Blue Book and must be
# remapped (rather than appended). The tumour attribute is everything else.
CHARACTERISTICS = "Characteristics"
STUDY_DESIGN = "Study design"
LOE_TOKEN = "LoE"  # matches "Level of Evidence (LoE)"

MERGED_SUFFIX = "_merged_loe.json"
LOE_SUFFIX = "_loe.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_error_msg(error_msg: str) -> None:
    """Print a coloured error message to the command-line."""
    print(f"\033[91m{error_msg}\033[0m")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"Saved merged JSON to: {path}")


def is_shared_axis(name: str) -> bool:
    """Whether an attribute name is one of the shared axes
    (Characteristics, Study design, Level of Evidence)."""
    return name in (CHARACTERISTICS, STUDY_DESIGN) or LOE_TOKEN in name


def get_tumour_attr(codeset: dict) -> dict:
    """Return the Blue-Book tumour attribute (the attribute that is not one of
    the shared axes). Located by name so the script does not depend on a fixed
    ordering of the attribute list."""
    attrs = codeset["Attributes"]["AttributesList"]
    tumour_attrs = [a for a in attrs if not is_shared_axis(a["AttributeName"])]
    if len(tumour_attrs) != 1:
        names = [a["AttributeName"] for a in attrs]
        raise ValueError(
            "Expected exactly one tumour attribute axis, found "
            f"{len(tumour_attrs)} in attributes: {names}"
        )
    return tumour_attrs[0]


def get_tumour_groups(codeset: dict) -> list:
    """Return the list of tumour-group dicts nested inside the Blue-Book
    tumour attribute of a given CodeSet entry."""
    return get_tumour_attr(codeset)["Attributes"]["AttributesList"]


def collect_attr_ids(data, result: dict) -> None:
    """Recursively collect all AttributeId -> AttributeName pairs found
    anywhere in data (dict or list)."""
    if isinstance(data, dict):
        if "AttributeId" in data and "AttributeName" in data:
            result[data["AttributeId"]] = data["AttributeName"]
        for v in data.values():
            collect_attr_ids(v, result)
    elif isinstance(data, list):
        for item in data:
            collect_attr_ids(item, result)


def build_remap_table(secondary_codeset: dict, primary_codeset: dict) -> dict:
    """Build a mapping from secondary AttributeId to primary AttributeId for
    all codes that exist in both codesets under the same AttributeName (i.e.
    the shared axes: Characteristics, Study design, LoE, and the Blue-Book
    tumour node itself).

    Secondary tumour-type codes that have no primary counterpart are omitted
    from the table; they are handled correctly already because the merged
    codeset retains their original AttributeIds.

    Returns:
        dict mapping secondary AttributeId -> primary AttributeId.
    """
    secondary_ids: dict = {}
    primary_ids: dict = {}
    collect_attr_ids(secondary_codeset, secondary_ids)
    collect_attr_ids(primary_codeset, primary_ids)

    # Reverse lookup: AttributeName -> AttributeId for the primary file.
    primary_name_to_id = {name: aid for aid, name in primary_ids.items()}

    remap: dict = {}
    unmapped = []
    for sec_id, name in secondary_ids.items():
        if name in primary_name_to_id:
            primary_id = primary_name_to_id[name]
            if sec_id != primary_id:
                remap[sec_id] = primary_id
        else:
            unmapped.append((sec_id, name))

    print(f"    shared codes remapped: {len(remap)}")
    print(f"    file-only codes kept as-is: {len(unmapped)}")
    for sid, name in unmapped:
        print(f"      {sid} -> {name}")

    return remap


def remap_reference_codes(refs: list, remap: dict) -> list:
    """Return a deep copy of refs with each code's AttributeId translated via
    remap where applicable."""
    remapped_refs = []
    for ref in refs:
        new_ref = copy.deepcopy(ref)
        for code in new_ref.get("Codes", []):
            old_id = code["AttributeId"]
            if old_id in remap:
                code["AttributeId"] = remap[old_id]
        remapped_refs.append(new_ref)
    return remapped_refs


def check_id_collisions(
    merged_codeset: dict, group: dict, source_label: str
) -> None:
    """Warn if any AttributeId in an incoming tumour group already exists in
    the merged codeset under a different name. With distinct EPPI-Reviewer ID
    ranges this should never fire, but it guards against silent corruption."""
    existing: dict = {}
    collect_attr_ids(merged_codeset, existing)
    incoming: dict = {}
    collect_attr_ids(group, incoming)
    for aid, name in incoming.items():
        if aid in existing and existing[aid] != name:
            print_error_msg(
                f"    WARNING: AttributeId {aid} from {source_label} "
                f"('{name}') collides with existing '{existing[aid]}'"
            )


def check_reference_integrity(merged: dict) -> bool:
    """Confirm every reference code AttributeId resolves to a name in the
    merged codeset. Returns True if all codes resolve."""
    known: dict = {}
    collect_attr_ids(merged["CodeSets"][0], known)
    orphan_ids = set()
    for ref in merged.get("References", []):
        for code in ref.get("Codes", []):
            if code["AttributeId"] not in known:
                orphan_ids.add(code["AttributeId"])
    if orphan_ids:
        print_error_msg(
            f"  Integrity check FAILED: {len(orphan_ids)} reference code IDs "
            f"do not resolve in the merged codeset: {sorted(orphan_ids)}"
        )
        return False
    print("  Integrity check passed: all reference codes resolve in the merged codeset.")
    return True


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def merge(input_paths: list, output_path: str) -> None:
    if len(input_paths) < 2:
        print_error_msg("Need at least two input files to merge.")
        sys.exit(1)

    primary_path = input_paths[0]
    print(f"Primary file : {primary_path}")
    primary = load_json(primary_path)
    primary_cs = primary["CodeSets"][0]

    merged = copy.deepcopy(primary)
    merged_cs = merged["CodeSets"][0]
    merged_tumour_attr = get_tumour_attr(merged_cs)

    print(f"  Blue-Book node: '{merged_tumour_attr['AttributeName']}'")
    primary_groups = get_tumour_groups(primary_cs)
    print(f"  primary tumour groups: {[g['AttributeName'] for g in primary_groups]}")

    merged_refs = list(primary.get("References", []))
    ref_counts = [(os.path.basename(primary_path), len(merged_refs))]

    for secondary_path in input_paths[1:]:
        print()
        print(f"Merging in   : {secondary_path}")
        secondary = load_json(secondary_path)
        secondary_cs = secondary["CodeSets"][0]

        secondary_groups = get_tumour_groups(secondary_cs)
        print(f"  tumour groups: {[g['AttributeName'] for g in secondary_groups]}")

        # Append secondary tumour groups (deep copies) under the merged node.
        for group in secondary_groups:
            check_id_collisions(merged_cs, group, os.path.basename(secondary_path))
            merged_tumour_attr["Attributes"]["AttributesList"].append(
                copy.deepcopy(group)
            )

        # Remap secondary shared-axis codes to the primary's IDs, then append.
        remap = build_remap_table(secondary_cs, primary_cs)
        secondary_refs = remap_reference_codes(
            secondary.get("References", []), remap
        )
        merged_refs += secondary_refs
        ref_counts.append((os.path.basename(secondary_path), len(secondary_refs)))

    # Concatenate without de-duplication (see module docstring).
    merged["References"] = merged_refs

    print()
    merged_groups = get_tumour_groups(merged_cs)
    print(f"Merged tumour groups ({len(merged_groups)}):")
    for g in merged_groups:
        types = g.get("Attributes", {}).get("AttributesList", [])
        print(f"  - {g['AttributeName']}  ({len(types)} tumour types)")

    print()
    print("References per file:")
    for name, count in ref_counts:
        print(f"  {name}: {count}")
    print(f"  merged total: {len(merged_refs)}")

    print()
    check_reference_integrity(merged)

    print()
    save_json(merged, output_path)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def discover_loe_files(folder: str) -> list:
    """Return sorted *_loe.json files in folder, excluding any previously
    produced *_merged_loe.json output."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(LOE_SUFFIX) and not f.endswith(MERGED_SUFFIX)
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all WCT EVI MAP JSON files of one Blue Book into one JSON."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing the *_loe.json files for one Blue Book.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of input files, primary first (overrides folder).",
    )
    parser.add_argument(
        "--output",
        help="Path of the merged output JSON.",
    )
    args = parser.parse_args()

    if args.files:
        input_paths = args.files
        default_dir = os.path.dirname(os.path.abspath(input_paths[0]))
    elif args.folder:
        input_paths = discover_loe_files(args.folder)
        default_dir = args.folder
        if len(input_paths) < 2:
            print_error_msg(
                f"Found {len(input_paths)} *_loe.json file(s) in {args.folder}; "
                "need at least two."
            )
            sys.exit(1)
    else:
        parser.error("Provide a folder or --files.")

    for path in input_paths:
        if not (os.path.isfile(path) and path.endswith(".json")):
            print_error_msg(f"Invalid input file: {path}")
            sys.exit(1)

    output_path = args.output or os.path.join(default_dir, "Report_merged_loe.json")

    print(f"Discovered {len(input_paths)} input file(s):")
    for path in input_paths:
        print(f"  {path}")
    print()

    merge(input_paths, output_path)


if __name__ == "__main__":
    main()