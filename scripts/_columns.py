"""
Shared column-name detection: ForestGEO CSV exports use slightly
different header conventions release to release. This maps whatever
the real file uses to a standard internal name, and fails loudly
(rather than silently guessing wrong) if it can't find a confident match.
"""
CANDIDATES = {
    "plot": ["plot", "plotname", "plot_name", "site"],
    "tree_id": ["tag", "treeid", "tree_id", "stemtag", "stem_id"],
    "dbh": ["dbh", "diameter", "dbh_cm", "dbh_mm"],
    "height": ["height", "height_m", "treeheight", "tree_height", "treeht_m"],
    "species": ["species", "sp", "spcode", "latin", "latinbinomial"],
    "year": ["year", "census", "censusid", "date", "yyyy"],
}

def detect_columns(df):
    lower_map = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    found = {}
    missing = []
    for standard, options in CANDIDATES.items():
        match = next((lower_map[o] for o in options if o in lower_map), None)
        if match:
            found[standard] = match
        elif standard in ("dbh", "height", "plot"):
            missing.append(standard)
    if missing:
        raise ValueError(
            f"Could not confidently find columns for: {missing}. "
            f"Actual columns in the file: {list(df.columns)}. "
            f"Run 01_explore_data.py, then add the real column name(s) to "
            f"CANDIDATES in scripts/_columns.py."
        )
    return found
