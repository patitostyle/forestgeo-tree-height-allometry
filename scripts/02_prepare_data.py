"""
Load the raw ForestGEO CSV, detect the real column names (see
_columns.py), standardize them, drop rows with missing DBH/height, and
save a clean tabular dataset for the modeling scripts.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _columns import detect_columns

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

csv_candidates = [p for p in RAW.glob("*.csv")]
if not csv_candidates:
    raise SystemExit(f"No CSV in {RAW} -- download the dataset first (see README).")

df = pd.read_csv(csv_candidates[0])
cols = detect_columns(df)
print("Detected columns:", cols)

clean = df.rename(columns={v: k for k, v in cols.items()})
keep = [c for c in ["plot", "tree_id", "dbh", "height", "species", "year"] if c in clean.columns]
clean = clean[keep]

n_before = len(clean)
clean = clean.dropna(subset=["dbh", "height"])
clean = clean[(clean["dbh"] > 0) & (clean["height"] > 0)]
n_after = len(clean)
print(f"Rows: {n_before} -> {n_after} after dropping missing/invalid DBH or height "
      f"({n_before - n_after} removed, {(n_before - n_after) / n_before:.1%})")

print("\nPlots:", clean["plot"].nunique())
print(clean.groupby("plot").size().describe())

clean.to_csv(OUT / "tree_data_clean.csv", index=False)
print(f"\nWritten: {OUT / 'tree_data_clean.csv'}")
