"""
Run this FIRST after downloading the real data. It prints the actual
column names, dtypes, and a sample of rows -- ForestGEO data dictionaries
vary slightly between releases, so 02_prepare_data.py's column-detection
logic should be checked against this output before trusting downstream
results.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

csv_candidates = list(RAW.glob("*.csv"))
if not csv_candidates:
    print(f"No CSV found in {RAW}. Download the dataset first:")
    print("https://smithsonian.figshare.com/ndownloader/articles/24954204/versions/1")
    raise SystemExit(1)

for path in csv_candidates:
    print(f"=== {path.name} ===")
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(10))
    print("\ndtypes:\n", df.dtypes)
    print("\nmissing values:\n", df.isna().sum())
    print()
