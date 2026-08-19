"""
Logic check on tiny synthetic data mimicking the expected ForestGEO
column structure -- confirms column detection, allometric fitting, and
the GroupKFold benchmark all run before you point them at the real
downloaded CSV.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from _columns import detect_columns

rng = np.random.default_rng(0)
n_plots, n_per_plot = 4, 60
rows = []
for p in range(n_plots):
    dbh = rng.uniform(5, 60, n_per_plot)
    b = rng.uniform(0.55, 0.75)
    height = 1.3 * dbh ** b * rng.normal(1, 0.05, n_per_plot)
    for d, h in zip(dbh, height):
        rows.append({"PlotName": f"plot_{p}", "Tag": rng.integers(1e5), "DBH": d, "Height": h})
df = pd.DataFrame(rows)

cols = detect_columns(df)
assert set(cols.keys()) >= {"plot", "dbh", "height"}, cols
print("OK: column detection works on synthetic ForestGEO-style headers")

clean = df.rename(columns={v: k for k, v in cols.items()})
from sklearn.linear_model import LinearRegression
X = np.log(clean[["dbh"]].values)
y = np.log(clean["height"].values)
m = LinearRegression().fit(X, y)
assert 0.3 < m.coef_[0] < 1.2
print("OK: log-log allometric fit produces a sane exponent")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
Xf = clean[["dbh"]].values
yf = clean["height"].values
groups = clean["plot"].values
scores = cross_val_score(RandomForestRegressor(n_estimators=50, random_state=0), Xf, yf,
                          cv=GroupKFold(4), groups=groups, scoring="r2")
assert len(scores) == 4
print("OK: GroupKFold benchmark runs end-to-end")

print("\nALL SMOKE TESTS PASSED (synthetic data -- run with real data for actual results)")
