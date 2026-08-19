"""
Classical forestry allometry: fit the power-law height-diameter model
  height = a * dbh^b
via log-log linear regression (log(height) = log(a) + b*log(dbh)),
independently per plot AND pooled across all 24 plots. If site
conditions (light, competition, soil) genuinely differ between plots --
the same idea as "site quality" in the thesis this project takes
inspiration from -- the fitted b (and R2) should vary meaningfully
plot-to-plot instead of being identical everywhere.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "tree_data_clean.csv")

def fit_loglog(sub):
    X = np.log(sub[["dbh"]].values)
    y = np.log(sub["height"].values)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return {
        "n": len(sub),
        "b_slope": model.coef_[0],
        "log_a_intercept": model.intercept_,
        "a": np.exp(model.intercept_),
        "r2": r2_score(y, pred),
    }

rows = []
for plot, sub in df.groupby("plot"):
    if len(sub) < 20:
        continue
    r = fit_loglog(sub)
    r["plot"] = plot
    rows.append(r)

per_plot = pd.DataFrame(rows).sort_values("r2", ascending=False)
pooled = fit_loglog(df)
pooled["plot"] = "POOLED (all plots)"

print("Pooled (all plots combined):")
print(f"  height = {pooled['a']:.3f} * dbh^{pooled['b_slope']:.3f}   (R2 = {pooled['r2']:.3f})")
print()
print("Per-plot allometry:")
print(per_plot[["plot", "n", "a", "b_slope", "r2"]].to_string(index=False))

print(f"\nRange of b (exponent) across plots: {per_plot['b_slope'].min():.3f} to {per_plot['b_slope'].max():.3f}")
print(f"Range of R2 across plots: {per_plot['r2'].min():.3f} to {per_plot['r2'].max():.3f}")
print("If these ranges are wide, plots genuinely differ in their height-diameter")
print("relationship (consistent with site-quality effects); if narrow, one pooled")
print("model is about as good as 24 separate ones.")

out = BASE / "reports"
out.mkdir(exist_ok=True)
per_plot.to_csv(out / "allometry_per_plot.csv", index=False)
pd.DataFrame([pooled]).to_csv(out / "allometry_pooled.csv", index=False)
