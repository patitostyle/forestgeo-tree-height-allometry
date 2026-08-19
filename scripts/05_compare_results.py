"""Consolidate allometry and ML-benchmark results into one summary + plot."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

per_plot = pd.read_csv(REPORTS / "allometry_per_plot.csv")
cv = pd.read_csv(REPORTS / "ml_cv_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(per_plot["plot"], per_plot["b_slope"])
axes[0].set_title("Height-DBH allometric exponent (b) by plot")
axes[0].set_ylabel("b")
axes[0].tick_params(axis="x", rotation=90, labelsize=7)

axes[1].bar(cv["scheme"], cv["r2_mean"], yerr=cv["r2_std"])
axes[1].set_title("R2: naive KFold vs GroupKFold-by-plot")
axes[1].set_ylabel("R2")

fig.tight_layout()
fig.savefig(FIGS / "summary.png", dpi=120)
print(f"Written: {FIGS / 'summary.png'}")
print("\n", per_plot[["plot", "b_slope", "r2"]].to_string(index=False))
print("\n", cv.to_string(index=False))
