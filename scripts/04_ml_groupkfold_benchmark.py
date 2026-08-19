"""
Same information-leakage lesson as crop-yield-ml-benchmark, on new
data: predict tree height from DBH using Random Forest, compared under
naive KFold (folds mix rows from all 24 plots) vs GroupKFold-by-plot
(each fold holds out entire plots the model never saw). Naive KFold
answers "how well do I fit trees I've already seen many neighbors of";
GroupKFold answers the question that actually matters in practice --
"how well would this generalize to a brand new forest plot?"
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "tree_data_clean.csv")

X = df[["dbh"]].values
y = df["height"].values
groups = df["plot"].values

model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)

naive_scores = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
n_groups = df["plot"].nunique()
group_scores = cross_val_score(model, X, y, cv=GroupKFold(min(6, n_groups)), groups=groups, scoring="r2")

print(f"Naive KFold(5) R2:        {naive_scores.mean():.3f} +/- {naive_scores.std():.3f}")
print(f"GroupKFold-by-plot R2:    {group_scores.mean():.3f} +/- {group_scores.std():.3f}")
print(f"\nGap: {naive_scores.mean() - group_scores.mean():.3f}")
print("A large gap means naive CV is optimistic because it lets the model see")
print("other trees from the same plot during training -- it isn't really being")
print("tested on a new, unseen forest site.")

out = BASE / "reports"
pd.DataFrame({
    "scheme": ["naive_kfold", "groupkfold_by_plot"],
    "r2_mean": [naive_scores.mean(), group_scores.mean()],
    "r2_std": [naive_scores.std(), group_scores.std()],
}).to_csv(out / "ml_cv_comparison.csv", index=False)
