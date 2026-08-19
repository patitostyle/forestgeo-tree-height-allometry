# Tree Height-DBH Allometry Across 24 Tropical Forest Plots (ForestGEO Panama)

Height-diameter allometric modeling across 24 real ForestGEO forest
plots in Panama, asking a core forestry question: **does the
relationship between tree size measurements vary meaningfully by
site/plot?

## Data source

Smithsonian ForestGEO Program. *Tree height allometry data for 5,109
trees in twenty-four small ForestGEO plots in Panama, 2011-2020.*
Smithsonian Figshare. https://doi.org/10.25573/data.24954204 (CC0/open).

Download: https://smithsonian.figshare.com/ndownloader/articles/24954204/versions/1
(808 KB, ZIP containing a data dictionary and the combined CSV). Extract
`TreeHeightDataSmallPlotsCombined_2024_08_05.csv` into `data/raw/`.

## Methodology

1. **Explore** (`01_explore_data.py`): print the real column names/sample
   -- ForestGEO CSV headers vary slightly by release, so check this
   against `scripts/_columns.py`'s detection logic before trusting
   anything downstream.
2. **Prepare** (`02_prepare_data.py`): standardize columns, drop
   missing/invalid DBH or height.
3. **Classical allometry** (`03_allometric_models.py`): fit the
   power-law height-diameter model (height = a * dbh^b) via log-log
   linear regression, independently per plot and pooled. Real site
   variation should show up as different fitted exponents/R2 across
   plots.
4. **ML benchmark with the leakage lesson from `crop-yield-ml-benchmark`
   applied to new data** (`04_ml_groupkfold_benchmark.py`): Random
   Forest under naive KFold (mixes trees from the same plot across
   train/test) vs GroupKFold-by-plot (entire plots held out) -- the gap
   between them is the honest answer to "how well would this actually
   generalize to a forest plot the model has never seen."

## How to run it

```bash
pip install -r requirements.txt
# download + extract the CSV into data/raw/ (see Data source above)
python tests/run_smoke_test.py    # optional: synthetic logic check, no download needed
python scripts/01_explore_data.py
python scripts/02_prepare_data.py
python scripts/03_allometric_models.py
python scripts/04_ml_groupkfold_benchmark.py
python scripts/05_compare_results.py
```

## Results (real data: 5,107 trees, 24 plots, after dropping 2 rows with missing DBH/height)

Pooled model: `height = 0.833 * dbh^0.533` (R2 = 0.579)

Per-plot exponent (b) ranges from 0.281 (P25) to 0.664 (P18); per-plot
R2 ranges from 0.324 (P25) to 0.840 (P24) -- a wide enough spread to
say plots genuinely differ in their height-diameter relationship, not
just noise.

Random Forest DBH-only model, naive KFold(5) vs GroupKFold-by-plot:

| CV scheme | R2 (mean +/- std) |
|---|---|
| Naive KFold | 0.696 +/- 0.019 |
| GroupKFold-by-plot | 0.671 +/- 0.055 |

Gap = 0.024. Smaller than the dramatic gap shown on synthetic data in
`tests/run_smoke_test.py` (0.654 vs 0.012, engineered on purpose to
make the effect obvious) -- here the model only sees DBH, not plot
identity, so the leakage is real but modest: same-plot trees share
unmeasured site effects (soil, competition, microclimate) that
correlate their DBH-height residuals even without an explicit plot
feature. Honest finding either way: naive CV is optimistic, just not
catastrophically so on this dataset.

See `reports/figures/summary.png` for the full per-plot breakdown.

## Note on this repo's state

Ran end-to-end against the real Smithsonian ForestGEO download (see
Results above). `tests/run_smoke_test.py` remains in the repo as a
fast synthetic-data regression check, not a substitute for the real
run. Column names are detected defensively (`scripts/_columns.py`)
rather than hard-coded -- verify `01_explore_data.py`'s output first
if you rerun this against a different ForestGEO release.

## Stack

Python, pandas, scikit-learn (Random Forest, GroupKFold), matplotlib
