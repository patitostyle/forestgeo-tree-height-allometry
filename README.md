# Tree Height-DBH Allometry Across 24 Tropical Forest Plots (ForestGEO Panama)

Adapted from the same idea as a La Molina University master's thesis on
site quality in Peruvian *Guazuma crinita* plantations (which is not
publicly downloadable) -- here applied to real, open, tropical forest
plot data from a different network and country, asking the same kind
of question: **does the relationship between tree size measurements
vary meaningfully by site/plot?**

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
   plots -- this is the direct, open-data analog of "site quality
   affects growth" from the thesis this project takes inspiration from.
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

## Note on this repo's state

This project was scaffolded and logic-tested against **synthetic data
matching the documented ForestGEO column structure** (see
`tests/run_smoke_test.py`) -- not yet run against the real download.
Column names are detected defensively (`scripts/_columns.py`) rather
than hard-coded, but verify `01_explore_data.py`'s output against real
data before trusting the modeling scripts' results.

## Stack

Python, pandas, scikit-learn (Random Forest, GroupKFold), matplotlib
