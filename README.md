# Sealed-window TPB (Paper 1A) — replication package

Replication materials for:

> *Stated Intention Predicts Observed Return, Two Years On: A sealed-window test of the theory of planned behaviour on 73.9 million guest reviews across 123 accommodation markets.*

This repository covers **Paper 1A only** (intention → observed behaviour under a sealed measurement window). Measurement-instrument diagnostics belong to the companion paper (1B) and are not included.

Target journal: *International Journal of Contemporary Hospitality Management*.

## What is in this package

| Path | Contents |
|---|---|
| `src/` | Pipeline: metadata → panel → text → lexicon scores → GSEM, out-of-time ML, SHAP, DML, tables, figures, three GSEM robustness specs |
| `data/analysis.parquet` | Analytic sample (N = 706,921). Keyword counts, VADER, outcomes, listing context. **No review text.** IDs are re-hashed |
| `tables/` | CSV sources for manuscript Tables 1–12 |
| `figures/` | Manuscript Figures 1–4 |
| `notebooks/Paper1A_Robustness_GSEM.ipynb` | Re-runs the three robustness specifications and the number audit |

Raw Inside Airbnb review CSVs (~22 GB) are **not** distributed. They remain available from [Inside Airbnb](https://insideairbnb.com/get-the-data/) under CC BY 4.0.

## Reproduce reported tables from the analytic file

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p work
cp data/analysis.parquet work/analysis.parquet

python src/p1_step5_gsem.py
python src/p1_step6_ml.py
python src/p1_step7_xai.py
python src/p1_step8_causal.py
python src/p1a_robustness_gsem.py
```

Outputs are written to `outputs/tables/` (created by the scripts) relative to the repository root.

The destination-fixed-effects logit for same-listing return (Table 12, Y1) is estimated by **LBFGS**. Newton–Raphson does not converge under rare events plus city dummies; do not replace LBFGS with the default Newton solver for that specification.

## Rebuild from Inside Airbnb (optional)

1. Download city-level `reviews.csv.gz` files into `DataPaper1/`.
2. Run passes 1–4, then the analysis steps above:

```bash
python src/p1_pass1_metadata.py
python src/p1_pass2_panel.py
python src/p1_pass3_text.py
python src/p1_pass4_measure.py
```

The sample is a deterministic 5% hash of reviewer identifiers, first-review t₀, 90-day blackout, 24-month window, right-censoring at 15 June 2026. Constants live at the top of `src/p1_pass2_panel.py`.

## Mint a DOI (Zenodo)

1. Push this folder to a **public** GitHub repository (do not upload `DataPaper1/` or unhashed IDs).
2. Sign in to [Zenodo](https://zenodo.org) with GitHub.
3. *Account → GitHub → Enable* the repository.
4. Create a GitHub Release (e.g. `v1.0.0`).
5. Zenodo mints a DOI automatically. Paste that DOI into the manuscript data-availability statement.

Cite the software/data release separately from the article once the DOI exists.

## Licence

- Code: MIT (`LICENSE`).
- Analytic file: derived from Inside Airbnb data (CC BY 4.0). Retain that attribution in any reuse.
- We do not redistribute review text.

## Paper 1A findings this package supports

- Intention → Y1: β = 0.2339 (SE 0.0107); intention → Y2: β = 0.0482 (SE 0.0029).
- Robustness (BI → Y1): 0.2386 (drop 2020–21), 0.2453 (top 30 cities), 0.2376 (city FE, LBFGS).
- DML (n = 200,000): +0.84 pp same-listing return per 1 SD of intention.
- Counterfactual Table 7 **source CSV**: attitude moves 20.65% of guests (not 20.78).
