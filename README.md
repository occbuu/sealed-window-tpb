# Sealed-window TPB — replication package

Replication materials for:

> Le Ngoc, H. (2026). *Stated intention predicts observed return, two years on: A sealed-window test of the theory of planned behaviour on 73.9 million guest reviews across 123 accommodation markets.* Working paper.

This package supports a sealed-window test of text-derived revisit intention (frozen at the first review t₀) against observed return in a later window.

## Cite this package

If you use the code or `data/analysis.parquet`, cite both the working paper and this release:

```
Le Ngoc, H. (2026). Sealed-window TPB replication package (v1.1.0)
[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22274705
```

GitHub also reads `CITATION.cff` for the “Cite this repository” button.

## What is in this package

| Path | Contents |
|---|---|
| `src/` | Pipeline: metadata → panel → text → lexicon scores → GSEM, out-of-time ML, SHAP, DML, tables, figures, three GSEM robustness specs |
| `data/analysis.parquet` | Analytic sample (N = 706,921). Keyword counts, VADER, outcomes, listing context. **No review text.** IDs are re-hashed |
| `tables/` | CSV sources for manuscript Tables 1–13, including Table 9 (gold-standard reliability of the intention measure) |
| `figures/` | Manuscript Figures 1–4 |
| `notebooks/Paper1A_Robustness_GSEM.ipynb` | Re-runs the three robustness specifications and the number audit |

Raw Inside Airbnb review CSVs (~22 GB) are **not** distributed. They remain available from [Inside Airbnb](https://insideairbnb.com/get-the-data/) under CC BY 4.0. Document-level coder labels are not redistributed; Table 9 is the reliability summary used in the manuscript.

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

Outputs are written to `outputs/tables/` relative to the repository root.

The destination-fixed-effects logit for same-listing return (Y1) is estimated by **LBFGS**. Newton–Raphson does not converge under rare events plus city dummies; do not replace LBFGS with the default Newton solver for that specification.

## Rebuild from Inside Airbnb (optional)

1. Download city-level `reviews.csv.gz` files into `DataPaper1/`.
2. Run passes 1–4, then the analysis steps above.

```bash
python src/p1_pass1_metadata.py
python src/p1_pass2_panel.py
python src/p1_pass3_text.py
python src/p1_pass4_measure.py
```

The sample is a deterministic hash of reviewer identifiers, first-review t₀, 90-day blackout, 24-month window, right-censoring at 15 June 2026. Constants live at the top of `src/p1_pass2_panel.py`.

## Headline numbers this package supports

- Intention → Y1: β = 0.2339 (SE 0.0107); intention → Y2: β = 0.0482 (SE 0.0029).
- Robustness (BI → Y1): 0.2386 (drop 2020–21), 0.2453 (top 30 cities), 0.2376 (city FE, LBFGS).
- DML (n = 200,000): +0.84 pp same-listing return per 1 SD of intention.
- Gold-standard intention (Table 9): Krippendorff's α = 0.873; dictionary precision 0.850, recall 0.539.
- Counterfactual Table 7 **source CSV**: attitude moves 20.65% of guests (not 20.78).

## Licence

- Code: MIT (`LICENSE`).
- Analytic file: derived from Inside Airbnb data (CC BY 4.0). Retain that attribution in any reuse.
- We do not redistribute review text.

## Contact

Hieu Le Ngoc ([ORCID 0000-0002-1133-1433](https://orcid.org/0000-0002-1133-1433)), Posts and Telecommunications Institute of Technology, Ho Chi Minh City Campus. Email: lnhieu@ptit.edu.vn
