# Analytic file

`analysis.parquet` is the modelling sample (N = 706,921 first-review guests).

- No review text.
- `reviewer_id`, `review_id`, and `listing_id` are SHA-256 prefixes salted for this release. They cannot be joined back to Inside Airbnb identifiers.
- `city` is the Inside Airbnb corpus name (public).
- Outcomes `y_revisit` and `y_continue` are observed in (t0 + 90 days, t0 + 24 months].

Upstream source: Inside Airbnb, [Get the data](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0.

Document-level coder labels are not in this file. Table 9 (`tables/table9_gold_reliability.csv`) is the reliability summary used to support the intention measure.

To rebuild from source CSVs, place city `reviews.csv.gz` files in `DataPaper1/` at the repository root and run `src/p1_pass1_metadata.py` through `src/p1_pass4_measure.py`. That path is not required to reproduce the reported tables from this file.
