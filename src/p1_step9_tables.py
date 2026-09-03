"""Paper 1 - descriptive tables (corpus, lexicon, descriptives, correlations)."""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK, TAB = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs", "tables")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tpb_lexicon import LEXICON  # noqa: E402

LABEL = {"ATT": "Attitude (ATT)", "SN": "Subjective norm (SN)",
         "PBC": "Perceived behavioural control (PBC)",
         "BI": "Revisit intention (BI)", "SAT": "Satisfaction (SAT)"}

df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
meta = json.load(open(os.path.join(WORK, "panel_meta.json")))
cs = pd.read_csv(os.path.join(WORK, "city_stats.csv"))

# ---- Table 1: corpus construction ------------------------------------
t1 = pd.DataFrame([
    ["Raw reviews in the 123 Inside Airbnb city corpora", f"{cs['n'].sum():,}"],
    ["Cities / destinations", f"{len(cs):,}"],
    ["Calendar coverage", f"{cs.date_min.min()[:10]} to {cs.date_max.max()[:10]}"],
    ["Unique reviewers (estimated from the 5% hash sample)",
     f"{meta['sampled_reviewers']*20:,}"],
    ["Reviewers drawn into the sample (deterministic 5% hash)",
     f"{meta['sampled_reviewers']:,}"],
    ["Reviews written by sampled reviewers", f"{meta['total_sampled_reviews']:,}"],
    ["Focal (first) reviews with a fully observable 24-month window",
     f"{meta['eligible_focal']:,}"],
    ["Retained after English-language and minimum-length screening",
     f"{len(df):,}"],
    ["Analytic sample - cities represented", f"{df.city.nunique():,}"],
    ["Analytic sample - period", f"{df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}"],
    ["Behaviour 1: returned to the same listing (Y1)",
     f"{int(df.y_revisit.sum()):,} ({100*df.y_revisit.mean():.2f}%)"],
    ["Behaviour 2: continued on the platform (Y2)",
     f"{int(df.y_continue.sum()):,} ({100*df.y_continue.mean():.2f}%)"],
    ["Pre-COVID-19 focal reviews",
     f"{int((df.covid==0).sum()):,} ({100*(df.covid==0).mean():.2f}%)"],
    ["During/after COVID-19 focal reviews",
     f"{int((df.covid==1).sum()):,} ({100*(df.covid==1).mean():.2f}%)"],
], columns=["Corpus construction step", "Value"])
t1.to_csv(os.path.join(TAB, "table1_corpus.csv"), index=False)

# ---- Table 2: lexicon -------------------------------------------------
rows = []
for c, pats in LEXICON.items():
    ex = [p.replace("(?:", "(").replace(r"\s", " ") for p in pats[:8]]
    rows.append({"Construct": LABEL[c], "Terms/patterns": len(pats),
                 "Reviews containing >=1 term":
                     f"{int(df[f'pres_{c}'].sum()):,} ({100*df[f'pres_{c}'].mean():.1f}%)",
                 "Mean hits per review": round(float(df[f"hits_{c}"].mean()), 3),
                 "Illustrative entries": "; ".join(ex)})
pd.DataFrame(rows).to_csv(os.path.join(TAB, "table2_lexicon.csv"), index=False)

# ---- Table 3: descriptives -------------------------------------------
spec = [("hits_ATT", "Attitude (keyword count)"), ("hits_SN", "Subjective norm"),
        ("hits_PBC", "Perceived behavioural control"),
        ("hits_BI", "Revisit intention"), ("hits_SAT", "Satisfaction"),
        ("vader_compound", "Sentiment (VADER compound)"),
        ("n_tokens", "Review length (tokens)"),
        ("listing_prior_reviews", "Listing prior reviews at t0"),
        ("listing_age_days", "Listing age at t0 (days)"),
        ("covid", "During COVID-19 (0/1)"),
        ("y_revisit", "Y1 same-listing return (0/1)"),
        ("y_continue", "Y2 platform continuance (0/1)")]
rows = []
for c, lab in spec:
    v = df[c].astype(float)
    rows.append({"Variable": lab, "Mean": round(v.mean(), 4),
                 "SD": round(v.std(), 4), "Min": round(v.min(), 3),
                 "P25": round(v.quantile(.25), 3), "Median": round(v.median(), 3),
                 "P75": round(v.quantile(.75), 3), "Max": round(v.max(), 2)})
pd.DataFrame(rows).to_csv(os.path.join(TAB, "table3_descriptives.csv"), index=False)

# ---- Table 4: correlations -------------------------------------------
cols = ["hits_ATT", "hits_SN", "hits_PBC", "hits_SAT", "hits_BI",
        "vader_compound", "n_tokens", "y_revisit", "y_continue"]
names = ["ATT", "SN", "PBC", "SAT", "BI", "SENT", "LEN", "Y1", "Y2"]
M = df[cols].copy()
for c in ["hits_ATT", "hits_SN", "hits_PBC", "hits_SAT", "hits_BI", "n_tokens"]:
    M[c] = np.log1p(M[c])
C = M.corr(method="spearman").round(3)
C.index = names; C.columns = names
C.to_csv(os.path.join(TAB, "table4_correlations.csv"))

# ---- Table: geographic coverage --------------------------------------
g = (df.groupby("city").agg(n=("y_revisit", "size"), y1=("y_revisit", "mean"),
                            y2=("y_continue", "mean")).sort_values("n", ascending=False))
g["y1"] = (100 * g.y1).round(2); g["y2"] = (100 * g.y2).round(1)
g.head(30).to_csv(os.path.join(TAB, "tableA1_top_cities.csv"))

for f in ["table1_corpus", "table3_descriptives", "table4_correlations"]:
    print("==", f)
    print(pd.read_csv(os.path.join(TAB, f + ".csv")).to_string(index=False))
