"""
Three GSEM robustness specifications:

  R1  drop first-review cohorts 2020–2021 (COVID years)
  R2  keep the 30 largest destinations by analytic N
  R3  destination fixed effects on a 150,000-row subsample (no Mundlak means)

The primary count specification of p1_step5_gsem.py is reused: z(log1p(hits)),
city-clustered standard errors, year dummies. R3 replaces Mundlak city means
with city dummies and drops cities with no outcome variation (complete
separation under logit FE).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "work")
if not os.path.isfile(os.path.join(WORK, "analysis.parquet")):
    WORK = os.path.join(ROOT, "data")
TAB = os.path.join(ROOT, "outputs", "tables")
OUT = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

CONS = ["ATT", "SN", "PBC", "SAT"]
ALL = CONS + ["BI"]
SEED = 20260901
NSUB_FE = 150_000

LABELS = {
    "BI": "Revisit intention (BI)",
    "ATT": "Attitude",
    "SN": "Subjective norm",
    "PBC": "Perceived behavioural control",
    "SAT": "Satisfaction",
    "SENT": "Sentiment (VADER)",
    "covid": "During COVID-19",
    "log_len": "Review length (log)",
    "log_prior": "Listing prior reviews (log)",
    "log_age": "Listing age (log)",
    "const": "Constant",
}


def z(v):
    v = np.asarray(v, dtype="float64")
    return (v - v.mean()) / v.std()


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def fmt(c, s, p):
    return f"{c:.4f}{stars(p)} ({s:.4f})"


def build(df, mundlak=True):
    X = pd.DataFrame(index=df.index)
    for c in ALL:
        X[c] = z(np.log1p(df[f"hits_{c}"]))
    X["SENT"] = z(df["vader_compound"])
    X["log_len"] = z(np.log1p(df["n_tokens"]))
    X["log_prior"] = z(np.log1p(df["listing_prior_reviews"]))
    X["log_age"] = z(np.log1p(np.clip(df["listing_age_days"], 0, None)))
    X["covid"] = df["covid"].astype("float64").to_numpy()
    if mundlak:
        cm = df.groupby("city_id")[[f"hits_{c}" for c in ALL]].transform("mean")
        for c in ALL:
            X[f"cm_{c}"] = z(cm[f"hits_{c}"])
    yr = pd.get_dummies(df["year"].astype(int), prefix="yr",
                        drop_first=True).astype("float64")
    return pd.concat([X, yr.set_index(X.index)], axis=1)


def fit_pair(df, ycol, mundlak=True, city_fe=False, _retry=0):
    work = df.copy()
    if city_fe:
        vc = work.groupby("city_id")[ycol].nunique()
        keep = vc[vc >= 2].index
        work = work[work["city_id"].isin(keep)].copy()
    X = build(work, mundlak=mundlak and not city_fe)
    if city_fe:
        dum = pd.get_dummies(work["city_id"].astype(int), prefix="city",
                             drop_first=True).astype("float64")
        X = pd.concat([X, dum.set_index(X.index)], axis=1)
    g = work["city_id"].to_numpy()
    y = work[ycol].to_numpy(dtype="float64")
    X1 = sm.add_constant(X.drop(columns=["BI"]), has_constant="add")
    m1 = sm.OLS(X["BI"].to_numpy(), X1.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    X2 = sm.add_constant(X, has_constant="add")
    fit_kw = {"disp": 0, "maxiter": 300}
    if city_fe:
        fit_kw["method"] = "lbfgs"
    try:
        w = sm.Logit(y, X2.to_numpy()).fit(**fit_kw)
    except Exception:
        if _retry >= 2:
            raise
        need = 5 if _retry else 2
        pos = work.groupby("city_id")[ycol].sum()
        keep = pos[pos >= need].index
        work = work[work["city_id"].isin(keep)].copy()
        return fit_pair(work, ycol, mundlak=mundlak, city_fe=city_fe,
                        _retry=_retry + 1)
    ret = w.mle_retvals or {}
    if city_fe and not ret.get("converged", False):
        raise RuntimeError(
            f"logit {ycol} did not converge (method={fit_kw.get('method', 'newton')}, "
            f"warnflag={ret.get('warnflag')}). Do not save this table.")
    m2 = sm.Logit(y, X2.to_numpy()).fit(
        start_params=w.params, cov_type="cluster", cov_kwds={"groups": g},
        **fit_kw)
    return m1, list(X1.columns), m2, list(X2.columns), work


def coef_table(m1, n1, m2, n2):
    i1 = {n: k for k, n in enumerate(n1)}
    i2 = {n: k for k, n in enumerate(n2)}
    rows = [{"variable": LABELS["BI"], "eq1_BI": "",
             "eq2_behaviour": fmt(m2.params[i2["BI"]], m2.bse[i2["BI"]],
                                  m2.pvalues[i2["BI"]])}]
    for c in ["ATT", "SN", "PBC", "SAT", "SENT", "covid",
              "log_len", "log_prior", "log_age", "const"]:
        r = {"variable": LABELS[c], "eq1_BI": "", "eq2_behaviour": ""}
        if c in i1:
            r["eq1_BI"] = fmt(m1.params[i1[c]], m1.bse[i1[c]], m1.pvalues[i1[c]])
        if c in i2:
            r["eq2_behaviour"] = fmt(m2.params[i2[c]], m2.bse[i2[c]],
                                     m2.pvalues[i2[c]])
        rows.append(r)
    return pd.DataFrame(rows), i1, i2


def key_paths(m1, i1, m2, i2):
    def grab(model, idx, name):
        k = idx[name]
        return fmt(model.params[k], model.bse[k], model.pvalues[k])
    return {
        "BI->Y": grab(m2, i2, "BI"),
        "ATT->BI": grab(m1, i1, "ATT"),
        "SN->BI": grab(m1, i1, "SN"),
        "PBC->BI": grab(m1, i1, "PBC"),
        "SAT->BI": grab(m1, i1, "SAT"),
        "ATT->Y": grab(m2, i2, "ATT"),
        "SN->Y": grab(m2, i2, "SN"),
        "PBC->Y": grab(m2, i2, "PBC"),
        "SAT->Y": grab(m2, i2, "SAT"),
    }


def save_both(df, name):
    p1 = os.path.join(OUT, name)
    p2 = os.path.join(TAB, name)
    df.to_csv(p1, index=False)
    df.to_csv(p2, index=False)
    return p1


def run_spec(df, label, ycol, **fit_kw):
    t0 = time.time()
    m1, n1, m2, n2, used = fit_pair(df, ycol, **fit_kw)
    tbl, i1, i2 = coef_table(m1, n1, m2, n2)
    keys = key_paths(m1, i1, m2, i2)
    meta = {
        "spec": label, "outcome": ycol, "N": int(len(used)),
        "cities": int(used["city_id"].nunique()),
        "base_rate": round(float(used[ycol].mean()), 5),
        "seconds": round(time.time() - t0, 1),
        **keys,
    }
    print(json.dumps({k: meta[k] for k in
                      ("spec", "outcome", "N", "cities", "base_rate",
                       "seconds", "BI->Y")}, default=str),
          flush=True)
    return tbl, meta


def main():
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    print(f"analytic N = {len(df):,}", flush=True)

    covid = df[~df["year"].isin([2020, 2021])].copy()
    top30 = (df.groupby("city").size().sort_values(ascending=False)
               .head(30).index)
    big = df[df["city"].isin(top30)].copy()
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(df), size=NSUB_FE, replace=False)
    sub = df.iloc[take].copy()

    specs = [
        ("R1 exclude 2020–2021 cohorts", covid, True, False,
         "table_robust_covid"),
        ("R2 thirty largest destinations", big, True, False,
         "table_robust_top30"),
        ("R3 destination FE, n=150,000", sub, False, True,
         "table_robust_cityfe"),
    ]
    summary = []
    for label, sample, mundlak, city_fe, stem in specs:
        print(f"\n=== {label}  N={len(sample):,} ===", flush=True)
        parts = []
        for ycol, tag in [("y_revisit", "revisit"), ("y_continue", "continue")]:
            tbl, meta = run_spec(sample, label, ycol,
                                 mundlak=mundlak, city_fe=city_fe)
            save_both(tbl, f"{stem}_{tag}.csv")
            parts.append(tbl.rename(columns={
                "eq1_BI": f"eq1_BI_{tag}",
                "eq2_behaviour": f"eq2_{tag}"}))
            summary.append(meta)
        merged = parts[0].merge(parts[1], on="variable")
        save_both(merged, f"{stem}.csv")

    sdf = pd.DataFrame(summary)
    save_both(sdf, "table_robust_gsem_summary.csv")
    print("\nSummary", flush=True)
    print(sdf.to_string(index=False), flush=True)
    return sdf


def run_r3_only():
    """Re-estimate destination FE only (LBFGS). Keeps R1/R2 CSVs."""
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(df), size=NSUB_FE, replace=False)
    sub = df.iloc[take].copy()
    label, stem = "R3 destination FE, n=150,000", "table_robust_cityfe"
    print(f"=== {label}  N={len(sub):,} ===", flush=True)
    parts, extra = [], []
    for ycol, tag in [("y_revisit", "revisit"), ("y_continue", "continue")]:
        tbl, meta = run_spec(sub, label, ycol, mundlak=False, city_fe=True)
        save_both(tbl, f"{stem}_{tag}.csv")
        parts.append(tbl.rename(columns={
            "eq1_BI": f"eq1_BI_{tag}",
            "eq2_behaviour": f"eq2_{tag}"}))
        extra.append(meta)
    save_both(parts[0].merge(parts[1], on="variable"), f"{stem}.csv")
    sum_path = os.path.join(OUT, "table_robust_gsem_summary.csv")
    sdf = pd.read_csv(sum_path)
    sdf = sdf[~sdf["spec"].str.contains("destination FE", na=False)]
    sdf = pd.concat([sdf, pd.DataFrame(extra)], ignore_index=True)
    save_both(sdf, "table_robust_gsem_summary.csv")
    print(sdf.to_string(index=False), flush=True)
    return sdf


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--r3-only":
        sys.exit(0 if run_r3_only() is not None else 1)
    sys.exit(0 if main() is not None else 1)
