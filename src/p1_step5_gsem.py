"""
Paper 1 - Layer 2 (explanatory branch): generalised structural equation model.

Two simultaneous equations estimated on the same design matrix:
  (1) BI       = a0 + a1 ATT + a2 SN + a3 PBC + a4 SAT + a5 SENT + controls  [Gaussian]
  (2) logit(Y) = b0 + b1 BI + c1 ATT + c2 SN + c3 PBC + c4 SAT + b2 SENT + controls

Unobserved city heterogeneity is absorbed with a correlated-random-effects
(Mundlak, 1978) device: city-level means of every construct enter both
equations.  Standard errors are clustered at the city level.  Mediation is
tested with the Monte-Carlo method (Preacher & Selig, 2012), 5,000 draws.

Primary operationalisation of a construct score: z(log1p(keyword hits)) with
review length controlled.  Density (hits per 100 tokens) and binary presence
are reported as robustness checks (Table gsem_robustness).
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "work")
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(TAB, exist_ok=True)

CONS = ["ATT", "SN", "PBC", "SAT"]
ALL = CONS + ["BI"]
RNG = np.random.default_rng(20260901)


def z(v):
    v = np.asarray(v, dtype="float64")
    return (v - v.mean()) / v.std()


def build(df, mode="count"):
    X = pd.DataFrame(index=df.index)
    for c in ALL:
        if mode == "count":
            X[c] = z(np.log1p(df[f"hits_{c}"]))
        elif mode == "dens":
            X[c] = z(df[f"dens_{c}"])
        else:
            X[c] = z(df[f"pres_{c}"])
    X["SENT"] = z(df["vader_compound"])
    X["log_len"] = z(np.log1p(df["n_tokens"]))
    X["log_prior"] = z(np.log1p(df["listing_prior_reviews"]))
    X["log_age"] = z(np.log1p(np.clip(df["listing_age_days"], 0, None)))
    X["covid"] = df["covid"].astype("float64").to_numpy()
    cm = df.groupby("city_id")[[f"hits_{c}" for c in ALL]].transform("mean")
    for c in ALL:
        X[f"cm_{c}"] = z(cm[f"hits_{c}"])
    yr = pd.get_dummies(df["year"].astype(int), prefix="yr",
                        drop_first=True).astype("float64")
    return pd.concat([X, yr.set_index(X.index)], axis=1)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def fmt(c, s, p):
    return f"{c:.4f}{stars(p)} ({s:.4f})"


def fit_pair(df, ycol, mode="count"):
    X = build(df, mode)
    g = df["city_id"].to_numpy()
    X1 = sm.add_constant(X.drop(columns=["BI"]), has_constant="add")
    m1 = sm.OLS(X["BI"].to_numpy(), X1.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    X2 = sm.add_constant(X, has_constant="add")
    y = df[ycol].to_numpy(dtype="float64")
    w = sm.Logit(y, X2.to_numpy()).fit(disp=0, maxiter=200)
    m2 = sm.Logit(y, X2.to_numpy()).fit(disp=0, maxiter=200,
                                        start_params=w.params, cov_type="cluster",
                                        cov_kwds={"groups": g})
    return m1, list(X1.columns), m2, list(X2.columns)


def run(df, ycol, tag):
    m1, n1, m2, n2 = fit_pair(df, ycol, "count")
    i1 = {n: k for k, n in enumerate(n1)}
    i2 = {n: k for k, n in enumerate(n2)}

    rows = [{"variable": "Revisit intention (BI)", "eq1_BI": "",
             "eq2_behaviour": fmt(m2.params[i2["BI"]], m2.bse[i2["BI"]],
                                  m2.pvalues[i2["BI"]])}]
    labels = {"ATT": "Attitude", "SN": "Subjective norm",
              "PBC": "Perceived behavioural control", "SAT": "Satisfaction",
              "SENT": "Sentiment (VADER)", "covid": "During COVID-19",
              "log_len": "Review length (log)",
              "log_prior": "Listing prior reviews (log)",
              "log_age": "Listing age (log)", "const": "Constant"}
    for c, lab in labels.items():
        r = {"variable": lab, "eq1_BI": "", "eq2_behaviour": ""}
        if c in i1:
            r["eq1_BI"] = fmt(m1.params[i1[c]], m1.bse[i1[c]], m1.pvalues[i1[c]])
        if c in i2:
            r["eq2_behaviour"] = fmt(m2.params[i2[c]], m2.bse[i2[c]], m2.pvalues[i2[c]])
        rows.append(r)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(TAB, f"table_gsem_{tag}.csv"), index=False)

    b, sb = m2.params[i2["BI"]], m2.bse[i2["BI"]]
    med = []
    for c in CONS:
        a, sa = m1.params[i1[c]], m1.bse[i1[c]]
        dr = RNG.normal(a, sa, 5000) * RNG.normal(b, sb, 5000)
        lo, hi = np.percentile(dr, [2.5, 97.5])
        cd, sd = m2.params[i2[c]], m2.bse[i2[c]]
        pd_ = 2 * (1 - norm.cdf(abs(cd / sd)))
        sig_ind = lo * hi > 0
        med.append({"antecedent": labels[c], "a (X->BI)": round(a, 4),
                    "b (BI->Y)": round(b, 4), "indirect a*b": round(a * b, 5),
                    "MC 95% CI": f"[{lo:.5f}, {hi:.5f}]",
                    "direct c'": round(cd, 4), "direct p": round(pd_, 4),
                    "total": round(a * b + cd, 4),
                    "mediation type": (
                        "complementary (partial)" if sig_ind and pd_ < .05 and (a * b) * cd > 0
                        else "competitive (inconsistent)" if sig_ind and pd_ < .05
                        else "indirect-only (full)" if sig_ind
                        else "direct-only" if pd_ < .05 else "no effect")})
    mdf = pd.DataFrame(med)
    mdf.to_csv(os.path.join(TAB, f"table_mediation_{tag}.csv"), index=False)

    fit = {"outcome": ycol, "N": int(len(df)),
           "base_rate": round(float(df[ycol].mean()), 5),
           "eq1_r2": round(float(m1.rsquared), 4),
           "eq2_pseudo_r2": round(float(m2.prsquared), 5),
           "aic": round(float(m2.aic), 2), "bic": round(float(m2.bic), 2)}
    print(json.dumps(fit), flush=True)
    print(tbl.to_string(index=False), flush=True)
    print(mdf.to_string(index=False), flush=True)
    return fit


def robustness(df):
    out = []
    for mode in ["count", "dens", "pres"]:
        for ycol in ["y_revisit", "y_continue"]:
            m1, n1, m2, n2 = fit_pair(df, ycol, mode)
            i1 = {n: k for k, n in enumerate(n1)}
            i2 = {n: k for k, n in enumerate(n2)}
            row = {"measurement": mode, "outcome": ycol,
                   "BI->Y": fmt(m2.params[i2["BI"]], m2.bse[i2["BI"]],
                                m2.pvalues[i2["BI"]])}
            for c in CONS:
                row[f"{c}->BI"] = fmt(m1.params[i1[c]], m1.bse[i1[c]], m1.pvalues[i1[c]])
                row[f"{c}->Y"] = fmt(m2.params[i2[c]], m2.bse[i2[c]], m2.pvalues[i2[c]])
            out.append(row)
    rdf = pd.DataFrame(out)
    rdf.to_csv(os.path.join(TAB, "table_gsem_robustness.csv"), index=False)
    print(rdf.to_string(index=False), flush=True)


def main():
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    fits = [run(df, "y_revisit", "revisit"), run(df, "y_continue", "continue")]
    with open(os.path.join(WORK, "gsem_fit.json"), "w") as fh:
        json.dump(fits, fh, indent=2)
    robustness(df)


if __name__ == "__main__":
    main()
