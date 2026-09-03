"""
Paper 1 - Layer 5 (causal): the layer the source article does not attempt.

Partially-linear Double Machine Learning (Chernozhukov et al., 2018) with
5-fold cross-fitting estimates the effect of each TPB construct on observed
behaviour, holding the remaining constructs, sentiment, listing context, city
and period fixed with gradient-boosted nuisance learners:

    Y = theta * T + g(Z) + e ,   T = m(Z) + v
    theta_hat = <v_hat, Y_res> / <v_hat, v_hat>

Heterogeneity (CATE) is estimated with an R-learner causal forest: a random
forest fitted to the residual-on-residual pseudo-outcome, weighted by the
squared treatment residual (Nie & Wager, 2021).

Estimates are cluster-robust at the city level.  Results accumulate in
work/dml_results.csv so the script can be re-run in short slices.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "work")
TAB = os.path.join(ROOT, "outputs", "tables")
SEED = 20260901
NSUB = 200_000
CONSTRUCTS = ["ATT", "SN", "PBC", "SAT", "BI"]
sys.path.insert(0, os.path.join(ROOT, "src"))
from p1_step6_ml import make_features  # noqa: E402


def nuisance():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08,
                             num_leaves=31, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED,
                             n_jobs=2, verbose=-1)


def cluster_se(v, u, groups):
    """Cluster-robust SE for theta = <v,y_res>/<v,v> with score v*u."""
    s = v * u
    den = float(np.sum(v * v))
    tot = 0.0
    for g in np.unique(groups):
        sg = float(np.sum(s[groups == g]))
        tot += sg * sg
    return float(np.sqrt(tot) / den)


def dml_one(X, y, tname, groups, folds=5):
    Z = X.drop(columns=[tname]).to_numpy()
    T = X[tname].to_numpy(dtype="float64")
    Y = y.astype("float64")
    yhat = np.zeros(len(Y)); that = np.zeros(len(Y))
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, va in kf.split(Z):
        yhat[va] = nuisance().fit(Z[tr], Y[tr]).predict(Z[va])
        that[va] = nuisance().fit(Z[tr], T[tr]).predict(Z[va])
    v = T - that
    yres = Y - yhat
    theta = float(np.sum(v * yres) / np.sum(v * v))
    se = cluster_se(v, yres - theta * v, groups)
    return theta, se, v, yres


def cate_forest(X, v, yres, theta, tname):
    keep = np.abs(v) > 1e-6
    pseudo = yres[keep] / v[keep]
    w = v[keep] ** 2
    Z = X.drop(columns=[tname]).iloc[keep]
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=200,
                               max_features=0.5, n_jobs=2, random_state=SEED)
    rf.fit(Z, pseudo, sample_weight=w)
    return rf.predict(Z), Z.index.to_numpy()


def main(ycol, only=None):
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    tr_mask = df["year"] <= 2019
    X = make_features(df, tr_mask, ycol)
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(df), size=min(NSUB, len(df)), replace=False)
    Xs = X.iloc[take].reset_index(drop=True)
    ys = df[ycol].to_numpy()[take]
    gs = df["city_id"].to_numpy()[take]

    path = os.path.join(WORK, "dml_results.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(
        columns=["outcome", "construct"])
    have = set(zip(done.get("outcome", []), done.get("construct", [])))
    out = done.to_dict("records")

    for c in (CONSTRUCTS if only is None else [only]):
        if (ycol, c) in have:
            continue
        t0 = time.time()
        theta, se, v, yres = dml_one(Xs, ys, c, gs)
        zst = theta / se
        from scipy.stats import norm
        rec = {"outcome": ycol, "construct": c, "theta": round(theta, 6),
               "se": round(se, 6), "z": round(zst, 3),
               "p": round(float(2 * (1 - norm.cdf(abs(zst)))), 5),
               "ci_lo": round(theta - 1.96 * se, 6),
               "ci_hi": round(theta + 1.96 * se, 6),
               "pct_of_base_rate": round(100 * theta / ys.mean(), 3),
               "seconds": round(time.time() - t0, 1)}
        out.append(rec)
        pd.DataFrame(out).to_csv(path, index=False)
        print(json.dumps(rec), flush=True)

        if c == "BI":
            cate, ridx = cate_forest(Xs, v, yres, theta, c)
            sub = df.iloc[take].reset_index(drop=True).iloc[ridx].copy()
            sub["cate"] = cate
            q = pd.qcut(sub["cate"], 5, labels=[f"Q{i}" for i in range(1, 6)])
            seg = sub.groupby(q, observed=True).agg(
                n=("cate", "size"), cate=("cate", "mean"),
                base_rate=(ycol, "mean"),
                mean_len=("n_tokens", "mean"),
                mean_sent=("vader_compound", "mean"),
                prior_reviews=("listing_prior_reviews", "mean"),
                covid_share=("covid", "mean")).round(4)
            seg.to_csv(os.path.join(TAB, f"table_cate_quintiles_{ycol}.csv"))
            top = (sub.groupby("city", observed=True)
                      .agg(n=("cate", "size"), cate=("cate", "mean"))
                      .query("n >= 1500").sort_values("cate", ascending=False))
            top.round(5).to_csv(os.path.join(TAB, f"table_cate_by_city_{ycol}.csv"))
            sub[["cate", ycol, "n_tokens", "vader_compound", "covid",
                 "listing_prior_reviews"]].to_parquet(
                os.path.join(WORK, f"cate_{ycol}.parquet"), index=False)
            print(seg.to_string(), flush=True)
            print("CATE spread (Q5-Q1):",
                  round(float(seg['cate'].iloc[-1] - seg['cate'].iloc[0]), 5),
                  flush=True)

    fin = pd.DataFrame(out)
    fin.to_csv(os.path.join(TAB, "table_dml.csv"), index=False)
    print(fin.to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y_revisit",
         sys.argv[2] if len(sys.argv) > 2 else None)
