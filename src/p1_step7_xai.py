"""
Paper 1 - Layer 4 (interpretability): the layer the source article leaves open.

  1. TreeSHAP global importance and per-construct attribution
  2. Accumulated Local Effects (ALE) - preferred over PDP under correlated
     features (Apley & Zhu, 2020); implemented directly
  3. A glassbox spline-GAM (additive logistic model with natural splines) that
     is estimated alongside the boosted model as an interpretable counterpart
  4. Counterfactual analysis: how much construct expression a guest would have
     needed for the model to place them in the top decile of return risk
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "work")
TAB = os.path.join(ROOT, "outputs", "tables")
SEED = 20260901
CONSTRUCTS = ["ATT", "SN", "PBC", "SAT", "BI"]
sys.path.insert(0, os.path.join(ROOT, "src"))
from p1_step6_ml import make_features  # noqa: E402


def ale_1d(model, X, col, bins=20):
    """Accumulated local effects for one column of a DataFrame X."""
    x = X[col].to_numpy()
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return None
    idx = np.clip(np.searchsorted(qs, x, side="left") - 1, 0, len(qs) - 2)
    eff = np.zeros(len(qs) - 1)
    for k in range(len(qs) - 1):
        m = idx == k
        if m.sum() < 30:
            continue
        lo = X.loc[m].copy(); lo[col] = qs[k]
        hi = X.loc[m].copy(); hi[col] = qs[k + 1]
        p_lo = model.predict_proba(lo.to_numpy())[:, 1]
        p_hi = model.predict_proba(hi.to_numpy())[:, 1]
        eff[k] = np.mean(p_hi - p_lo)
    acc = np.concatenate([[0.0], np.cumsum(eff)])
    acc = acc - acc.mean()
    return pd.DataFrame({"feature": col, "x": qs, "ale": acc})


def main(ycol):
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    tr = df["year"] <= 2019
    te = df["year"] >= 2022
    X = make_features(df, tr, ycol)
    with open(os.path.join(WORK, f"lgbm_{ycol}.pkl"), "rb") as fh:
        blob = pickle.load(fh)
    model, sc = blob["model"], blob["scaler"]
    Xte = pd.DataFrame(sc.transform(X.loc[te]), columns=X.columns)
    yte = df.loc[te, ycol].to_numpy()

    rng = np.random.default_rng(SEED)
    take = rng.choice(len(Xte), size=min(40000, len(Xte)), replace=False)
    Xs = Xte.iloc[take].reset_index(drop=True)

    # ---- 1. TreeSHAP ----------------------------------------------------
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": Xs.columns,
                        "mean_abs_shap": np.abs(sv).mean(axis=0),
                        "mean_shap": sv.mean(axis=0)})
    imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["block"] = np.where(imp.feature.isin(CONSTRUCTS), "TPB construct",
                            np.where(imp.feature.str.startswith(("SENT", "vader")),
                                     "sentiment", "control"))
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.round(5).to_csv(os.path.join(TAB, f"table_shap_importance_{ycol}.csv"),
                        index=False)
    np.save(os.path.join(WORK, f"shap_values_{ycol}.npy"), sv.astype("float32"))
    Xs.to_parquet(os.path.join(WORK, f"shap_X_{ycol}.parquet"), index=False)
    print(imp.head(12).to_string(index=False), flush=True)
    print("TPB block share of |SHAP|: "
          f"{imp.loc[imp.block=='TPB construct','share_pct'].sum():.2f}%", flush=True)

    # ---- 2. ALE ---------------------------------------------------------
    ales = [a for a in (ale_1d(model, Xs, c) for c in
                        CONSTRUCTS + ["SENT", "log_len", "log_prior"])
            if a is not None]
    pd.concat(ales).to_csv(os.path.join(WORK, f"ale_{ycol}.csv"), index=False)

    # ---- 3. glassbox spline GAM ----------------------------------------
    Xtr = pd.DataFrame(sc.transform(X.loc[tr]), columns=X.columns)
    ytr = df.loc[tr, ycol].to_numpy()
    spl = SplineTransformer(n_knots=6, degree=3, include_bias=False)
    Ztr = spl.fit_transform(Xtr)
    Zte = spl.transform(Xte)
    gam = LogisticRegression(max_iter=3000, C=0.5).fit(Ztr, ytr)
    p = gam.predict_proba(Zte)[:, 1]
    gam_perf = {"model": "glassbox spline-GAM",
                "test_auc": round(roc_auc_score(yte, p), 4),
                "test_pr_auc": round(average_precision_score(yte, p), 5)}
    print(json.dumps(gam_perf), flush=True)

    # shape functions of the GAM for the five constructs
    shapes = []
    nb = spl.transform(np.zeros((1, Xtr.shape[1])))
    for j, c in enumerate(X.columns):
        if c not in CONSTRUCTS + ["SENT"]:
            continue
        grid = np.quantile(Xtr[c], np.linspace(.01, .99, 40))
        Zg = np.repeat(nb, len(grid), axis=0)
        base = np.zeros((len(grid), Xtr.shape[1]))
        base[:, j] = grid
        Zg = spl.transform(base)
        f = Zg @ gam.coef_[0]
        shapes.append(pd.DataFrame({"feature": c, "x": grid, "f": f - f.mean()}))
    pd.concat(shapes).to_csv(os.path.join(WORK, f"gam_shapes_{ycol}.csv"), index=False)

    # ---- 4. counterfactuals --------------------------------------------
    p_te = model.predict_proba(Xte.to_numpy())[:, 1]
    thr = np.quantile(p_te, 0.90)
    below = np.where(p_te < thr)[0]
    below = rng.choice(below, size=min(6000, len(below)), replace=False)
    Xcf = Xte.iloc[below].reset_index(drop=True)
    rows = []
    for c in CONSTRUCTS:
        hi = np.quantile(Xte[c], 0.99)
        cur = Xcf[c].to_numpy()
        need = np.full(len(Xcf), np.nan)
        grid = np.linspace(0, 1, 21)
        for gstep in grid[1:]:
            trial = Xcf.copy()
            trial[c] = np.maximum(cur, cur + gstep * (hi - cur))
            pr = model.predict_proba(trial.to_numpy())[:, 1]
            hit = (pr >= thr) & np.isnan(need)
            need[hit] = gstep
        rows.append({"construct": c,
                     "share_flippable_pct": round(100 * np.mean(~np.isnan(need)), 2),
                     "median_effort_of_max": (round(float(np.nanmedian(need)), 3)
                                              if np.any(~np.isnan(need)) else None)})
    cf = pd.DataFrame(rows)
    cf.to_csv(os.path.join(TAB, f"table_counterfactual_{ycol}.csv"), index=False)
    print(cf.to_string(index=False), flush=True)

    with open(os.path.join(WORK, f"xai_meta_{ycol}.json"), "w") as fh:
        json.dump({"gam": gam_perf, "top_decile_threshold": float(thr),
                   "tpb_shap_share_pct": float(
                       imp.loc[imp.block == "TPB construct", "share_pct"].sum())},
                  fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y_revisit")
