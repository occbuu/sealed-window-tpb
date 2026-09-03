"""
Paper 1 - Layer 3 (predictive branch): model bake-off under a strict
train-past / test-future protocol.

Split      train 2010-2019 | validation 2020-2021 | test 2022-2024(Jun)
Baselines  majority class, sentiment-only, TPB-only
Learners   logistic regression, linear SVM, HistGradientBoosting, LightGBM,
           multilayer perceptron (5-100 hidden neurons, replicating the source
           article's neuron sweep)
Metrics    ROC-AUC, PR-AUC, Brier score, F1 at the validation-optimal threshold

Results accumulate in work/ml_results_<outcome>.csv so the script can be
re-run in short slices until every model is done.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "work")
TAB = os.path.join(ROOT, "outputs", "tables")
SEED = 20260901

TPB = ["ATT", "SN", "PBC", "SAT", "BI"]
SENTF = ["SENT", "vader_pos", "vader_neg"]
CTRL = ["log_len", "n_exclaim", "log_prior", "log_age", "covid",
        "month_sin", "month_cos", "city_rate"]


def z(v):
    v = np.asarray(v, dtype="float64")
    return (v - v.mean()) / (v.std() + 1e-12)


def make_features(df, train_mask, ycol):
    X = pd.DataFrame(index=df.index)
    for c in TPB:
        X[c] = np.log1p(df[f"hits_{c}"])
    X["SENT"] = df["vader_compound"]
    X["vader_pos"] = df["vader_pos"]
    X["vader_neg"] = df["vader_neg"]
    X["log_len"] = np.log1p(df["n_tokens"])
    X["n_exclaim"] = np.log1p(df["n_exclaim"])
    X["log_prior"] = np.log1p(df["listing_prior_reviews"])
    X["log_age"] = np.log1p(np.clip(df["listing_age_days"], 0, None))
    X["covid"] = df["covid"].astype("float64")
    X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # city target encoding fitted on the TRAINING period only (no leakage)
    tr = df.loc[train_mask]
    prior = tr[ycol].mean()
    agg = tr.groupby("city_id")[ycol].agg(["sum", "count"])
    sm_rate = (agg["sum"] + 50 * prior) / (agg["count"] + 50)
    X["city_rate"] = df["city_id"].map(sm_rate).fillna(prior).astype("float64")
    return X


def evaluate(name, p_val, p_te, y_val, y_te, seconds, extra=""):
    ths = np.quantile(p_val, np.linspace(.80, .999, 60))
    f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in ths]
    th = ths[int(np.argmax(f1s))]
    return {"model": name, "config": extra,
            "val_auc": round(roc_auc_score(y_val, p_val), 4),
            "test_auc": round(roc_auc_score(y_te, p_te), 4),
            "val_pr_auc": round(average_precision_score(y_val, p_val), 5),
            "test_pr_auc": round(average_precision_score(y_te, p_te), 5),
            "test_brier": round(brier_score_loss(y_te, p_te), 6),
            "test_f1": round(f1_score(y_te, (p_te >= th).astype(int),
                                      zero_division=0), 4),
            "seconds": round(seconds, 1)}


def main(ycol):
    df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
    tr = df["year"] <= 2019
    va = (df["year"] >= 2020) & (df["year"] <= 2021)
    te = df["year"] >= 2022
    X = make_features(df, tr, ycol)
    y = df[ycol].to_numpy()

    sc = StandardScaler().fit(X.loc[tr])
    Xtr, Xva, Xte = (sc.transform(X.loc[m]) for m in (tr, va, te))
    ytr, yva, yte = y[tr.to_numpy()], y[va.to_numpy()], y[te.to_numpy()]
    print(f"{ycol}: train {len(ytr):,} ({ytr.mean():.4f}) | "
          f"val {len(yva):,} ({yva.mean():.4f}) | test {len(yte):,} "
          f"({yte.mean():.4f})", flush=True)

    idx = {c: k for k, c in enumerate(X.columns)}
    res_path = os.path.join(WORK, f"ml_results_{ycol}.csv")
    done = (pd.read_csv(res_path) if os.path.exists(res_path)
            else pd.DataFrame(columns=["model", "config"]))
    have = set(zip(done.get("model", []), done.get("config", []).fillna("")))
    out = done.to_dict("records")

    def add(rec):
        out.append(rec)
        pd.DataFrame(out).to_csv(res_path, index=False)
        print(json.dumps(rec), flush=True)

    def todo(name, cfg=""):
        return (name, cfg) not in have

    # ---------------- baselines -----------------------------------------
    if todo("B0 majority class"):
        t = time.time()
        add(evaluate("B0 majority class", np.full(len(yva), ytr.mean()),
                     np.full(len(yte), ytr.mean()), yva, yte, time.time() - t))
    for nm, cols in [("B1 sentiment-only", SENTF), ("B2 TPB-only", TPB),
                     ("B3 controls-only", CTRL)]:
        if todo(nm):
            t = time.time()
            k = [idx[c] for c in cols]
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr[:, k], ytr)
            add(evaluate(nm, m.predict_proba(Xva[:, k])[:, 1],
                         m.predict_proba(Xte[:, k])[:, 1], yva, yte,
                         time.time() - t, f"{len(cols)} features"))

    # ---------------- full-feature learners ------------------------------
    if todo("M1 logistic regression"):
        t = time.time()
        m = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
        add(evaluate("M1 logistic regression", m.predict_proba(Xva)[:, 1],
                     m.predict_proba(Xte)[:, 1], yva, yte, time.time() - t))

    if todo("M2 linear SVM"):
        t = time.time()
        m = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=30,
                          random_state=SEED).fit(Xtr, ytr)
        add(evaluate("M2 linear SVM", m.predict_proba(Xva)[:, 1],
                     m.predict_proba(Xte)[:, 1], yva, yte, time.time() - t,
                     "SGD hinge/huber"))

    for lr_, leaves in [(0.05, 31), (0.10, 63)]:
        cfg = f"lr={lr_}, leaves={leaves}"
        if todo("M3 HistGradientBoosting", cfg):
            t = time.time()
            m = HistGradientBoostingClassifier(
                learning_rate=lr_, max_leaf_nodes=leaves, max_iter=300,
                early_stopping=True, validation_fraction=0.1,
                random_state=SEED).fit(Xtr, ytr)
            add(evaluate("M3 HistGradientBoosting", m.predict_proba(Xva)[:, 1],
                         m.predict_proba(Xte)[:, 1], yva, yte, time.time() - t, cfg))

    for leaves in [31, 63, 127]:
        cfg = f"leaves={leaves}, lr=0.05"
        if todo("M4 LightGBM", cfg):
            t = time.time()
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                   num_leaves=leaves, subsample=0.8,
                                   colsample_bytree=0.8, random_state=SEED,
                                   n_jobs=2, verbose=-1)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            add(evaluate("M4 LightGBM", m.predict_proba(Xva)[:, 1],
                         m.predict_proba(Xte)[:, 1], yva, yte, time.time() - t, cfg))
            if leaves == 63:
                import pickle
                with open(os.path.join(WORK, f"lgbm_{ycol}.pkl"), "wb") as fh:
                    pickle.dump({"model": m, "cols": list(X.columns),
                                 "scaler": sc}, fh)

    for h in [5, 10, 20, 30, 50, 70, 100]:
        cfg = f"{h} hidden neurons"
        if todo("M5 neural network", cfg):
            t = time.time()
            m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=60,
                              early_stopping=True, n_iter_no_change=5,
                              random_state=SEED, learning_rate_init=3e-3,
                              batch_size=4096).fit(Xtr, ytr)
            add(evaluate("M5 neural network", m.predict_proba(Xva)[:, 1],
                         m.predict_proba(Xte)[:, 1], yva, yte, time.time() - t, cfg))

    final = pd.DataFrame(out).sort_values("test_pr_auc", ascending=False)
    final.to_csv(os.path.join(TAB, f"table_ml_bakeoff_{ycol}.csv"), index=False)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y_continue")
