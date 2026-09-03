"""
Paper 1 - Pass 2: reviewer panel construction with strict temporal ordering.

Unit of analysis = a reviewer's FIRST observed review in the global corpus.
TPB constructs are later measured from that review's text only (t0), while the
behavioural outcomes are observed strictly afterwards, in the window
(t0 + LAG_DAYS, t0 + WINDOW_DAYS].

Outcomes
  y_revisit  : the same reviewer reviews the SAME listing again in the window
  y_continue : the same reviewer reviews ANY listing again in the window
               (cross-city continuance is detectable because reviewer_id is
               global across Inside Airbnb city files)

Reviewers are sampled deterministically by a hash of reviewer_id, so the
sample is reproducible and every review written by a sampled reviewer is
retained regardless of the city it appears in.
"""
import glob, json, os, time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "work", "meta")
WORK = os.path.join(ROOT, "work")

SAMPLE_PER_1000 = 50          # 2% of reviewers
LAG_DAYS = 90                 # behaviour must occur at least 90 days after t0
WINDOW_DAYS = 730             # 24-month observation window
CENSOR_DATE = "2026-06-15"    # earliest city snapshot -> conservative censoring
START_DATE = "2010-01-01"     # discard the sparse pre-2010 tail
EPOCH = pd.Timestamp("1970-01-01")

KNUTH = np.uint64(2654435761)


def hash_sample_mask(reviewer_id: np.ndarray) -> np.ndarray:
    h = (reviewer_id.astype(np.uint64) * KNUTH) & np.uint64(0xFFFFFFFF)
    return (h % np.uint64(1000)) < np.uint64(SAMPLE_PER_1000)


def main():
    t_start = time.time()
    censor_i = (pd.Timestamp(CENSOR_DATE) - EPOCH).days
    start_i = (pd.Timestamp(START_DATE) - EPOCH).days

    files = sorted(glob.glob(os.path.join(META, "*.parquet")))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # listing context computed on the FULL city corpus (pre-t0 only)
        df = df.sort_values(["listing_id", "date_i", "review_id"], kind="mergesort")
        df["listing_prior_reviews"] = df.groupby("listing_id").cumcount()
        df["listing_first_date"] = df.groupby("listing_id")["date_i"].transform("min")
        keep = hash_sample_mask(df["reviewer_id"].to_numpy())
        frames.append(df.loc[keep].copy())
        del df
    panel = pd.concat(frames, ignore_index=True)
    del frames
    print(f"sampled reviews: {len(panel):,}", flush=True)

    panel = panel.sort_values(["reviewer_id", "date_i", "review_id"],
                              kind="mergesort").reset_index(drop=True)
    first_idx = ~panel["reviewer_id"].duplicated(keep="first")
    focal = panel.loc[first_idx].copy()
    print(f"sampled reviewers: {len(focal):,}", flush=True)

    # ---- eligibility: full 24-month window observable, post-2010 --------
    focal = focal[(focal["date_i"] >= start_i) &
                  (focal["date_i"] <= censor_i - WINDOW_DAYS)].copy()
    print(f"eligible focal reviews: {len(focal):,}", flush=True)

    # ---- outcomes -------------------------------------------------------
    t0 = focal.set_index("reviewer_id")["date_i"]
    l0 = focal.set_index("reviewer_id")["listing_id"]
    sub = panel[panel["reviewer_id"].isin(focal["reviewer_id"])].copy()
    sub["t0"] = sub["reviewer_id"].map(t0)
    sub["l0"] = sub["reviewer_id"].map(l0)
    sub["gap"] = sub["date_i"] - sub["t0"]
    in_win = (sub["gap"] >= LAG_DAYS) & (sub["gap"] <= WINDOW_DAYS)

    cont = sub.loc[in_win].groupby("reviewer_id").size().rename("n_future")
    rev = (sub.loc[in_win & (sub["listing_id"] == sub["l0"])]
             .groupby("reviewer_id").size().rename("n_future_same"))
    nxt = sub.loc[in_win].groupby("reviewer_id")["gap"].min().rename("days_to_next")

    focal = focal.merge(cont, on="reviewer_id", how="left") \
                 .merge(rev, on="reviewer_id", how="left") \
                 .merge(nxt, on="reviewer_id", how="left")
    focal[["n_future", "n_future_same"]] = focal[["n_future", "n_future_same"]].fillna(0)
    focal["y_continue"] = (focal["n_future"] > 0).astype("int8")
    focal["y_revisit"] = (focal["n_future_same"] > 0).astype("int8")

    # ---- time features available at t0 ----------------------------------
    dt = EPOCH + pd.to_timedelta(focal["date_i"], unit="D")
    focal["date"] = dt
    focal["year"] = dt.dt.year.astype("int16")
    focal["month"] = dt.dt.month.astype("int8")
    focal["listing_age_days"] = (focal["date_i"] - focal["listing_first_date"]).astype("int32")
    covid_lo = (pd.Timestamp("2020-03-11") - EPOCH).days
    covid_hi = (pd.Timestamp("2023-05-05") - EPOCH).days
    focal["covid"] = ((focal["date_i"] >= covid_lo) &
                      (focal["date_i"] <= covid_hi)).astype("int8")

    cols = ["reviewer_id", "review_id", "listing_id", "city_id", "date", "date_i",
            "year", "month", "covid", "listing_prior_reviews", "listing_age_days",
            "n_future", "n_future_same", "days_to_next", "y_revisit", "y_continue"]
    focal = focal[cols]
    focal.to_parquet(os.path.join(WORK, "panel_focal.parquet"),
                     index=False, compression="zstd")

    meta = dict(total_sampled_reviews=int(len(panel)),
                sampled_reviewers=int(first_idx.sum()),
                eligible_focal=int(len(focal)),
                y_revisit_rate=float(focal["y_revisit"].mean()),
                y_continue_rate=float(focal["y_continue"].mean()),
                sample_per_1000=SAMPLE_PER_1000, lag_days=LAG_DAYS,
                window_days=WINDOW_DAYS, censor_date=CENSOR_DATE,
                start_date=START_DATE,
                runtime_sec=round(time.time() - t_start, 1))
    with open(os.path.join(WORK, "panel_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
