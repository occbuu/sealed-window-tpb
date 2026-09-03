"""
Paper 1 - Pass 4 (Layer 1, dictionary branch): turn the focal review text into
TPB construct scores, sentiment and text controls.

Outputs work/features_dict.parquet with one row per eligible focal review.
Resumable per city shard.
"""
import glob, os, re, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tpb_lexicon import LEXICON, EN_STOP, AUTO_PATTERNS  # noqa: E402

TEXT = os.path.join(ROOT, "work", "text")
OUT = os.path.join(ROOT, "work", "feat")
os.makedirs(OUT, exist_ok=True)

MIN_CHARS = 100
MIN_TOKENS = 15
MIN_EN_HITS = 3

COMPILED = {c: re.compile("|".join(f"(?:{p})" for p in pats), re.I)
            for c, pats in LEXICON.items()}
AUTO_RE = re.compile("|".join(AUTO_PATTERNS), re.I)
TOKEN_RE = re.compile(r"[a-zA-Z']+")
BR_RE = re.compile(r"<br\s*/?>|\r|\n")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402

VADER = SentimentIntensityAnalyzer()


def score_text(txt):
    """Return dict of raw measurements for one review, or None if screened out."""
    if not isinstance(txt, str):
        return None
    t = BR_RE.sub(" ", txt).strip()
    if len(t) < MIN_CHARS or AUTO_RE.search(t):
        return None
    toks = TOKEN_RE.findall(t.lower())
    n_tok = len(toks)
    if n_tok < MIN_TOKENS:
        return None
    if sum(1 for w in toks[:80] if w in EN_STOP) < MIN_EN_HITS:
        return None
    out = {"n_tokens": n_tok, "n_chars": len(t),
           "n_exclaim": t.count("!"), "n_sent": t.count(".") + t.count("!") + 1}
    for c, rgx in COMPILED.items():
        h = len(rgx.findall(t))
        out[f"hits_{c}"] = h
        out[f"dens_{c}"] = 100.0 * h / n_tok
        out[f"pres_{c}"] = int(h > 0)
    vs = VADER.polarity_scores(t[:1500])
    out["vader_compound"] = vs["compound"]
    out["vader_pos"] = vs["pos"]
    out["vader_neg"] = vs["neg"]
    return out


def main():
    files = sorted(glob.glob(os.path.join(TEXT, "*.parquet")))
    for f in files:
        cid = os.path.basename(f)[:3]
        out_path = os.path.join(OUT, f"{cid}.parquet")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        df = pd.read_parquet(f)
        if len(df) == 0:
            pd.DataFrame({"review_id": pd.Series(dtype="int64")}).to_parquet(
                out_path, index=False)
            continue
        recs, ids = [], []
        for rid, txt in zip(df["review_id"].to_numpy(), df["comments"].to_numpy()):
            r = score_text(txt)
            if r is not None:
                recs.append(r)
                ids.append(rid)
        res = pd.DataFrame(recs)
        res.insert(0, "review_id", ids)
        tmp = out_path + ".tmp"
        res.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, out_path)
        print(f"{cid}: {len(res):,}/{len(df):,} kept in {time.time()-t0:.1f}s",
              flush=True)
    print("Pass 4 shards complete", flush=True)


if __name__ == "__main__":
    main()
