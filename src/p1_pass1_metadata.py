"""
Paper 1 - Pass 1: Metadata extraction from Inside Airbnb review corpora.

Streams every city-level reviews CSV and writes a compact Parquet file
containing only the columns required to build the reviewer panel:
    listing_id, review_id, date_i (days since 1970-01-01), reviewer_id, city_id

Text (`comments`) is deliberately NOT retained here; it is fetched in Pass 3
only for the reviews that enter the modelling sample.

Resumable: a city whose Parquet already exists is skipped.
"""
import os, sys, glob, json, time, unicodedata
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "DataPaper1")
OUT = os.path.join(ROOT, "work", "meta")
LOG = os.path.join(ROOT, "work", "logs", "pass1.log")
CHUNK = 400_000

os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def slug(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    log(f"Pass 1 start: {len(files)} city files")

    registry_path = os.path.join(ROOT, "work", "city_registry.csv")
    rows = []
    for cid, path in enumerate(files):
        city = os.path.splitext(os.path.basename(path))[0]
        rows.append({"city_id": cid, "city": city,
                     "file": os.path.basename(path),
                     "bytes": os.path.getsize(path)})
    pd.DataFrame(rows).to_csv(registry_path, index=False)

    for cid, path in enumerate(files):
        city = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT, f"{cid:03d}_{slug(city)}.parquet")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        parts, nrows, nbad = [], 0, 0
        try:
            reader = pd.read_csv(
                path, usecols=["listing_id", "id", "date", "reviewer_id"],
                dtype={"listing_id": "str", "id": "str", "reviewer_id": "str"},
                chunksize=CHUNK, engine="c", on_bad_lines="skip",
                low_memory=False)
            for ch in reader:
                ch = ch.rename(columns={"id": "review_id"})
                for c in ("listing_id", "review_id", "reviewer_id"):
                    ch[c] = pd.to_numeric(ch[c], errors="coerce")
                d = pd.to_datetime(ch["date"], errors="coerce", format="mixed")
                ch["date_i"] = (d - pd.Timestamp("1970-01-01")).dt.days
                before = len(ch)
                ch = ch.dropna(subset=["listing_id", "review_id",
                                       "reviewer_id", "date_i"])
                nbad += before - len(ch)
                out = pd.DataFrame({
                    "listing_id": ch["listing_id"].astype("int64"),
                    "review_id": ch["review_id"].astype("int64"),
                    "reviewer_id": ch["reviewer_id"].astype("int64"),
                    "date_i": ch["date_i"].astype("int32"),
                    "city_id": np.int16(cid)})
                parts.append(out)
                nrows += len(out)
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
                columns=["listing_id", "review_id", "reviewer_id",
                         "date_i", "city_id"])
            tmp = out_path + ".tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, out_path)
            log(f"[{cid+1:3d}/{len(files)}] {city}: {nrows:,} rows "
                f"(dropped {nbad:,}) in {time.time()-t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            log(f"[{cid+1:3d}/{len(files)}] {city}: FAILED -> {exc!r}")
        finally:
            parts = None

    log("Pass 1 complete")


if __name__ == "__main__":
    main()
