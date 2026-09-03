"""
Paper 1 - Pass 3: fetch the raw text of the focal (t0) reviews only.

Scans the raw city CSVs once more and keeps `comments` for the review ids that
entered the panel built in Pass 2. Resumable per city.
"""
import glob, os, time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "DataPaper1")
OUT = os.path.join(ROOT, "work", "text")
LOG = os.path.join(ROOT, "work", "logs", "pass3.log")
MAXCHARS = 4000
CHUNK = 200_000

os.makedirs(OUT, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    focal = pd.read_parquet(os.path.join(ROOT, "work", "panel_focal.parquet"),
                            columns=["review_id", "city_id"])
    wanted = {int(c): np.sort(g["review_id"].to_numpy())
              for c, g in focal.groupby("city_id")}
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    log(f"Pass 3 start: {len(focal):,} focal reviews across {len(wanted)} cities")

    for cid, path in enumerate(files):
        out_path = os.path.join(OUT, f"{cid:03d}.parquet")
        if os.path.exists(out_path):
            continue
        ids = wanted.get(cid)
        if ids is None or len(ids) == 0:
            pd.DataFrame({"review_id": pd.Series(dtype="int64"),
                          "comments": pd.Series(dtype="str")}).to_parquet(
                out_path, index=False)
            continue
        t0, parts, got = time.time(), [], 0
        try:
            for ch in pd.read_csv(path, usecols=["id", "comments"],
                                  dtype={"id": "str", "comments": "str"},
                                  chunksize=CHUNK, on_bad_lines="skip",
                                  low_memory=False):
                rid = pd.to_numeric(ch["id"], errors="coerce")
                ok = rid.notna()
                rid = rid[ok].astype("int64").to_numpy()
                txt = ch["comments"][ok]
                pos = np.searchsorted(ids, rid)
                pos[pos >= len(ids)] = 0
                hit = ids[pos] == rid
                if hit.any():
                    parts.append(pd.DataFrame(
                        {"review_id": rid[hit],
                         "comments": txt.to_numpy()[hit]}))
                    got += int(hit.sum())
            df = (pd.concat(parts, ignore_index=True) if parts else
                  pd.DataFrame({"review_id": pd.Series(dtype="int64"),
                                "comments": pd.Series(dtype="str")}))
            df["comments"] = df["comments"].astype("str").str.slice(0, MAXCHARS)
            tmp = out_path + ".tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, out_path)
            log(f"[{cid+1:3d}/{len(files)}] {os.path.basename(path)[:34]}: "
                f"{got:,}/{len(ids):,} in {time.time()-t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            log(f"[{cid+1:3d}/{len(files)}] FAILED {exc!r}")

    log("Pass 3 complete")


if __name__ == "__main__":
    main()
