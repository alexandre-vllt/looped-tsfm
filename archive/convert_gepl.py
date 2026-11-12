"""
Download GEP-L and convert it to the same (N, 2304) float32 layout as the
synthetic corpus, so data.py can mix the two at 50/50.

    python convert_gepl.py --survey                  # metadata only
    python convert_gepl.py --download --out $WORK/real
    python convert_gepl.py --convert  --out $WORK/real

SOURCE
------
abel-lab/nanoTSFM-pretrain, directory GEP-L: 6.15 GB, 17 parquet files
(13 train + 2 validation + 2 test). Only the TRAIN shards are used -- the
validation and test splits are held out by construction and must not be
pretrained on.

Schema, read from the parquet footer:

    item_id      string      e.g. "BEIJING_SUBWAY_30MIN/1"
    source       string      the originating GIFT-Eval collection
    freq         string      e.g. "30T"
    source_row   int64
    target       list<list<float>>     (channels, time)

Measured on 40 of 322 row groups of train-00000:
    1-3 channels per row (1890 / 277 / 393)
    length p0 1572, p25 5856, p50-p100 8192   (build.py caps at max_length)
    88.2 % of channel-series are >= 2096
    15.3 % of channel-series contain at least one NaN

DECISIONS, AND WHY
------------------
EACH CHANNEL IS ITS OWN SERIES. Reverso is univariate; a 3-channel row is three
independent training series, which is how a univariate foundation model consumes
multivariate data.

NON-OVERLAPPING 2304-POINT CHUNKS. A Reverso sample needs seq_len + horizon =
2048 + 48 = 2096 points; 2304 leaves 209 window offsets inside each chunk, the
same as the synthetic corpus. Stride = 2304 (no overlap) so no two chunks share
a timestep -- overlapping chunks would make "distinct" training samples
correlated, which inflates the apparent corpus size without adding information.
The remainder below 2304 is discarded (~16 % of points at the median length).

CHUNKS WITH ANY NON-FINITE VALUE ARE DROPPED. This is not cosmetic. Reverso's
loss masks NaN in the TARGET, so gaps there are survivable -- but
Model.forward does

    x_min = x.min(1); x_max = x.max(1); x = (x - x_min) / (x_max - x_min)

and min/max over a tensor containing NaN return NaN, so a single NaN anywhere in
the context silently turns the whole normalised input into NaN. No exception,
no warning, just a poisoned forward pass. Dropping the chunk is the only safe
option, and the rate is reported per source.

ONE .npy PER SOURCE. data.py's Corpus with weights=None samples UNIFORMLY over
its files, so one file per source makes the sampling roughly uniform across
datasets. That is Reverso B.1's rebalancing -- "we precompute the strides on
each dataset necessary to achieve a target (roughly uniform) fraction of time
series sampled" -- obtained from the file layout instead of a stride table.
--max-per-source additionally caps each one (default 100 000, the paper's N_max).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

REPO = "abel-lab/nanoTSFM-pretrain"
SEQ_LEN, HORIZON = 2048, 48
SPAN = SEQ_LEN + HORIZON        # 2096, the minimum a sample needs
CHUNK = 2304                    # stored series length, matches the synthetic corpus
GB = 2 ** 30


def safe(name: str) -> str:
    """A source name usable as a filename, and as a Corpus key."""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def sha256_file(path: str, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------

def survey() -> None:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    rows = fs.ls(f"datasets/{REPO}/GEP-L", detail=True)
    tr = [r for r in rows if "/train-" in r["name"]]
    print(f"GEP-L: {len(rows)} files, {sum(r['size'] for r in rows)/GB:.2f} GB")
    print(f"  train shards: {len(tr)}, {sum(r['size'] for r in tr)/GB:.2f} GB "
          f"<- only these are used")
    for r in sorted(tr, key=lambda d: d["name"]):
        print(f"    {r['name'].split('/')[-1]:<34}{r['size']/2**20:>8.1f} MB")
    print(f"\n  validation/test are held out by construction -- not pretrained on.")
    print(f"  expect roughly 550-640 k chunks of {CHUNK} after NaN filtering,")
    print(f"  about 4.7-5.5 GB of float32.")


def download(dest: str) -> str:
    from huggingface_hub import snapshot_download
    print(f"downloading GEP-L train shards (~5 GB) ...")
    p = snapshot_download(REPO, repo_type="dataset",
                          allow_patterns=["GEP-L/train-*.parquet"],
                          max_workers=8)
    got = sorted(glob.glob(os.path.join(p, "GEP-L", "train-*.parquet")))
    print(f"  {len(got)} shards in {p}")
    for f in got:
        print(f"    {os.path.basename(f):<34}{os.path.getsize(f)/2**20:>8.1f} MB")
    return p


def convert(snap: str, out_dir: str, max_per_source: int, limit_shards: int) -> None:
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(snap, "GEP-L", "train-*.parquet")))
    if not shards:
        raise SystemExit(f"no train shards under {snap}/GEP-L -- run --download first")
    if limit_shards:
        shards = shards[:limit_shards]
    os.makedirs(out_dir, exist_ok=True)

    per_source: dict[str, list[np.ndarray]] = defaultdict(list)
    stat = Counter()
    nan_by_source: Counter = Counter()
    seen_by_source: Counter = Counter()
    t0 = time.time()

    for si, sh in enumerate(shards, 1):
        pf = pq.ParquetFile(sh)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["source", "target"])
            for src, tgt in zip(t.column("source").to_pylist(),
                                t.column("target").to_pylist()):
                stat["rows"] += 1
                key = safe(src)
                if len(per_source[key]) >= max_per_source:
                    stat["capped"] += 1
                    continue
                for ch in tgt:                         # each channel = one series
                    stat["channel_series"] += 1
                    seen_by_source[key] += 1
                    a = np.asarray(ch, dtype=np.float32)
                    if a.size < CHUNK:
                        stat["too_short"] += 1
                        continue
                    n = a.size // CHUNK                # non-overlapping
                    stat["points_dropped_remainder"] += a.size - n * CHUNK
                    for i in range(n):
                        if len(per_source[key]) >= max_per_source:
                            break
                        c = a[i * CHUNK:(i + 1) * CHUNK]
                        if not np.isfinite(c).all():
                            stat["chunks_nan"] += 1
                            nan_by_source[key] += 1
                            continue
                        per_source[key].append(c)
                        stat["chunks_kept"] += 1
        done = sum(len(v) for v in per_source.values())
        print(f"  shard {si}/{len(shards)}  rows {stat['rows']:,}  "
              f"chunks {done:,}  nan-dropped {stat['chunks_nan']:,}  "
              f"{time.time() - t0:.0f}s", flush=True)

    # ---- write one .npy per source ------------------------------------
    print(f"\n  writing {len(per_source)} source files to {out_dir}")
    parts = []
    for key in sorted(per_source):
        rows = per_source[key]
        if len(rows) < 32:            # too few to be worth sampling from
            stat["sources_skipped"] += 1
            print(f"    skip {key:<30} only {len(rows)} chunks")
            continue
        arr = np.stack(rows)
        p = os.path.join(out_dir, f"{key}.npy")
        with open(p, "wb") as fh:     # file handle: np.save would append .npy
            np.save(fh, arr)
        h = sha256_file(p)
        meta = {"generator": key, "n_series": int(len(arr)),
                "length": int(arr.shape[1]), "dtype": "float32",
                "sha256": h, "bytes": int(arr.nbytes),
                "requested": int(len(arr)),
                "source": f"{REPO} GEP-L train, channel-wise, "
                          f"non-overlapping {CHUNK}-point chunks",
                "nan_chunks_dropped": int(nan_by_source[key]),
                "channel_series_seen": int(seen_by_source[key])}
        json.dump(meta, open(p.replace(".npy", ".json"), "w"), indent=2)
        parts.append(meta)
        print(f"    {key:<30}{len(arr):>8,} x {arr.shape[1]}  "
              f"{arr.nbytes/GB:>6.2f} GB  std med {np.median(arr.std(1)):.3g}")

    if not parts:
        raise SystemExit("nothing written")

    total = sum(p["n_series"] for p in parts)
    corpus = hashlib.sha256("".join(
        f'{p["generator"]}:{p["sha256"]}'
        for p in sorted(parts, key=lambda d: d["generator"])).encode()).hexdigest()

    json.dump({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "length": CHUNK, "dtype": "float32",
        "total_series": total, "total_points": total * CHUNK,
        "total_bytes": sum(p["bytes"] for p in parts),
        "generators": {p["generator"]: p for p in parts},
        "source_dataset": f"{REPO} / GEP-L (train split only)",
        "splits_excluded": "GEP-L validation and test, held out by construction",
        "policy": {
            "channels": "each channel is an independent univariate series",
            "chunking": f"non-overlapping {CHUNK}-point chunks; remainder discarded",
            "nan": "chunks containing any non-finite value are dropped, because "
                   "Model.forward's min-max over a NaN context silently produces "
                   "an all-NaN input",
            "max_per_source": max_per_source,
            "rebalancing": "one file per source; Corpus samples uniformly over "
                           "files, approximating Reverso B.1's per-dataset strides",
        },
        "stats": dict(stat),
        "corpus_sha256": corpus,
        "contract": "R0 (k=1) and R1 (k=4) MUST both assert corpus_sha256.",
    }, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)

    print(f"\n  {len(parts)} sources, {total:,} series x {CHUNK}, "
          f"{total * CHUNK / 1e9:.2f} B points, "
          f"{sum(p['bytes'] for p in parts)/GB:.2f} GB")
    print(f"  rows read              {stat['rows']:,}")
    print(f"  channel-series         {stat['channel_series']:,}")
    print(f"  chunks kept            {stat['chunks_kept']:,}")
    print(f"  chunks dropped (NaN)   {stat['chunks_nan']:,} "
          f"({100*stat['chunks_nan']/max(stat['chunks_kept']+stat['chunks_nan'],1):.1f}%)")
    print(f"  series too short       {stat['too_short']:,}")
    print(f"  points lost to remainder {stat['points_dropped_remainder']:,} "
          f"({100*stat['points_dropped_remainder']/max(stat['chunks_kept']*CHUNK + stat['points_dropped_remainder'],1):.1f}%)")
    print(f"\n  corpus_sha256 {corpus}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--survey", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--snapshot", default="",
                   help="HF snapshot dir; default = whatever --download returned")
    p.add_argument("--max-per-source", type=int, default=100000,
                   help="Reverso B.1's N_max per dataset per epoch")
    p.add_argument("--limit-shards", type=int, default=0,
                   help="use only the first N train shards (for a quick test)")
    a = p.parse_args()

    if a.survey:
        survey(); return
    if not (a.download or a.convert):
        p.error("pass --survey, --download and/or --convert")

    snap = a.snapshot
    if a.download:
        snap = download(a.out or ".")
    if a.convert:
        if not a.out:
            p.error("--convert needs --out")
        if not snap:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(REPO, repo_type="dataset",
                                     allow_patterns=["GEP-L/train-*.parquet"],
                                     local_files_only=True)
        convert(snap, a.out, a.max_per_source, a.limit_shards)


if __name__ == "__main__":
    main()
