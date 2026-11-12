#!/usr/bin/env python3
"""Is our absolute MASE the paper's number divided by the seasonal-naive one?

The paper reports Reverso-Nano at 0.757 on full GIFT-Eval; our harness scores
their own published checkpoint at 1.0742 -- a factor of 1.419 on identical
weights. The hypothesis is that GIFT-Eval's leaderboard divides each row's MASE
by the seasonal-naive baseline's before aggregating, and that we report raw
MASE.

This script tests it. Both inputs are GIFT-Eval `all_results.csv` files: a
`dataset` column of the form <name>/<freq>/<term> and an
`eval_metrics/MASE[0.5]` column.

    python harness_scale.py OURS.csv .../results/Seasonal_Naive/all_results.csv

If the hypothesis holds, the third figure printed is the paper's number.
"""
import csv
import math
import sys

KEY = "dataset"
MASE = "eval_metrics/MASE[0.5]"


def load(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if MASE not in (reader.fieldnames or []):
            sys.exit(f"!! {path}: no column {MASE!r}\n   columns: {reader.fieldnames}")
        for row in reader:
            try:
                v = float(row[MASE])
            except (TypeError, ValueError):
                continue
            if v > 0 and math.isfinite(v):
                rows[row[KEY]] = v
    return rows


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main(ours_path, sn_path):
    ours, sn = load(ours_path), load(sn_path)
    common = sorted(set(ours) & set(sn))
    if not common:
        sys.exit("!! no row in common -- the dataset keys do not match")

    o = [ours[k] for k in common]
    s = [sn[k] for k in common]
    r = [ours[k] / sn[k] for k in common]

    print(f"rows: ours {len(ours)}   seasonal-naive {len(sn)}   common {len(common)}")
    missing = sorted(set(ours) - set(sn))
    if missing:
        print(f"  {len(missing)} of our rows absent from seasonal-naive: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    print()
    print(f"geo mean, our raw MASE           {geo(o):.4f}")
    print(f"geo mean, seasonal-naive MASE    {geo(s):.4f}    <- the implied factor")
    print(f"geo mean, ours / seasonal-naive  {geo(r):.4f}    <- compare with 0.757")
    print()
    print(f"identity check   {geo(o):.4f} / {geo(s):.4f} = {geo(o)/geo(s):.4f}"
          f"   (must equal the line above)")
    print()
    for term in ("short", "medium", "long"):
        sub = [k for k in common if k.endswith("/" + term)]
        if sub:
            print(f"  {term:6s} {len(sub):3d} rows    raw {geo([ours[k] for k in sub]):.4f}"
                  f"    normalised {geo([ours[k] / sn[k] for k in sub]):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
