"""Which windows does the scale-free loss guard drop, and are they benign?

WHY THIS EXISTS -- A CORRECTION
done.md 9.1 claims the degenerate-window count "was 0 across the 20 480 windows
scale_probe.py measured". That claim is FALSE. scale_probe.py printed x_range
percentiles; it never counted how many windows hit the 1e-5 clamp. The real
k=1 smoke run reports DROPPED 320/51200 = 0.6 %, which is the first actual
measurement of this quantity.

WHY 0.6 % IS NOT OBVIOUSLY FINE
A constant context (x_max - x_min <= 1e-5) splits into two very different cases:

  FLAT -> FLAT    encode gives x_tilde = 0 everywhere, and y_tilde = (c-c)/s = 0,
                  so predicting 0 is exactly right: loss 0, nothing explodes.
                  These windows are BENIGN and free -- they teach "a flat series
                  stays flat", which is a real pattern in metered data.

  FLAT -> JUMP    y_tilde = (y-c)/1e-5 explodes. Genuinely unpredictable from a
                  constant context, and the window that would dominate the loss.

The current guard drops BOTH. This script measures the split, and attributes the
drops to source files, because 0.6 % removed uniformly is noise but 0.6 % taken
out of one domain is a bias in a corpus whose per-source weighting (alpha = 0.5,
done.md 5.3.1) was chosen deliberately.

    python degen_probe.py --data $WORK/synth --real $WORK/real --per-file 3000
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DEGEN = 1e-5          # the guard threshold, == Reverso's own encode() clamp


def scan(root: str, label: str, per_file: int, seq_len: int, horizon: int,
         seed: int) -> list:
    span = seq_len + horizon
    rows = []
    paths = sorted(glob.glob(os.path.join(root, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"no .npy in {root}")

    print(f"\n{'=' * 78}\n {label}  --  {len(paths)} sources, "
          f"{per_file} random windows each\n{'=' * 78}")
    print(f"  {'source':<34}{'degen%':>9}{'flat>flat':>11}{'flat>jump':>11}"
          f"{'worst |y~|':>12}")

    for p in paths:
        a = np.load(p, mmap_mode="r")
        n, L = a.shape
        if L < span:
            continue
        rng = np.random.default_rng(abs(hash((seed, os.path.basename(p)))) % 2**32)
        idx = rng.integers(0, n, size=per_file)
        off = rng.integers(0, L - span + 1, size=per_file)

        ndeg = nflat = njump = 0
        worst = 0.0
        for i, o in zip(idx, off):
            w = np.asarray(a[i, o:o + span], dtype=np.float64)
            x, y = w[:seq_len], w[seq_len:]
            s = x.max() - x.min()
            if s > DEGEN:
                continue
            ndeg += 1
            # Is the future flat at the same level? Compare against the same
            # threshold, so "benign" means the whole span is one constant.
            dev = np.abs(y - x[-1]).max()
            if dev <= DEGEN:
                nflat += 1
            else:
                njump += 1
                worst = max(worst, dev / max(s, DEGEN))

        pct = 100.0 * ndeg / per_file
        rows.append({"source": os.path.basename(p)[:-4], "n_series": int(n),
                     "degen_pct": pct, "flat": nflat, "jump": njump,
                     "worst": worst})
        if ndeg:
            print(f"  {rows[-1]['source'][:33]:<34}{pct:>8.2f}%{nflat:>11d}"
                  f"{njump:>11d}{worst:>12.3g}")

    silent = [r for r in rows if r["degen_pct"] == 0.0]
    print(f"  ({len(silent)} sources with no degenerate windows, not listed)")

    tot_d = sum(r["degen_pct"] for r in rows) / max(len(rows), 1)
    tot_f = sum(r["flat"] for r in rows)
    tot_j = sum(r["jump"] for r in rows)
    print(f"\n  mean degenerate rate across sources {tot_d:.3f}%")
    print(f"  of the degenerate windows: {tot_f} flat->flat (benign), "
          f"{tot_j} flat->jump (pathological)")
    if tot_f + tot_j:
        print(f"  benign share {100.0*tot_f/(tot_f+tot_j):.1f}%")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--real", default="")
    p.add_argument("--per-file", type=int, default=3000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    allrows = []
    allrows += [("synthetic", r) for r in
                scan(a.data, "SYNTHETIC", a.per_file, a.seq_len, a.horizon, a.seed)]
    if a.real:
        allrows += [("real", r) for r in
                    scan(a.real, "REAL (GEP-L)", a.per_file, a.seq_len,
                         a.horizon, a.seed)]

    print(f"\n{'=' * 78}\n VERDICT\n{'=' * 78}")
    hot = sorted((r for _, r in allrows), key=lambda r: -r["degen_pct"])[:8]
    print("  worst sources by degenerate rate:")
    for r in hot:
        if r["degen_pct"] == 0:
            break
        print(f"    {r['source'][:34]:<36}{r['degen_pct']:>7.2f}%  "
              f"({r['n_series']:,} series)")

    tf = sum(r["flat"] for _, r in allrows)
    tj = sum(r["jump"] for _, r in allrows)
    print(f"""
  benign (flat->flat) {tf}   pathological (flat->jump) {tj}

  HOW TO DECIDE

  If MOST degenerate windows are flat->flat, the guard is throwing away free
  correct predictions. Better: keep them and give the scale a floor so the
  normalised target stays 0 -- they then contribute loss 0 and teach "flat
  stays flat", while only flat->jump windows are excluded.

  If MOST are flat->jump, the guard is doing its job and dropping ~0.6 % of
  windows is the correct, cheap answer.

  Either way: if the drops concentrate in one or two sources, say so in
  done.md 9.1 next to the alpha = 0.5 weighting, because the effective
  per-source mixture is then not the one that decision chose.
""")


if __name__ == "__main__":
    main()
