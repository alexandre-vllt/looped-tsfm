"""How concentrated is the raw-unit MAE objective? -- MEASURE BEFORE CHANGING IT.

reverso_torch.looped.decode ends with

    out = out * stdev + means          # stdev = x_range = context max - min

so the head emits NORMALISED values and reverso_loss takes MAE in RAW units.
Each window's contribution to the loss is therefore weighted by its own range.

With 43 sources spanning orders of magnitude, that may mean a handful of windows
decide every gradient. "May" is the point: this script measures it instead of
asserting it, because moving the loss into normalised space is a deviation from
Reverso's code and needs evidence, not a hunch.

What it reports, per corpus and for the mix:

  1. the distribution of x_range (the per-window min-max range)
  2. the CONCENTRATION of raw-unit MAE: what share of the total comes from the
     top 0.1 / 1 / 10 % of windows. This is the number that matters -- if the
     top 1 % of windows carry 90 % of the loss, then 99 % of the data is not
     training the model.
  3. the same loss computed in NORMALISED space, for comparison

The predictor is the `last` baseline (repeat the final context value), not a
model: it needs no checkpoint, it is the strongest of the three naive baselines
on this data, and the concentration of ITS error is a property of the DATA, not
of any particular model's failure mode.

    python scale_probe.py --data $WORK/synth --real $WORK/real --batches 40
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from reverso_torch.data import make_loader


def pct(a: np.ndarray, qs=(0, 1, 25, 50, 75, 99, 99.9, 100)) -> str:
    v = np.percentile(a, qs)
    return "  ".join(f"p{q:g}={x:.4g}" for q, x in zip(qs, v))


def concentration(err: np.ndarray, shares=(0.001, 0.01, 0.1)) -> None:
    """Share of total error carried by the top fraction of windows."""
    tot = err.sum()
    order = np.sort(err)[::-1]
    n = len(order)
    print(f"    total MAE mass {tot:.4g} over {n} windows")
    for s in shares:
        k = max(int(round(s * n)), 1)
        frac = order[:k].sum() / tot
        print(f"    top {s*100:>5.1f}% of windows ({k:>5d}) carry "
              f"{frac*100:>5.1f}% of the loss")
    # Where does the median window sit relative to the mean?
    print(f"    mean per-window MAE {err.mean():.4g}, "
          f"median {np.median(err):.4g}  "
          f"(mean/median = {err.mean()/max(np.median(err),1e-12):.1f}x)")


def probe(label: str, loader, n_batches: int) -> dict:
    ranges, raw_err, nrm_err = [], [], []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        # context min/max, exactly as encode() computes them
        xmin = x.min(1, keepdim=True)[0]
        xmax = x.max(1, keepdim=True)[0]
        xrng = torch.clamp(xmax - xmin, min=1e-5)

        last = x[:, -1:, :]                       # the `last` baseline
        e_raw = (last - y).abs().mean(dim=(1, 2))         # raw units
        e_nrm = ((last - y) / xrng).abs().mean(dim=(1, 2))  # normalised units

        ranges.append(xrng.flatten().numpy())
        raw_err.append(e_raw.numpy())
        nrm_err.append(e_nrm.numpy())

    ranges = np.concatenate(ranges)
    raw_err = np.concatenate(raw_err)
    nrm_err = np.concatenate(nrm_err)

    print(f"\n{'=' * 70}\n {label}  ({len(raw_err)} windows)\n{'=' * 70}")
    print(f"  x_range (per-window context min-max span)")
    print(f"    {pct(ranges)}")
    print(f"    spread p99.9/p50 = {np.percentile(ranges,99.9)/max(np.percentile(ranges,50),1e-12):.4g}x")

    print(f"\n  RAW-unit MAE of the `last` baseline -- this is Reverso's loss")
    concentration(raw_err)

    print(f"\n  NORMALISED-unit MAE of the same baseline, same windows")
    concentration(nrm_err)

    print(f"\n  the two objectives side by side")
    print(f"    raw        mean {raw_err.mean():.4g}   p99.9 {np.percentile(raw_err,99.9):.4g}")
    print(f"    normalised mean {nrm_err.mean():.4g}   p99.9 {np.percentile(nrm_err,99.9):.4g}")

    def top(e, s):
        o = np.sort(e)[::-1]
        return o[:max(int(round(s * len(o))), 1)].sum() / e.sum()
    return {"label": label, "n": int(len(raw_err)),
            "raw_top1": top(raw_err, 0.01), "raw_top10": top(raw_err, 0.10),
            "nrm_top1": top(nrm_err, 0.01), "nrm_top10": top(nrm_err, 0.10),
            "raw_mean": float(raw_err.mean()), "nrm_mean": float(nrm_err.mean()),
            "raw_ratio": float(raw_err.mean() / max(np.median(raw_err), 1e-12)),
            "nrm_ratio": float(nrm_err.mean() / max(np.median(nrm_err), 1e-12))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--real", default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-real", type=float, default=0.5)
    p.add_argument("--real-alpha", type=float, default=0.5)
    p.add_argument("--verify", default="none", choices=["full", "quick", "none"])
    a = p.parse_args()

    kw = dict(data_dir=a.data, real_dir=a.real, seq_len=2048, horizon=48,
              batch_size=a.batch_size, seed=a.seed, num_workers=4,
              verify=a.verify, real_alpha=a.real_alpha)

    # All three probes use the SAME loader construction and differ only in
    # p_real. Passing the real corpus as data_dir would make Corpus demand a
    # TempoPFN generator weight for each of GEP-L's 34 source names -- which is
    # exactly how the first version of this script crashed. p_real=1.0 draws
    # every window from the real corpus through the verified code path instead.
    runs = [("MIX  (p_real=%.2f, alpha=%.2f) -- WHAT TRAINING SEES"
             % (a.p_real, a.real_alpha), a.p_real),
            ("SYNTHETIC only", 0.0)]
    if a.real:
        runs.append(("REAL only (GEP-L, alpha=%.2f)" % a.real_alpha, 1.0))

    summary = []
    for label, pr in runs:
        summary.append(probe(label, make_loader(p_real=pr, **kw), a.batches))

    print(f"\n{'=' * 70}")
    print(" SUMMARY -- the decision numbers, so they do not scroll away")
    print(f"{'=' * 70}")
    print(f"  {'corpus':<34}{'top1%':>8}{'top10%':>9}{'mean/med':>10}")
    for r in summary:
        print(f"  {r['label'][:33]:<34}"
              f"{r['raw_top1']*100:>7.1f}%{r['raw_top10']*100:>8.1f}%"
              f"{r['raw_ratio']:>9.1f}x   RAW")
        print(f"  {'':<34}{r['nrm_top1']*100:>7.1f}%{r['nrm_top10']*100:>8.1f}%"
              f"{r['nrm_ratio']:>9.1f}x   scale-free")

    print(f"\n{'=' * 70}")
    print(" HOW TO READ THIS")
    print(f"{'=' * 70}")
    print("""
  If the top 1 % of windows carry a LARGE share of the raw loss (say > 50 %),
  the raw-unit objective is training on a small, scale-selected subset and the
  other 99 % of the corpus is nearly inert. The fix is to compute the loss in
  normalised space -- a declared deviation from Reverso's decode path, cheap to
  implement (compare the pre-denormalisation output against y normalised by the
  same context statistics), and it would need recording in done.md section 9.

  If the share is modest (say < 20 %), the raw objective is acceptable as-is and
  the |g| spread seen in the smoke run is better handled by raising --clip so
  that gradient magnitude is not thrown away on every step.

  Either way this affects BOTH arms identically, so it is a question of
  statistical power, not of bias between R0 and R1.
""")


if __name__ == "__main__":
    main()
