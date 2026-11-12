"""Look at what the generator actually produced.

    python plot_synth.py --data /scratch/izar/$USER/synth --out synth_sample.png

Generating data you have never looked at is the fastest way to train on noise
without noticing. You should see seasonality, trends, plateaus, spikes and
level shifts — and they should not all look alike.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="synth_sample.png")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    shards = sorted(glob.glob(os.path.join(a.data, "shard_*.npy")))
    if not shards:
        raise SystemExit(f"no shard_*.npy in {a.data}")
    arr = np.load(shards[0], mmap_mode="r")
    print(f"{len(shards)} shards | first shard {arr.shape}", flush=True)

    rng = np.random.default_rng(a.seed)
    idx = rng.choice(arr.shape[0], size=min(a.n, arr.shape[0]), replace=False)

    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 1.7 * rows))
    for ax, i in zip(axes.ravel(), idx):
        x = np.asarray(arr[i], dtype=np.float64)
        ax.plot(x, lw=0.7, color="steelblue")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{i}  sd={x.std():.2g}  rng={x.ptp():.2g}",
                     fontsize=7, pad=2)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"-> {a.out}", flush=True)

    # quick sanity statistics over a larger sample
    m = min(4000, arr.shape[0])
    sub = np.asarray(arr[:m], dtype=np.float64)
    sd = sub.std(axis=1)
    flat = int((sd < 1e-8).sum())
    bad = int((~np.isfinite(sub)).any(axis=1).sum())
    print(f"\nover {m} series:")
    print(f"  std      median {np.median(sd):.3g}   "
          f"p01 {np.percentile(sd, 1):.3g}   p99 {np.percentile(sd, 99):.3g}")
    print(f"  constant series   {flat}  ({100 * flat / m:.2f} %)")
    print(f"  non-finite series {bad}  ({100 * bad / m:.2f} %)")
    # crude periodicity check: share of series with a dominant spectral peak
    F = np.abs(np.fft.rfft(sub - sub.mean(1, keepdims=True), axis=1))
    F[:, 0] = 0
    peak = F.max(1) / (F.mean(1) + 1e-12)
    print(f"  spectral peak / mean   median {np.median(peak):.1f}   "
          f"share > 20: {100 * (peak > 20).mean():.1f} %")


if __name__ == "__main__":
    main()
