"""Can the dataloader keep an A100 fed? -- measured, with the GPU's demand known.

WHY THIS EXISTS
The k=1 and k=4 smoke runs both reported 2.31 M pts/s at batch 64, which looked
like "the loop is free". bench_real.py then measured k=4 at 4.06x k=1 on
synthetic tensors. Both are right: the smoke runs were DATALOADER-BOUND, so both
arms sat waiting on disk and reported the same number.

bench_real.py (batch 512, fused_recurrent, randn input) gives the GPU's appetite:

    k=1   0.091 s/step  ->  512 / 0.091  =  5626 windows/s
    k=4   0.370 s/step  ->  512 / 0.370  =  1384 windows/s

The smoke runs' 2.31 M pts/s is about 1100 windows/s. If that is the loader's
ceiling then R0 runs at ~20 % GPU utilisation and the measured R0-vs-R1 ratio
comes out near 1.3x -- an artefact of starvation, not a property of looping.
That would also waste four fifths of a GPU slot announced on a shared box.

WHAT THIS MEASURES
Windows/s out of the real loader, over a worker sweep, with the real corpora and
the real p_real / alpha. It does NOT touch the GPU, so the number is the loader's
ceiling rather than a joint rate.

A note on the page cache: the corpora are 10.6 GB and a DGX has far more RAM, so
a second pass may read from cache and look much faster than a cold start. That
is not cheating -- training does many passes -- but the FIRST pass is what the
first minutes of a run see. --repeat 2 reports both.

    python loader_probe.py --data $WORK/synth --real $WORK/real
"""
from __future__ import annotations

import argparse
import time

from reverso_torch.data import make_loader

# From bench_real.py, batch 512, fused_recurrent, randn input, A100-40GB.
GPU_DEMAND = {1: 5626.0, 4: 1384.0}      # windows/s


# Physical sanity bound. Every sample is one scattered memmap read of
# 2096 float32 = 8.4 KB. Even served entirely from page cache, a rate implying
# more than this much bandwidth means the measurement is not measuring reads.
# v1 of this script reported 997 994 windows/s = 8.4 GB/s and that went
# unchallenged; this check exists so it cannot happen silently again.
SANITY_GB_S = 6.0
WINDOW_BYTES = 2096 * 4


def timed(loader, batch: int, nw: int, seconds: float,
          prefetch: int = 2) -> tuple:
    """Sustained production rate, measured after the prefetch queue is drained.

    THE BUG THIS FIXES
    make_loader sets persistent_workers with DataLoader's default
    prefetch_factor=2, so nw workers pre-fill nw*2 batches while the first
    next() is being served. v1 timed 30 batches at nw=32 -- fewer than the 64
    already sitting in the queue -- so it timed IPC dequeue with no I/O behind
    it and reported a rate three orders of magnitude too high.

    So: consume past the buffer first (untimed), then measure over a fixed
    wall-clock window, which puts the steady-state producer rate in the number.
    """
    it = iter(loader)
    warm = nw * prefetch + 4
    for i, _ in enumerate(it):
        if i >= warm:
            break
    t0 = time.time()
    seen = 0
    for _ in it:
        seen += 1
        if time.time() - t0 >= seconds:
            break
    dt = time.time() - t0
    wps = seen * batch / dt
    gbs = wps * WINDOW_BYTES / 2**30
    return wps, dt, gbs, warm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--real", default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seconds", type=float, default=12.0,
                   help="timed window per setting, AFTER draining the prefetch "
                        "queue; a fixed wall-clock window is robust where a "
                        "fixed batch count is not")
    p.add_argument("--workers", type=int, nargs="+",
                   default=[8, 16, 32])
    p.add_argument("--repeat", type=int, default=1,
                   help="passes per setting (warm the cache first instead)")
    p.add_argument("--p-real", type=float, default=0.5)
    p.add_argument("--real-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    kw = dict(data_dir=a.data, real_dir=a.real, seq_len=2048, horizon=48,
              batch_size=a.batch_size, seed=a.seed, p_real=a.p_real,
              real_alpha=a.real_alpha, verify="none")

    print(f"batch {a.batch_size}, {a.seconds:.0f}s timed window per setting "
          f"(after draining nw*2+4 prefetched batches), "
          f"p_real {a.p_real}, alpha {a.real_alpha}")
    print(f"independent estimate to check against: the batch-64 smoke runs both "
          f"read 2.31 M pts/s\n  (= ~1100 win/s) while bench_real says k=4 at "
          f"batch 64 should do ~1122 win/s -- so k=4 was fed,\n  k=1 was "
          f"starved, and the loader ceiling at 8 workers is ~1100 win/s.")
    print(f"GPU appetite at batch 512 (bench_real.py): "
          f"k=1 needs {GPU_DEMAND[1]:.0f} win/s, k=4 needs {GPU_DEMAND[4]:.0f}\n")

    print(f"  {'workers':>8}{'warmup':>8}{'windows/s':>11}{'M pts/s':>9}"
          f"{'GB/s':>7}{'k=1':>6}{'k=4':>6}")
    best = (0.0, None)
    for nw in a.workers:
        dl = make_loader(num_workers=nw, **kw)
        for r in range(a.repeat):
            try:
                wps, dt, gbs, warm = timed(dl, a.batch_size, nw, a.seconds)
            except Exception as e:
                print(f"  {nw:>8}   {type(e).__name__}: {str(e)[:40]}")
                continue
            f1, f4 = wps / GPU_DEMAND[1], wps / GPU_DEMAND[4]
            flag = ""
            if gbs > SANITY_GB_S:
                flag = (f"   !! {gbs:.1f} GB/s EXCEEDS THE SANITY BOUND -- this "
                        f"is not measuring reads, do not trust it")
            elif f1 < 1.0:
                flag = f"   k=1 starved {1/f1:.1f}x"
            print(f"  {nw:>8}{warm:>8}{wps:>11.0f}{wps*2096/1e6:>9.2f}"
                  f"{gbs:>7.2f}{min(f1,1.0)*100:>5.0f}%{min(f4,1.0)*100:>5.0f}%"
                  + flag)
            if wps > best[0] and gbs <= SANITY_GB_S:
                best = (wps, nw)
        del dl

    wps, nw = best
    print(f"\n{'=' * 70}\n VERDICT\n{'=' * 70}")
    if nw is None:
        print("  no setting completed; investigate before launching.")
        return
    print(f"  best: {wps:.0f} windows/s at --workers {nw}")
    f1, f4 = wps / GPU_DEMAND[1], wps / GPU_DEMAND[4]
    print(f"  k=1 (R0): loader supplies {min(f1,1.0)*100:.0f}% of demand"
          + ("" if f1 >= 1 else f"  -> GPU idle {max(0,1-1/max(f1,1e-9))*0:.0f}"
                                f"  starved {1/f1:.1f}x"))
    print(f"  k=4 (R1): loader supplies {min(f4,1.0)*100:.0f}% of demand")

    print(f"""
  IF BOTH REACH 100 %  -> launch as is; the 4.06x ratio will show up in
      wall-clock and R0-vs-R1 timings mean what they look like.

  IF k=1 IS STARVED    -> the measured ratio will understate the loop's cost and
      R0 will waste most of a GPU slot. In order of cost:
        1. raise --workers to the best value above
        2. warm the page cache before launching, which costs one sequential
           read of each corpus:
             cat $WORK/synth/*.npy $WORK/real/*.npy > /dev/null
           10.6 GB, well inside a DGX's RAM. Re-run this probe after.
        3. if still starved, the bottleneck is random-access I/O, not workers:
           every sample is an 8.4 KB scattered memmap read. Options then are a
           larger --batch-size (amortises per-batch overhead, not per-window
           I/O), or drawing several windows per series per batch -- but that
           CHANGES THE SAMPLING DISTRIBUTION and therefore stream_sha256, so it
           is a data decision, not a tuning knob, and both arms would have to be
           regenerated together.

  Whatever is chosen, --workers does NOT affect stream_sha256: that was the
  point of the Philox counter-based mapping, and data.py --verify-only proved
  it at 0, 2 and 8 workers.
""")


if __name__ == "__main__":
    main()
