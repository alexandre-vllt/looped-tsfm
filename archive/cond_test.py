"""Does the Cholesky failure rate come from the missing WhiteKernel, from the
sequence length, or from both? Fills the 2x2.

Reverso's Table 9 bank has no WhiteKernel; Chronos and TempoPFN both keep one.
Without it a large share of composite covariance matrices are rank-deficient,
and kernelsynth() responds by adding jitter to the diagonal -- which is white
noise injected into series the recipe never said should have any. Two
observations so far are confounded:

    L=2096, with white   ->  46.3 %  (corpus synth_white, 625k matrices)
    L=4096, without      ->  73.3 %  (probe, 180 matrices)

They differ in length AND in bank, so neither explains the other. This runs all
four cells with the same seed and the same code path.

    srun --account=master --partition=gpu --gres=gpu:1 --mem=48G \
         --time=00:30:00 python cond_test.py

    python cond_test.py --n 20000          # tighter error bars, ~5x slower
    python cond_test.py --lengths 1024 2096 4096

At the default --n 4000 the standard error on a 70 % rate is ~0.7 points, which
is far finer than the differences being tested. Cost is dominated by the L=4096
cells: Cholesky is O(L^3).
"""
import argparse
import json
import time

import numpy as np
import torch

import synthetic as S


def run(white, L, n, batch, dev, seed):
    """One cell. Returns the conditioning counters for n matrices."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_bank = S.use_white_kernel(white)
    S.STATS.update(matrices=0, chol_fail=0, escalations=0, fallback=0)

    t0 = time.time()
    done = 0
    while done < n:
        b = min(batch, n - done)
        S.kernelsynth(b, L, dev, rng)
        done += b
    el = time.time() - t0

    st = dict(S.STATS)
    m = max(st["matrices"], 1)
    return dict(
        white=white, length=L, bank=n_bank, seconds=round(el, 1),
        matrices=st["matrices"], chol_fail=st["chol_fail"],
        escalations=st["escalations"], fallback=st["fallback"],
        pct_fail=100.0 * st["chol_fail"] / m,
        esc_per_matrix=st["escalations"] / m,
        # a matrix needing >1 escalation got jitter 1e0 * scale: at that point
        # the diagonal dominates and the draw is effectively white noise
        pct_severe=100.0 * max(st["escalations"] - st["chol_fail"], 0) / m,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000,
                   help="matrices per cell (SE on a 70%% rate is ~0.7 pt at 4000)")
    p.add_argument("--lengths", type=int, nargs="+", default=[2096, 4096])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="cond_test.json")
    a = p.parse_args()

    print(f"{a.n:,} matrices per cell | lengths {a.lengths} | "
          f"bank with and without WhiteKernel | seed {a.seed}\n", flush=True)

    rows = []
    for L in a.lengths:
        for white in (False, True):
            r = run(white, L, a.n, a.batch, a.device, a.seed)
            rows.append(r)
            print("  L=%-5d %-13s bank=%2d | fail %5.1f %% | esc %.2f/matrix | "
                  "severe %4.1f %% | fallback %d | %5.1f s"
                  % (L, "paper bank" if not white else "+ WhiteKernel",
                     r["bank"], r["pct_fail"], r["esc_per_matrix"],
                     r["pct_severe"], r["fallback"], r["seconds"]), flush=True)

    print("\n" + "=" * 74)
    print("FIRST-JITTER FAILURE RATE (%)\n")
    print("  %-10s %14s %14s %12s" % ("length", "paper bank", "+ WhiteKernel", "difference"))
    for L in a.lengths:
        no = next(r for r in rows if r["length"] == L and not r["white"])
        yes = next(r for r in rows if r["length"] == L and r["white"])
        print("  %-10d %13.1f %14.1f %+12.1f"
              % (L, no["pct_fail"], yes["pct_fail"],
                 yes["pct_fail"] - no["pct_fail"]))

    if len(a.lengths) > 1:
        lo, hi = min(a.lengths), max(a.lengths)
        for white in (False, True):
            l = next(r for r in rows if r["length"] == lo and r["white"] == white)
            h = next(r for r in rows if r["length"] == hi and r["white"] == white)
            print("  effect of length %d -> %d, %-13s : %+.1f pt"
                  % (lo, hi, "paper bank" if not white else "+ WhiteKernel",
                     h["pct_fail"] - l["pct_fail"]))
    print("=" * 74)
    print("\n'severe' = matrices needing a second escalation, i.e. jitter 1e0 x scale.")
    print("At that level the diagonal dominates the covariance and the sampled")
    print("series is effectively white noise rather than a draw from the kernel.")

    with open(a.out, "w") as f:
        json.dump(dict(n_per_cell=a.n, seed=a.seed, rows=rows), f, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
