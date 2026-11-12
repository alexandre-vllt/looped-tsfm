"""Does the augmentation wiring hold? Seven checks, before any 4-hour run.

The expensive mistake this exists to prevent is the one we already made once:
asserting a property from a probe that never measured it (done.md, the DROPPED
claim). Each check below prints the number it measured, not a verdict it assumed.

    python augment_probe.py              # all seven
    python augment_probe.py --quick      # skip the slow hash checks

Runs on CPU, no GPU needed. If the sweep arms are running, their 30 dataloader
workers are competing for cores: the throughput numbers are then a LOWER bound,
and the correctness checks are unaffected.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from numpy.random import Generator, Philox

from reverso_torch import augment as A
from reverso_torch.data import make_loader, stream_hash

R0_STREAM = "8bece5b4efdda856"      # the plain stream, seed 0, p_real 0.75, batch 512
OK, BAD = "  ok   ", "  FAIL "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--real", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    import os
    work = os.environ.get("WORK")
    data = a.data or (work + "/synth" if work else None)
    real = a.real or (work + "/real" if work else None)
    if not data:
        raise SystemExit("WORK unset and --data not given")

    cfg = dict(A.CFG)
    kw = dict(data_dir=data, real_dir=real, seq_len=2048, horizon=48,
              batch_size=a.batch_size, seed=0, p_real=0.75, verify="none")
    fails = []

    def check(name, good, detail):
        print(f"{OK if good else BAD}{name:<44} {detail}")
        if not good:
            fails.append(name)

    print("=" * 78)
    print(" 1. REGRESSION -- the plain stream must be exactly what it always was")
    print("=" * 78)
    if a.quick:
        print("  (skipped)")
    else:
        h = stream_hash(make_loader(num_workers=0, augment=None, **kw), a.batches)
        check("plain stream unchanged", h.startswith(R0_STREAM),
              f"{h[:16]}  expected {R0_STREAM}")

    print("\n" + "=" * 78)
    print(" 2. A downsample this corpus cannot honour must RAISE, not pad")
    print("=" * 78)
    try:
        A.check_cfg({**cfg, "p_downsample": 0.5}, 2304, 2096)
        check("p_downsample>0 at L=2304 raises", False, "it did not raise")
    except ValueError as e:
        check("p_downsample>0 at L=2304 raises", True, str(e).split(":")[0][:52])
    check("max_downsample_k(2304, 2096)", A.max_downsample_k(2304, 2096) == 1,
          f"= {A.max_downsample_k(2304, 2096)} (1 means none possible)")

    print("\n" + "=" * 78)
    print(" 3. Purity -- the same ordinal twice must give the same sample")
    print("=" * 78)
    ds = make_loader(num_workers=0, augment=cfg, **kw).dataset
    same = all(torch.equal(ds[n][0], ds[n][0]) and torch.equal(ds[n][1], ds[n][1])
               for n in (0, 1, 7, 4095, 1 << 20))
    check("sample(n) is a pure function of n", same, "5 ordinals, re-drawn")

    print("\n" + "=" * 78)
    print(" 4. Worker independence -- the contract the whole pipeline exists for")
    print("=" * 78)
    if a.quick:
        print("  (skipped)")
    else:
        hs = {w: stream_hash(make_loader(num_workers=w, augment=cfg, **kw), 4)
              for w in (0, 2, 8)}
        check("augmented stream same at workers 0/2/8", len(set(hs.values())) == 1,
              "  ".join(f"w{w}:{h[:12]}" for w, h in hs.items()))
        plain = stream_hash(make_loader(num_workers=0, augment=None, **kw), 4)
        check("augmented stream differs from plain",
              list(hs.values())[0] != plain, "as designed -- record it, do not gate on it")

    print("\n" + "=" * 78)
    print(" 5. Nothing non-finite reaches the model")
    print("=" * 78)
    N = 4000
    stats = {}
    span = 2048 + 48
    rng_rows = make_loader(num_workers=0, augment=cfg, **kw).dataset
    bad = 0
    for n in range(N):
        x, y = rng_rows[n][0], rng_rows[n][1]
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            bad += 1
    check("no NaN/Inf in the batch", bad == 0, f"{bad}/{N} samples")

    print("\n" + "=" * 78)
    print(" 6. Degenerate (flat-context) rate -- the loss guard drops these")
    print("=" * 78)
    print("  Production measured 0.6 % on the plain stream (done.md 9.1). If")
    print("  augmentation pushes this up a lot, censoring is eating the signal.")
    for label, c in (("plain", None), ("augmented", cfg)):
        d = make_loader(num_workers=0, augment=c, **kw).dataset
        flat = 0
        for n in range(N):
            ctx = d[n][0]
            if float(ctx.max() - ctx.min()) <= A.DEGEN:
                flat += 1
        print(f"         {label:<10} {flat / N * 100:6.2f} %   ({flat}/{N})")

    print("\n" + "=" * 78)
    print(" 7. Cost -- the loader was already at 97.4 % utilisation")
    print("=" * 78)
    for label, c in (("plain", None), ("augmented", cfg)):
        dl = make_loader(num_workers=4, augment=c, **kw)
        it = iter(dl)
        for _ in range(3):              # drain the prefetch before timing
            next(it)
        t0 = time.time()
        for _ in range(6):
            next(it)
        dt = (time.time() - t0) / 6
        print(f"         {label:<10} {dt * 1000:7.1f} ms/batch   "
              f"{a.batch_size / dt:9,.0f} windows/s")
    print("\n  R0's step took 94 ms at batch 512. A batch slower than that makes")
    print("  the loader the bottleneck and the arm no longer measures the model.")

    print("\n" + "=" * 78)
    print(" FAILED: " + ", ".join(fails) if fails else " ALL CHECKS PASSED")
    print("=" * 78)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
