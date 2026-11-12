"""Offline generation of the Reverso synthetic corpus.

The paper generates 1 million synthetic sequences with a maximum length of
4096, mixing KernelSynth (Algorithm 1), spike processes (Algorithm 2) and TSI
(Algorithm 3). KernelSynth needs a Cholesky factorisation per series, so this
runs offline and writes .npy shards that the training loader memory-maps.

    # measure throughput first
    python gen_synthetic.py --n 2000 --out /scratch/izar/$USER/synth_test

    # then the real thing
    python gen_synthetic.py --n 1000000 --out /scratch/izar/$USER/synth

Cost scales as L^3 for the KernelSynth part: length 4096 costs roughly 7x
length 2096. Start at 2096 (2048 context + 48 horizon) unless you need the
paper's full range of lengths.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import synthetic as S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=S.N_SYNTHETIC)
    p.add_argument("--length", type=int, default=2096,
                   help="paper max is 4096; 2096 = 2048 context + 48 horizon")
    p.add_argument("--batch", type=int, default=16, help="Cholesky batch size")
    p.add_argument("--shard", type=int, default=20_000)
    p.add_argument("--progress-every", type=int, default=256,
                   help="print a rate line every N series")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--white", action="store_true",
                   help="add the WhiteKernel to the bank (Chronos / TempoPFN "
                        "keep it, Reverso's Table 9 does not)")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    n_bank = S.use_white_kernel(a.white)
    mem = a.batch * a.length * a.length * 4 / 2**30
    print(f"{a.n:,} series x {a.length} pts | Cholesky batch {a.batch} "
          f"(~{mem:.2f} GB) | mix {S.MIX} | bank {n_bank} kernels "
          f"({'WITH' if a.white else 'without'} WhiteKernel) | -> {a.out}",
          flush=True)

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(n=a.n, length=a.length, seed=a.seed, mix=S.MIX,
                       j_max=S.J_MAX, periods=S.PERIODS, white_kernel=a.white,
                       bank=[k for k, _ in S.KERNEL_BANK],
                       spike=S.SPIKE, tsi=S.TSI), f, indent=2, default=str)

    done, shard_id, t0 = 0, 0, time.time()
    while done < a.n:
        m = min(a.shard, a.n - done)
        buf = np.empty((m, a.length), dtype=np.float32)
        filled, last = 0, 0
        while filled < m:
            b = min(a.batch, m - filled)
            buf[filled:filled + b] = S.mixed_batch(
                b, a.length, a.device, rng).cpu().numpy()
            filled += b
            if filled - last >= a.progress_every:
                last = filled
                el = time.time() - t0
                print(f"    {done + filled:,} series  {el:6.0f}s  "
                      f"{(done + filled) / el:7.1f} series/s", flush=True)
        path = os.path.join(a.out, f"shard_{shard_id:04d}.npy")
        np.save(path, buf)
        done += m
        shard_id += 1
        el = time.time() - t0
        eta = (a.n - done) / max(done / el, 1e-9)
        print(f"  {done:,}/{a.n:,}  {el:6.0f}s  {done / el:7.1f} series/s  "
              f"ETA {eta / 60:6.1f} min", flush=True)

    st = S.STATS
    m = max(st["matrices"], 1)
    print(f"\nconditioning of the composite kernels:")
    print(f"  matrices factorised      {st['matrices']:,}")
    print(f"  failed at first jitter   {st['chol_fail']:,} "
          f"({100 * st['chol_fail'] / m:.1f} %)")
    print(f"  jitter escalations       {st['escalations']:,} "
          f"({st['escalations'] / m:.2f} per matrix)")
    print(f"  fell back to white noise {st['fallback']:,}")
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(dict(st, white_kernel=a.white), f, indent=2)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
