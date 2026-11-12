"""Third synthetic corpus — the paper's stated parameters, with our two known
deviations removed.

What changes relative to gen_synthetic.py (corpora `synth` and `synth_white`):

  1. LENGTH.  4096 instead of 2096.  Reverso D.1: "The maximum sequence length
     is set to 4096."  We had frozen it at 2096 because Cholesky is O(L^3);
     4096 costs 7.5x more.  Read literally the sentence caps the generator, so
     --length-mode fixed (the default) puts every series at 4096.  If you read
     it as "lengths are sampled up to 4096" instead, --length-mode uniform
     draws one length per batch and pads the rest of the row with NaN.  The
     paper does not disambiguate; whichever you pick is recorded in meta.json.

  2. MIXTURE.  Exactly 60 / 20 / 20 in expectation.  The old mixed_batch() did
     int(round(B * p)) per batch, which at B=16 gives 10 / 3 / 3 = 62.5 /
     18.75 / 18.75 — a fixed 2.5-point bias, not sampling noise.  Here the
     three counts come from one multinomial draw per batch, so the realised
     corpus fractions converge to 60/20/20 (over 1M series, +/- ~0.05 pt).

synthetic.py is deliberately NOT modified: it remains the exact record of what
produced the first two corpora.

    # cost check first — 400 series tells you the real rate at L=4096
    python gen_synthetic_paper.py --n 400 --out /scratch/izar/$USER/synth_probe

    # the real thing (resumable: re-running skips finished shards)
    python gen_synthetic_paper.py --n 1000000 --out /scratch/izar/$USER/synth_paper

Budget: ~16.4 GB on disk (1M x 4096 x float32) and several hours of GPU.
Cholesky is O(L^3), so expect roughly 7.5x the 37 min that `synth_white` took.
Use --max-hours to exit cleanly before a walltime and chain jobs with
--dependency=afterany; finished shards are never regenerated.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import synthetic as S
import sampler_eigh as E


# --------------------------------------------------------------------------
def paper_batch(B, L, dev, rng, mix, ks_fn=None, ks_kw=None):
    """One batch with the generator counts drawn from a multinomial.

    Returns (x, counts). Unbiased: E[count_i] = B * mix_i exactly, with no
    rounding floor. Contrast with S.mixed_batch, which rounds and is biased.
    """
    p = [mix["kernelsynth"], mix["spike"], mix["tsi"]]
    n_k, n_s, n_t = (int(v) for v in rng.multinomial(B, p))

    ks_fn = ks_fn or S.kernelsynth
    ks_kw = ks_kw or {}
    parts = []
    if n_k:
        parts.append(ks_fn(n_k, L, dev, rng, **ks_kw))
    if n_s:
        parts.append(S.spike(n_s, L, dev, rng))
    if n_t:
        parts.append(S.tsi(n_t, L, dev, rng))

    x = torch.cat(parts, 0)
    return x[torch.randperm(B, device=dev)], (n_k, n_s, n_t)


def draw_length(rng, mode, lo, hi):
    """One length per BATCH, not per series.

    kernelsynth factorises a batch of (L, L) matrices together, so every series
    in a batch must share L. With 62,500 batches the corpus still sees tens of
    thousands of distinct lengths; only the within-batch variation is lost.
    """
    if mode == "fixed":
        return hi
    if mode == "uniform":
        return int(rng.integers(lo, hi + 1))
    if mode == "loguniform":
        return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))
    raise ValueError(mode)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=S.N_SYNTHETIC)
    p.add_argument("--length", type=int, default=S.MAX_LEN,
                   help="paper D.1 maximum; also the padded row width")
    p.add_argument("--length-mode", default="fixed",
                   choices=("fixed", "uniform", "loguniform"),
                   help="fixed = every series at --length (literal reading of "
                        "the paper); uniform/loguniform draw one length per "
                        "batch in [--length-min, --length] and NaN-pad")
    p.add_argument("--length-min", type=int, default=2096,
                   help="floor for the variable modes; below 2048+48 a series "
                        "cannot supply a full training window")
    p.add_argument("--batch", type=int, default=16,
                   help="Cholesky batch; ~5-6 GB of GPU at L=4096, B=16")
    p.add_argument("--shard", type=int, default=10_000,
                   help="series per .npy shard (10k x 4096 x f32 = 164 MB)")
    p.add_argument("--progress-every", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--white", action="store_true",
                   help="add the WhiteKernel to the bank; the paper's Table 9 "
                        "has none, so leave this off for the faithful corpus")
    p.add_argument("--sampler", default="eigh", choices=("eigh", "cholesky"),
                   help="eigh = TempoPFN numerics, exact on rank-deficient "
                        "kernels, no jitter (default). cholesky = the jittered "
                        "path that produced synth and synth_white.")
    p.add_argument("--eig-dtype", default="float32",
                   choices=("float32", "float64"),
                   help="float32 reconstructs K to 1.9e-4 relative, which is "
                        "500x cleaner than the jitter it replaces")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop cleanly after this long; re-run the same command "
                        "to continue (finished shards are skipped)")
    a = p.parse_args()

    if a.length_mode != "fixed" and a.length_min > a.length:
        raise SystemExit("--length-min must be <= --length")

    os.makedirs(a.out, exist_ok=True)
    n_bank = S.use_white_kernel(a.white)
    if a.sampler == "eigh":
        import torch as _t
        ks_fn = E.kernelsynth_eigh
        ks_kw = dict(dtype=_t.float64 if a.eig_dtype == "float64" else _t.float32)
        E.reset_stats()
    else:
        ks_fn, ks_kw = S.kernelsynth, {}

    n_shards = (a.n + a.shard - 1) // a.shard
    disk = a.n * a.length * 4 / 2**30
    gpu = a.batch * a.length * a.length * 4 / 2**30

    print(f"{a.n:,} series | rows of {a.length} | length mode {a.length_mode}"
          + (f" [{a.length_min}, {a.length}]" if a.length_mode != "fixed" else "")
          + f"\nmix {S.MIX} drawn per batch by multinomial (unbiased)"
          f"\nbank {n_bank} kernels ({'WITH' if a.white else 'without'} WhiteKernel)"
          f"\nsampler {a.sampler}"
          + (f" ({a.eig_dtype})" if a.sampler == "eigh" else " (+jitter 1e-4 on every matrix)")
          + f"\n{n_shards} shards of {a.shard:,} | ~{disk:.1f} GB on disk | "
          f"~{gpu:.2f} GB per Cholesky tensor"
          f"\n-> {a.out}\n", flush=True)

    meta = dict(
        n=a.n, row_width=a.length, length_mode=a.length_mode,
        length_min=a.length_min if a.length_mode != "fixed" else a.length,
        seed=a.seed, mix=S.MIX, mix_draw="multinomial per batch",
        j_max=S.J_MAX, periods=S.PERIODS, white_kernel=a.white,
        bank=[k for k, _ in S.KERNEL_BANK], spike=S.SPIKE, tsi=S.TSI,
        note=("Reverso D.1 says 'The maximum sequence length is set to 4096' "
              "without stating whether lengths are sampled below it. "
              f"This corpus used length_mode={a.length_mode}. Rows shorter "
              "than row_width are NaN-padded on the right."),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    counts = np.zeros(3, dtype=np.int64)
    lengths_all = []
    t0 = time.time()
    done_series = 0

    for sid in range(n_shards):
        path = os.path.join(a.out, f"shard_{sid:04d}.npy")
        lpath = os.path.join(a.out, f"shard_{sid:04d}.len.npy")
        m = min(a.shard, a.n - sid * a.shard)

        # ---- resume: a shard of the right shape is taken as finished -------
        if os.path.exists(path):
            try:
                if np.load(path, mmap_mode="r").shape == (m, a.length):
                    done_series += m
                    print(f"  shard {sid:04d}  already present, skipped",
                          flush=True)
                    continue
            except (ValueError, OSError):
                pass  # truncated or unreadable -> regenerate
            print(f"  shard {sid:04d}  present but malformed, regenerating",
                  flush=True)

        # Per-shard RNG so a resumed run reproduces the same shard it would
        # have produced in one pass.
        rng = np.random.default_rng([a.seed, sid])
        torch.manual_seed(a.seed * 1_000_003 + sid)

        buf = np.full((m, a.length), np.nan, dtype=np.float32)
        lens = np.empty(m, dtype=np.int32)
        filled, last = 0, 0
        ts = time.time()

        while filled < m:
            b = min(a.batch, m - filled)
            L = draw_length(rng, a.length_mode, a.length_min, a.length)
            x, c = paper_batch(b, L, a.device, rng, S.MIX, ks_fn, ks_kw)
            buf[filled:filled + b, :L] = x.cpu().numpy()
            lens[filled:filled + b] = L
            counts += c
            filled += b
            if filled - last >= a.progress_every:
                last = filled
                el = time.time() - ts
                print(f"    shard {sid:04d}  {filled:,}/{m:,}  {el:6.0f}s  "
                      f"{filled / el:6.1f} series/s", flush=True)

        np.save(path, buf)
        if a.length_mode != "fixed":
            np.save(lpath, lens)
            lengths_all.append(lens)
        done_series += m

        el = time.time() - t0
        rate = done_series / max(el, 1e-9)
        eta = (a.n - done_series) / max(rate, 1e-9)
        print(f"  shard {sid:04d} written  |  {done_series:,}/{a.n:,}  "
              f"{el / 60:6.1f} min  {rate:6.1f} series/s  "
              f"ETA {eta / 60:6.1f} min", flush=True)

        if a.max_hours and el > a.max_hours * 3600:
            print(f"\n--max-hours {a.max_hours} reached after "
                  f"{done_series:,} series. Re-run the identical command to "
                  f"continue; finished shards are skipped.", flush=True)
            break

    # ---- report ----------------------------------------------------------
    tot = max(int(counts.sum()), 1)
    st = S.STATS
    mm = max(st["matrices"], 1)

    print(f"\n(statistics below cover the {tot:,} series generated in THIS run; "
          f"shards skipped on resume are not counted)")
    print("\nrealised generator mix (target 60.00 / 20.00 / 20.00):")
    for name, c in zip(("kernelsynth", "spike", "tsi"), counts):
        print(f"  {name:<12} {int(c):>9,}  {100.0 * c / tot:6.2f} %")

    if a.sampler == "eigh":
        print()
        print(E.report())
    else:
        print("\nconditioning of the composite kernels:")
        print(f"  matrices factorised      {st['matrices']:,}")
        print(f"  failed at first jitter   {st['chol_fail']:,} "
              f"({100.0 * st['chol_fail'] / mm:.1f} %)")
        print(f"  jitter escalations       {st['escalations']:,} "
              f"({st['escalations'] / mm:.2f} per matrix)")
        print(f"  fell back to white noise {st['fallback']:,}")
        print("  NOTE: jitter 1e-4 x scale is added to EVERY matrix, so every")
        print("        GP series carries white noise at ~1 % of signal std.")

    if a.length_mode != "fixed" and lengths_all:
        q = np.percentile(np.concatenate(lengths_all), [0, 25, 50, 75, 100])
        print(f"\nlengths: min {q[0]:.0f}  p25 {q[1]:.0f}  median {q[2]:.0f}  "
              f"p75 {q[3]:.0f}  max {q[4]:.0f}")

    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(dict(st, white_kernel=a.white, sampler=a.sampler,
                       eig=dict(E.EIG_STATS) if a.sampler == "eigh" else None,
                       counts=dict(kernelsynth=int(counts[0]),
                                   spike=int(counts[1]), tsi=int(counts[2])),
                       fractions=dict(kernelsynth=float(counts[0] / tot),
                                      spike=float(counts[1] / tot),
                                      tsi=float(counts[2] / tot)),
                       series_written=done_series), f, indent=2)

    print(f"\n{done_series:,} series in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
