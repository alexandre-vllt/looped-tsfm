"""
TempoPFN's kernel_synth generator, with the GP sampling moved to the GPU.

WHY
---
TempoPFN's src/synthetic_generation/kernel_synth/kernel_synth.py samples through
sklearn's GaussianProcessRegressor.sample_y, which builds the full L x L
covariance and factorises it on one CPU core: O(L^3). Measured on nala at
L=2304 that is tens of seconds per series, so 75 000 series is several hundred
CPU-hours. A 128-process pool does not fix it -- sklearn's BLAS is already
multi-threaded, so the pool oversubscribes itself (we drove the load average to
5245 on a 256-core box and produced nothing in 10 hours).

WHAT IS KEPT IDENTICAL TO TempoPFN
----------------------------------
  * the 33-entry kernel bank, including the /length periodicity scaling
  * the composition rule: n ~ U{1..max_kernels}, n kernels drawn WITH
    replacement, folded left with a random choice of + or *
  * X = linspace(0, 1, length)
  * zero mean

WHAT CHANGES, AND WHY IT IS THE SAME DISTRIBUTION
-------------------------------------------------
Sampling from GP(0, K) needs any A with A A^T = K; then x = A z, z ~ N(0, I).
sklearn uses an SVD-based factor. We use Cholesky, falling back to the
eigendecomposition when Cholesky fails. Every such A gives EXACTLY the same
distribution N(0, K) -- only the realisation for a given z differs. So this is
not a different prior, it is a different route to the same one.

Two behavioural differences, both recorded as deviations in task.md:

  1. FAILURE PATH. sklearn raises LinAlgError on a non-PD covariance and
     TempoPFN then retries with seed+1, silently DISCARDING that composite. We
     measured 83 % of composites to be rank-deficient at L=512; those are
     legitimate GP priors (a degenerate Gaussian), and eigh samples them
     exactly. Our kernel mix is therefore slightly richer in rank-deficient
     composites than TempoPFN's.
  2. JITTER. sklearn adds alpha=1e-10 to the diagonal by default. We add none
     in the Cholesky attempt. (For reference, Reverso's own synthetic.py adds
     1e-4 x scale to EVERY matrix unconditionally -- a million times larger,
     which is the bug that made the earlier synth/synth_white corpora unusable.)

Cholesky-first is also what the Reverso paper specifies: "To enable efficient
sampling, we use batched Cholesky decomposition."

VERIFY BEFORE TRUSTING
----------------------
    python gen_kernel_gpu.py --self-test

checks every kernel in the bank against sklearn's own implementation, and checks
that the sampler reproduces the target covariance. Do not generate a corpus
without it passing.

    python gen_kernel_gpu.py --n 1000 --length 2304 --out $WORK/synth_gpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# The bank, transcribed from TempoPFN kernel_synth.py.
# Periodicities are divided by `length` there, so the bank depends on length --
# kept faithfully, including the duplicated entries (periodicity 4/length
# appears three times, labelled week / month / quarter). Duplicates matter:
# they change the sampling probability of that periodicity.
# ---------------------------------------------------------------------------
def kernel_bank(length: int):
    L = float(length)
    b = []
    for p in (24, 48, 96, 24 * 7, 48 * 7, 96 * 7, 7, 14, 30, 60,
              365, 365 * 2, 4, 26, 52, 4, 6, 12, 4, 4 * 10, 10):
        b.append(("expsine", {"periodicity": p / L, "length_scale": 1.0}))
    for s0 in (0.0, 1.0, 10.0):
        b.append(("dot", {"sigma_0": s0}))
    for ls in (0.1, 1.0, 10.0):
        b.append(("rbf", {"length_scale": ls}))
    for al in (0.1, 1.0, 10.0):
        b.append(("ratquad", {"alpha": al, "length_scale": 1.0}))
    for nl in (0.1, 1.0):
        b.append(("white", {"noise_level": nl}))
    b.append(("const", {"constant_value": 1.0}))
    return b


def gram(kind: str, prm: dict, x: torch.Tensor) -> torch.Tensor:
    """One kernel's Gram matrix. Formulas are sklearn's, verified by --self-test.

    x is (L,) on the target device. Returns (L, L) float64.
    """
    d = (x[:, None] - x[None, :]).abs()

    if kind == "rbf":                       # exp(-d^2 / (2 l^2))
        return torch.exp(-(d ** 2) / (2.0 * prm["length_scale"] ** 2))

    if kind == "expsine":                   # exp(-2 (sin(pi d / p) / l)^2)
        s = torch.sin(math.pi * d / prm["periodicity"])
        return torch.exp(-2.0 * (s / prm["length_scale"]) ** 2)

    if kind == "ratquad":                   # (1 + d^2 / (2 a l^2))^-a
        a, l = prm["alpha"], prm["length_scale"]
        return (1.0 + (d ** 2) / (2.0 * a * l ** 2)) ** (-a)

    if kind == "dot":                       # sigma_0^2 + x_i . x_j
        return prm["sigma_0"] ** 2 + x[:, None] * x[None, :]

    if kind == "white":                     # noise_level * I
        return prm["noise_level"] * torch.eye(x.numel(), dtype=x.dtype, device=x.device)

    if kind == "const":                     # constant_value
        return torch.full((x.numel(), x.numel()), prm["constant_value"],
                          dtype=x.dtype, device=x.device)

    raise ValueError(kind)


def composite(rng: np.random.Generator, bank, x: torch.Tensor, max_kernels: int):
    """n ~ U{1..max_kernels} kernels, with replacement, folded with random +/*.

    Mirrors KernelSynthGenerator.generate_time_series exactly, including the
    use of rng.integers(1, max_kernels + 1) and replace=True.
    """
    n = int(rng.integers(1, max_kernels + 1))
    idx = rng.choice(len(bank), size=n, replace=True)
    K = None
    for j in idx:
        kind, prm = bank[j]
        g = gram(kind, prm, x)
        if K is None:
            K = g
        else:
            K = K + g if rng.integers(2) == 0 else K * g
    return K


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch(Ks: torch.Tensor, gen: torch.Generator, jitter: float = 0.0):
    """(B, L, L) covariances -> (B, L) samples. Cholesky first, eigh fallback.

    Returns (samples, n_eigh) so the fallback rate is logged rather than hidden.

    jitter adds j * mean(diag(K)) to the diagonal before the Cholesky attempt.
    Measured at L=2304, ~80 % of composites fail Cholesky and fall through to
    eigh, which costs 10-30x more -- so a small jitter is a large speedup.

    What it costs in fidelity: a diagonal term j*s adds independent noise of
    variance j*s, i.e. std sqrt(j)*sqrt(s), so RELATIVE to a signal of std
    sqrt(s) the added noise is sqrt(j).

        j = 0        none; every rank-deficient matrix goes to eigh (exact)
        j = 1e-10    sklearn's default alpha -> 1e-5 relative noise
        j = 1e-8     1e-4 relative noise  (0.01 %)
        j = 1e-4     1e-2 relative noise  (1 %)  <- Reverso's synthetic.py,
                     applied unconditionally to EVERY matrix. That is the bug
                     that made the synth/synth_white corpora unusable.

    Anything up to ~1e-8 is far below the quantisation of float32 storage and
    is not detectable in the data. It is still recorded in the manifest.
    """
    B, L, _ = Ks.shape
    z = torch.randn(B, L, 1, dtype=Ks.dtype, device=Ks.device, generator=gen)

    if jitter > 0.0:
        scale = Ks.diagonal(dim1=-2, dim2=-1).mean(-1)          # (B,)
        Ks = Ks + (jitter * scale).view(B, 1, 1) * torch.eye(
            L, dtype=Ks.dtype, device=Ks.device)

    # cholesky_ex returns an info code instead of raising, so the whole batch
    # can be attempted at once and only the failures routed to eigh.
    Lc, info = torch.linalg.cholesky_ex(Ks)
    okm = info == 0
    out = torch.empty(B, L, dtype=Ks.dtype, device=Ks.device)

    if okm.any():
        out[okm] = (Lc[okm] @ z[okm]).squeeze(-1)

    nbad = int((~okm).sum())
    if nbad:
        # Rank-deficient / numerically indefinite: a degenerate Gaussian, which
        # is a perfectly valid GP prior. Clamp negative eigenvalues (they are
        # round-off, the true spectrum is >= 0) and use Q diag(sqrt(w)).
        w, Q = torch.linalg.eigh(Ks[~okm])
        sw = w.clamp(min=0.0).sqrt()
        out[~okm] = (Q @ (sw.unsqueeze(-1) * z[~okm])).squeeze(-1)

    return out, nbad


# ---------------------------------------------------------------------------
# self-test -- run this before trusting the output
# ---------------------------------------------------------------------------

def self_test(length: int = 128, device: str = "cpu") -> int:
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel, DotProduct, ExpSineSquared, RationalQuadratic,
        WhiteKernel,
    )
    print(f"self-test at L={length} on {device}\n")
    x = torch.linspace(0, 1, length, dtype=torch.float64, device=device)
    xn = x.cpu().numpy()[:, None]
    bank = kernel_bank(length)

    sk = {
        "expsine": lambda p: ExpSineSquared(length_scale=p["length_scale"],
                                            periodicity=p["periodicity"]),
        "dot":     lambda p: DotProduct(sigma_0=p["sigma_0"]),
        "rbf":     lambda p: RBF(length_scale=p["length_scale"]),
        "ratquad": lambda p: RationalQuadratic(alpha=p["alpha"],
                                               length_scale=p["length_scale"]),
        "white":   lambda p: WhiteKernel(noise_level=p["noise_level"]),
        "const":   lambda p: ConstantKernel(constant_value=p["constant_value"]),
    }

    worst, fails = 0.0, 0
    for kind, prm in bank:
        ours = gram(kind, prm, x).cpu().numpy()
        theirs = sk[kind](prm)(xn)
        e = float(np.abs(ours - theirs).max())
        worst = max(worst, e)
        if e > 1e-9:
            fails += 1
            print(f"  FAIL {kind:<9} {prm}  max|diff| {e:.3e}")
    print(f"  {'ok' if not fails else 'FAIL'}  all {len(bank)} kernels match "
          f"sklearn, worst max|diff| {worst:.3e}")

    # The sampler must reproduce the covariance it was given. Factorise ONE
    # matrix and reuse the factor for many draws -- expanding K itself would
    # materialise n_draws copies of an L x L matrix for no reason.
    print()
    g = torch.Generator(device=device).manual_seed(0)
    n_draw = 20000
    K = gram("rbf", {"length_scale": 0.2}, x)

    # NOTE: this K is NOT numerically positive-definite. A plain RBF with
    # length_scale 0.2 on 128 points spaced 1/127 apart is smooth enough to be
    # effectively low-rank, so torch.linalg.cholesky raises on it. That is not
    # a bug in the kernel -- it is the rank deficiency we measured at 83 % of
    # composites, showing up in the simplest possible case. It is exactly why
    # the sampler needs the eigh path, and why sklearn's "retry with a new
    # seed" policy discards so much.
    chol_ok = torch.linalg.cholesky_ex(K.unsqueeze(0))[1].item() == 0

    # Build the factor the way sample_batch does for a failing matrix, then
    # draw many samples from that one factor.
    w, Q = torch.linalg.eigh(K)
    A = Q @ torch.diag(w.clamp(min=0.0).sqrt())
    z = torch.randn(n_draw, length, dtype=K.dtype, device=device, generator=g)
    S = z @ A.T
    emp = (S.T @ S) / n_draw
    rel = float((emp - K).abs().max() / K.abs().max())

    # and confirm the real entry point handles it without raising
    S1, nbad_rbf = sample_batch(K.unsqueeze(0).repeat(4, 1, 1), g)
    ok2 = rel < 0.05 and torch.isfinite(S1).all() and nbad_rbf == 4
    print(f"  {'ok' if ok2 else 'FAIL'}  RBF(0.2): Cholesky "
          f"{'succeeds' if chol_ok else 'FAILS (rank-deficient)'}, eigh path used "
          f"for {nbad_rbf}/4")
    print(f"       empirical covariance of {n_draw // 1000}k samples matches K to "
          f"{rel * 100:.2f}% (sampling floor "
          f"{100 / math.sqrt(n_draw):.1f}%)")

    # A deliberately rank-1 kernel must still sample, via the eigh path.
    # ConstantKernel gives K = ones: rank 1, not positive definite, so Cholesky
    # must fail and eigh must take over. Each sample is then a constant series.
    K1 = gram("const", {"constant_value": 1.0}, x)
    S1, nbad1 = sample_batch(K1.unsqueeze(0).repeat(8, 1, 1), g)
    flat = float(S1.std(dim=1).max())
    ok3 = nbad1 == 8 and flat < 1e-6 and torch.isfinite(S1).all()
    print(f"  {'ok' if ok3 else 'FAIL'}  rank-1 kernel: {nbad1}/8 took the eigh "
          f"path, samples constant in time (max within-series std {flat:.2e})")

    bad = fails or not ok2 or not ok3
    print(f"\n{'SELF-TEST FAILED' if bad else 'self-test passed'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate(n: int, length: int, out_dir: str, device: str, seed: int,
             max_kernels: int, batch: int, name: str = "kernel",
             jitter: float = 0.0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    bank = kernel_bank(length)
    x = torch.linspace(0, 1, length, dtype=torch.float64, device=dev)

    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)

    series = np.empty((n, length), dtype=np.float32)
    done = 0
    t0 = time.time()
    tot_bad = 0
    build_s = fact_s = 0.0

    print(f"{name}: {n:,} series x {length} on {device}, "
          f"batch {batch}, max_kernels {max_kernels}, jitter {jitter:g}")
    print(f"  bank: {len(bank)} kernels   "
          f"covariance {length}x{length} float64 = "
          f"{length * length * 8 / 2**20:.0f} MB each, "
          f"{batch * length * length * 8 / 2**30:.2f} GB per batch")

    while done < n:
        b = min(batch, n - done)

        t = time.time()
        Ks = torch.stack([composite(rng, bank, x, max_kernels) for _ in range(b)])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        build_s += time.time() - t

        t = time.time()
        S, nbad = sample_batch(Ks, gen, jitter)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        fact_s += time.time() - t

        tot_bad += nbad
        series[done:done + b] = S.float().cpu().numpy()
        del Ks, S
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        done += b
        el = time.time() - t0
        print(f"  {done:>7,}/{n:,}  {el:6.1f}s  "
              f"{done / el:7.1f} series/s  "
              f"eigh fallback {tot_bad}/{done} ({100 * tot_bad / done:.1f}%)",
              flush=True)

    # NaN/Inf guard: a product of several DotProduct kernels can overflow
    finite = np.isfinite(series).all(axis=1)
    if not finite.all():
        print(f"  !! dropping {int((~finite).sum())} series with NaN/Inf")
        series = series[finite]

    npy = os.path.join(out_dir, f"{name}.npy")
    np.save(npy, series)
    h = hashlib.sha256(open(npy, "rb").read()).hexdigest()
    json.dump({"generator": name, "n_series": int(len(series)), "length": length,
               "dtype": "float32", "sha256": h, "bytes": int(series.nbytes),
               "arrow_files": 0, "series_in_arrow": int(len(series)),
               "requested": n, "source": "gen_kernel_gpu.py (TempoPFN bank, "
               "GPU Cholesky/eigh sampling)", "seed": seed,
               "max_kernels": max_kernels, "jitter": jitter,
               "eigh_fallback_frac": tot_bad / max(n, 1)},
              open(npy.replace(".npy", ".json"), "w"), indent=2)

    el = time.time() - t0
    print(f"\n  {len(series):,} series in {el:.1f}s  "
          f"({len(series) / el:.1f}/s)")
    print(f"  covariance build {build_s:.1f}s ({100 * build_s / el:.0f}%)  "
          f"factorisation {fact_s:.1f}s ({100 * fact_s / el:.0f}%)")
    print(f"  eigh fallback on {tot_bad:,} of {n:,} "
          f"({100 * tot_bad / max(n, 1):.1f}%)")
    print(f"  -> {npy}  {series.nbytes / 2**30:.2f} GB  sha256 {h[:16]}...")
    print(f"  extrapolated: 75,000 series would take "
          f"{75000 / (len(series) / el) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--length", type=int, default=2304)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-kernels", type=int, default=5)   # TempoPFN's default
    p.add_argument("--batch", type=int, default=64,
                   help="covariances factorised at once; 64 x 2304^2 float64 = 2.7 GB")
    p.add_argument("--name", default="kernel")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="diagonal jitter as a fraction of mean(diag(K)). "
                        "0 = exact (eigh for ~80%%). 1e-8 = 0.01%% added "
                        "noise, most matrices then pass Cholesky.")
    a = p.parse_args()

    if a.self_test:
        sys.exit(self_test(length=128, device="cpu"))
    if not a.out:
        p.error("--out is required (or pass --self-test)")

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda but torch.cuda.is_available() is False.\n"
                 "  This venv probably has CPU-only torch. Check with:\n    python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
                 "  A '+cpu' build has no CUDA. Use the Reverso venv instead,\n"
                 "  or pass --device cpu deliberately (about 25x slower).")
    if a.device == "cuda":
        d = torch.cuda.get_device_properties(0)
        print(f"GPU: {d.name}  sm_{d.major}{d.minor}  "
              f"{d.total_memory / 2**30:.0f} GB\n")

    generate(a.n, a.length, a.out, a.device, a.seed, a.max_kernels,
             a.batch, a.name, a.jitter)


if __name__ == "__main__":
    main()
