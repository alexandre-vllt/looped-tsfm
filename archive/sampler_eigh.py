"""Eigendecomposition sampler for KernelSynth — the TempoPFN numerics.

Reverso samples its Gaussian processes with a batched Cholesky and escalating
jitter when a matrix is not positive definite. Its Table 9 kernel bank has no
WhiteKernel, and composite kernels built from it are very often rank-deficient
(Constant is rank 1, Linear with sigma=0 is rank 1, products of narrow RBF and
periodic kernels are near-singular), so the fallback fires constantly. Measured
on the paper's own bank at the paper's own length L=4096:

    73.3 %  of matrices need >= 1 jitter escalation  -> noise at  10 % of signal std
     4.4 %  need a second escalation                 -> noise at 100 % of signal std

Those series are not draws from the kernel that was sampled. They are draws from
that kernel plus white noise the recipe never specified.

TempoPFN does not do this. Its KernelSynth calls

    gpr = GaussianProcessRegressor(kernel=kernel)
    ts  = gpr.sample_y(X, n_samples=1, random_state=seed)

which routes through numpy's multivariate_normal, and that factorises the
covariance by SVD rather than Cholesky. SVD does not fail on a rank-deficient
PSD matrix: the null directions get singular value 0, sqrt(0) = 0, and the
sample lands exactly in the range space. A correct draw from a degenerate
Gaussian, with no jitter. On the rare genuine numerical failure TempoPFN retries
with a new seed rather than regularising.

This module does the same thing on GPU, with eigh instead of SVD -- for a
symmetric matrix eigh is exact and about 3x cheaper (measured: eigh 4.7x
cholesky, svd 14x cholesky).

    K = Q diag(w) Q^T ,  w clamped at 0
    x = mu + Q diag(sqrt(w)) z ,  z ~ N(0, I)

Accuracy in float32: |QΛQ^T − K|max / |K|max = 1.9e-4, i.e. ~500x cleaner than
the 10 % noise the jitter injects. float64 reaches 1e-13 and is available via
--eig-dtype if ever needed.

Cost: ~4.7x the Cholesky path for the KernelSynth fraction. At L=2048 that is
comfortable; at L=4096 use --batch 8 and chain jobs.
"""
import torch

import synthetic as S

# Diagnostics, the honest replacement for the jitter counters. With this sampler
# nothing ever "fails", so the useful quantity is how degenerate the kernels
# actually are.
EIG_STATS = dict(
    matrices=0,          # how many covariance matrices factorised
    rank_deficient=0,    # how many had numerical rank < L
    rank_sum=0,          # sum of numerical ranks, for the mean
    dim_sum=0,           # sum of L, so mean rank fraction is rank_sum/dim_sum
    neg_clamped=0,       # eigenvalues below 0 clamped to 0 (roundoff)
    min_eig=0.0,         # most negative eigenvalue seen, relative to the largest
)


def reset_stats():
    EIG_STATS.update(matrices=0, rank_deficient=0, rank_sum=0, dim_sum=0,
                     neg_clamped=0, min_eig=0.0)


def kernelsynth_eigh(B, L, dev, rng, dtype=torch.float32, rank_tol=1e-8):
    """Algorithm 1, sampled by eigendecomposition. Returns (B, L) float32.

    Drop-in replacement for S.kernelsynth. Same kernel bank, same composition
    rule, same mean function -- only the factorisation changes. No jitter, no
    fallback, no failure path.
    """
    t = torch.linspace(0.0, 1.0, L, device=dev, dtype=torch.float32)
    r = (t[:, None] - t[None, :]).abs()
    r2 = r * r

    Ks = torch.stack([S._compose(rng, t, r, r2, dev) for _ in range(B)])
    Ks = 0.5 * (Ks + Ks.transpose(-1, -2))          # enforce exact symmetry
    mus = torch.stack([S._mean(rng, t) for _ in range(B)])

    if dtype == torch.float64:
        Ks = Ks.double()

    # ascending eigenvalues; Q columns are the eigenvectors
    w, Q = torch.linalg.eigh(Ks)

    # ---- diagnostics ----------------------------------------------------
    top = w[:, -1:].clamp(min=1e-30)
    rank = (w > rank_tol * top).sum(-1)
    EIG_STATS["matrices"] += B
    EIG_STATS["dim_sum"] += B * L
    EIG_STATS["rank_sum"] += int(rank.sum())
    EIG_STATS["rank_deficient"] += int((rank < L).sum())
    EIG_STATS["neg_clamped"] += int((w < 0).sum())
    rel_min = float((w.min(-1).values / top.squeeze(-1)).min())
    EIG_STATS["min_eig"] = min(EIG_STATS["min_eig"], rel_min)

    # ---- sample ----------------------------------------------------------
    # x = Q diag(sqrt(w+)) z  -- negative eigenvalues are float roundoff on
    # directions of true zero variance, so clamping is the correct projection
    # onto the PSD cone, not an approximation.
    sw = w.clamp(min=0.0).sqrt()                     # (B, L)
    z = torch.randn(B, L, 1, device=dev, dtype=Ks.dtype)
    x = (Q @ (sw.unsqueeze(-1) * z)).squeeze(-1)     # (B, L)

    return (mus + x.to(mus.dtype)).float()


def report():
    """One-line-per-fact summary, for the end of a generation run."""
    st = EIG_STATS
    m = max(st["matrices"], 1)
    d = max(st["dim_sum"], 1)
    return (
        "conditioning of the composite kernels (eigh sampler, no jitter):\n"
        f"  matrices factorised      {st['matrices']:,}\n"
        f"  rank-deficient           {st['rank_deficient']:,} "
        f"({100.0 * st['rank_deficient'] / m:.1f} %)\n"
        f"  mean numerical rank      {st['rank_sum'] / m:.1f} "
        f"({100.0 * st['rank_sum'] / d:.1f} % of L)\n"
        f"  negative eigenvalues clamped {st['neg_clamped']:,} "
        f"({st['neg_clamped'] / m:.1f} per matrix)\n"
        f"  most negative eigenvalue {st['min_eig']:.2e} (relative to the largest)\n"
        f"  noise injected           0"
    )
