"""Algorithm 4 — Pretraining Data Augmentation Pipeline (Reverso, appendix D.2).

Order is the paper's, and it matters:

    1. multi-scale downsampling      on the full raw series
    2. amplitude modulation          on the full raw series
    3. slicing to context length L
    4. flip augmentations            on the sliced sequence (y then x)
    5. censor augmentation           on the sliced sequence
    6. mixup                         on the assembled batch

The probabilities are listed as free hyperparameters in Algorithm 4 and are
never given a value in the paper. The defaults in CFG are ours — change them
in one place once the authors answer.
"""
import numpy as np
import torch

# --------------------------------------------------------------------------
# UNSPECIFIED in the paper. Our defaults.
# --------------------------------------------------------------------------
CFG = dict(
    p_downsample=0.5,
    k_range=(2, 8),        # [k_min, k_max] downsample ratios
    p_modulate=0.5,
    p_flip_x=0.5,          # temporal reversal
    p_flip_y=0.5,          # sign inversion
    p_censor=0.2,
    mixup_alpha=0.5,       # Beta(alpha, alpha)
)


def _downsample(x, k):
    """Take every k-th point. x is 1-D."""
    return x[::k]


def _amplitude_modulation(x, rng):
    """Multiply by a piecewise-linear function with a single changepoint.

    Algorithm 4: x1 = 0, x3 = len(X)-1, changepoint x2 ~ U{1..len-2},
    scalars y1, y2, y3 ~ N(1, 0.5).
    """
    n = len(x)
    if n < 3:
        return x
    x2 = int(rng.integers(1, n - 1))
    y1, y2, y3 = rng.normal(1.0, 0.5, size=3)
    f = np.empty(n, dtype=np.float64)
    f[:x2 + 1] = np.linspace(y1, y2, x2 + 1)
    f[x2:] = np.linspace(y2, y3, n - x2)
    return x * f


def _slice_or_pad(x, L, rng):
    """Algorithm 4, step 3."""
    n = len(x)
    if n > L:
        s = int(rng.integers(0, n - L + 1))
        return x[s:s + L]
    if n == L:
        return x
    # Pad(X, L): §2.1 says short sequences are back-filled with the leftmost
    # available data point.
    out = np.empty(L, dtype=x.dtype)
    out[: L - n] = x[0]
    out[L - n:] = x
    return out


def _censor(x, rng):
    """Algorithm 4, step 5: clip from the top, the bottom, or neither."""
    q = rng.uniform(0.0, 1.0)
    c = np.quantile(x, q)
    d = rng.choice(["top", "bottom", "none"])
    if d == "top":
        return np.minimum(x, c)
    if d == "bottom":
        return np.maximum(x, c)
    return x


def augment_series(x, L, rng, cfg=CFG):
    """Steps 1-5 on a single raw series, returning a length-L window."""
    x = np.asarray(x, dtype=np.float64)

    if rng.random() < cfg["p_downsample"]:
        k = int(rng.integers(cfg["k_range"][0], cfg["k_range"][1] + 1))
        if len(x) // k >= 8:
            x = _downsample(x, k)

    if rng.random() < cfg["p_modulate"]:
        x = _amplitude_modulation(x, rng)

    x = _slice_or_pad(x, L, rng)

    if rng.random() < cfg["p_flip_y"]:
        x = -x
    if rng.random() < cfg["p_flip_x"]:
        x = x[::-1].copy()

    if rng.random() < cfg["p_censor"]:
        x = _censor(x, rng)

    return x.astype(np.float32)


def mixup(batch, rng, alpha=None):
    """Algorithm 4, step 6: batch-level convex interpolation.

    Note the paper draws lambda ~ Beta(alpha, alpha) **per sample**, so each
    row gets its own mixing weight.
    """
    alpha = CFG["mixup_alpha"] if alpha is None else alpha
    B = batch.shape[0]
    perm = torch.from_numpy(rng.permutation(B)).to(batch.device)
    lam = torch.from_numpy(
        rng.beta(alpha, alpha, size=B).astype(np.float32)
    ).to(batch.device).view(B, *([1] * (batch.ndim - 1)))
    return lam * batch + (1.0 - lam) * batch[perm]


def make_batch(raw_iter, B, L, horizon, rng, device="cuda", cfg=CFG,
               apply_mixup=True):
    """Assemble one training batch of (context, horizon) from raw series.

    `raw_iter` yields 1-D numpy arrays of length >= L + horizon.

    Mixup is applied to the concatenated (context + horizon) window so that
    the target stays consistent with the mixed context.
    """
    need = L + horizon
    rows = []
    while len(rows) < B:
        rows.append(augment_series(next(raw_iter), need, rng, cfg))
    t = torch.from_numpy(np.stack(rows)).to(device)
    if apply_mixup:
        t = mixup(t, rng, cfg["mixup_alpha"])
    return t[:, :L, None], t[:, L:, None]
