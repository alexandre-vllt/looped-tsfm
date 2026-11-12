"""Algorithm 4 -- pretraining data augmentation (Reverso appendix D.2).

Wired into the deterministic pipeline on 12 October 2026. The standalone
`augment.py` at the repo root was the reading of the paper; this is the version
that runs, and it differs from it in three ways that are all forced by our
corpus, not by taste.

THE ORDER IS THE PAPER'S AND IT MATTERS
    1. multi-scale downsampling   on the full raw series   -- OFF, see below
    2. amplitude modulation       on the full raw series
    3. slice to context length    (span = seq_len + horizon)
    4. flips, y then x            on the sliced window
    5. censoring                  on the sliced window
    6. mixup                      on the assembled batch   -- in data.py's collate

WHY DOWNSAMPLING IS OFF
A sample spans seq_len + horizon = 2096 points. Our series are stored at 2304
(done.md 5.2, "Why L = 2304"). Taking every k-th point with k >= 2 leaves at
most 1152 points, fewer than one sample needs, so Pad() back-fills ~944 points
with x[0] -- a long flat prefix followed by real data. That is precisely the
flat->jump window the loss guard of done.md 9.1 drops, so the augmentation
would spend half the batch manufacturing samples the loss then throws away.
`max_downsample_k` computes what the corpus can actually support; at 2304/2096
it returns 1, meaning none. The paper's raw series are p50 8192, where k up to
3 is free -- this is a consequence of our chunking, not a disagreement with
Algorithm 4. Rebuild the corpus at a longer L and p_downsample becomes usable;
until then, setting it above 0 raises.

DETERMINISM
Every draw here comes from the `rng` handed in, which is
Philox(key=seed, counter=ordinal) from DeterministicMix. Nothing touches global
state, so sample n stays a pure function of (n, seed) and the stream remains
independent of worker count. The draw ORDER below is part of the contract:
reordering it changes every sample.

THE PROBABILITIES ARE NOT IN THE PAPER
Algorithm 4 lists them as free hyperparameters and never gives a value. CFG is
ours. Arber has been asked for the TempoPFN generator, which would settle them.
"""
from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------
# OURS -- the paper specifies none of these.
# --------------------------------------------------------------------------
CFG = dict(
    p_downsample=0.0,      # forced to 0 by series_len 2304; see the docstring
    k_range=(2, 8),
    p_modulate=0.5,
    p_flip_y=0.5,          # sign inversion
    p_flip_x=0.5,          # temporal reversal
    p_censor=0.2,
    p_mixup=1.0,           # probability a BATCH is mixed, not a sample
    mixup_alpha=0.5,       # Beta(alpha, alpha), drawn per sample
)

DEGEN = 1e-5               # same threshold as the loss guard, done.md 9.1


def max_downsample_k(series_len: int, span: int) -> int:
    """Largest k for which every downsampled series still fills one window."""
    return max(1, int(series_len) // int(span))


def check_cfg(cfg: dict, series_len: int, span: int) -> dict:
    """Refuse a configuration the corpus cannot honour. Called once at startup."""
    cfg = {**CFG, **(cfg or {})}
    if cfg["p_downsample"] > 0:
        kmax = max_downsample_k(series_len, span)
        if kmax < 2:
            raise ValueError(
                f"p_downsample={cfg['p_downsample']} but series are {series_len} "
                f"points and a sample needs {span}: no k >= 2 leaves a full "
                f"window, so every downsampled sample would be mostly padding. "
                f"Regenerate the corpus at L >= {2 * span} or leave it at 0.")
        cfg["k_range"] = (cfg["k_range"][0], min(cfg["k_range"][1], kmax))
    return cfg


def describe(cfg: dict) -> str:
    on = [k for k in ("p_downsample", "p_modulate", "p_flip_y", "p_flip_x",
                      "p_censor", "p_mixup") if cfg.get(k, 0) > 0]
    return " ".join(f"{k[2:]}={cfg[k]:g}" for k in on) or "nothing enabled"


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def _amplitude_modulation(x, rng):
    """Piecewise-linear gain with one changepoint (Algorithm 4, step 2).

    x1 = 0, x3 = len-1, changepoint x2 ~ U{1..len-2}, gains y1,y2,y3 ~ N(1, 0.5).
    """
    n = len(x)
    if n < 3:
        return x
    x2 = int(rng.integers(1, n - 1))
    y1, y2, y3 = rng.normal(1.0, 0.5, size=3)
    f = np.empty(n, dtype=np.float64)
    f[: x2 + 1] = np.linspace(y1, y2, x2 + 1)
    f[x2:] = np.linspace(y2, y3, n - x2)
    return x * f


def _slice(x, span, rng):
    """Algorithm 4 step 3. Pad() back-fills with the leftmost point (2.1)."""
    n = len(x)
    if n > span:
        s = int(rng.integers(0, n - span + 1))
        return x[s: s + span]
    if n == span:
        return x
    out = np.empty(span, dtype=x.dtype)
    out[: span - n] = x[0]
    out[span - n:] = x
    return out


def _censor(x, rng):
    """Algorithm 4 step 5: clip from the top, the bottom, or neither."""
    q = float(rng.uniform(0.0, 1.0))
    c = float(np.quantile(x, q))
    d = int(rng.integers(3))          # integers, not choice: choice costs 10x here
    if d == 0:
        return np.minimum(x, c)
    if d == 1:
        return np.maximum(x, c)
    return x


def augment_window(row, span, rng, cfg, seq_len=None, stats=None):
    """Steps 1-5 on one full stored series -> one float32 window of `span`.

    `row` is the WHOLE series as stored, not a pre-sliced window: steps 1 and 2
    are defined on the raw series and step 3 does the slicing.

    Falls back to a plain centred slice if a step produced a non-finite value.
    A single NaN would otherwise poison the whole forward pass silently --
    Model.forward does x.min(1)/x.max(1), and min/max over a NaN return NaN.
    """
    x = np.asarray(row, dtype=np.float64)
    raw = x

    if cfg["p_downsample"] > 0 and rng.random() < cfg["p_downsample"]:
        k = int(rng.integers(cfg["k_range"][0], cfg["k_range"][1] + 1))
        x = x[::k]

    if rng.random() < cfg["p_modulate"]:
        x = _amplitude_modulation(x, rng)

    x = _slice(x, span, rng)

    if rng.random() < cfg["p_flip_y"]:
        x = -x
    if rng.random() < cfg["p_flip_x"]:
        x = x[::-1].copy()

    if rng.random() < cfg["p_censor"]:
        x = _censor(x, rng)

    if not np.all(np.isfinite(x)):
        if stats is not None:
            stats["nonfinite"] = stats.get("nonfinite", 0) + 1
        s = max(0, (len(raw) - span) // 2)
        x = _slice(raw[s:], span, rng)

    if stats is not None:
        stats["n"] = stats.get("n", 0) + 1
        L = span if seq_len is None else seq_len
        ctx = x[:L]
        if float(ctx.max() - ctx.min()) <= DEGEN:
            stats["degenerate"] = stats.get("degenerate", 0) + 1

    return x.astype(np.float32)


def mixup_batch(t, rng, alpha):
    """Algorithm 4 step 6, on the assembled batch.

    Applied to the concatenated (context + horizon) window so the target stays
    consistent with the context it was mixed with. lambda ~ Beta(a, a) is drawn
    PER SAMPLE, as the paper writes it, not once for the batch.
    """
    B = t.shape[0]
    perm = torch.from_numpy(rng.permutation(B)).long()
    lam = torch.from_numpy(rng.beta(alpha, alpha, size=B).astype(np.float32))
    lam = lam.view(B, *([1] * (t.ndim - 1)))
    return lam * t + (1.0 - lam) * t[perm]
