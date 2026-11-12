"""Reverso synthetic data generators — faithful to the paper's appendix D.

Three generators, following the paper exactly where it is explicit:

  kernelsynth(...)  Algorithm 1 + kernel bank of Table 9
  spike(...)        Algorithm 2 (adapted from Kairos, Feng et al. 2026)
  tsi(...)          Algorithm 3 (TSI, following Chronos-2)

Everything marked "PAPER" below is taken from the text. Everything marked
"UNSPECIFIED" is a value the paper leaves as a free hyperparameter; those are
grouped at the top so they can be corrected in one place once the authors
answer.
"""
import math

import numpy as np
import torch

# ==========================================================================
# PAPER — appendix D.1 and Table 9
# ==========================================================================

# "We use the following set of periods P = {...} (normalized by time series
# length L_syn) to capture patterns of various timescales."
PERIODS = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730,
           4, 26, 52, 6, 12, 40, 10]

# Table 9 — Kernel Bank K used for Synthetic Data Generation
BASE_BANK = (
    [("constant", dict(C=1.0))]
    + [("linear", dict(sigma=s)) for s in (0.0, 1.0, 10.0)]
    + [("rbf", dict(l=l)) for l in (0.1, 1.0, 10.0)]
    + [("rq", dict(alpha=a)) for a in (0.1, 1.0, 10.0)]
    + [("matern", dict(nu=nu, l=l))
       for nu in (0.5, 1.5, 2.5) for l in (0.1, 1.0, 10.0)]
    + [("periodic", dict(p=p)) for p in PERIODS]
)

# NOT in Reverso's Table 9 — but present in both Chronos and TempoPFN, whose
# banks "include periodic kernels, stationary kernels (RBF, RationalQuadratic),
# and noise kernels (WhiteKernel)". A WhiteKernel adds a positive constant to
# the diagonal of K, which is mathematically what our Cholesky jitter does —
# except as a declared modelling choice rather than a numerical patch. Reverso
# removed it AND switched to batched Cholesky; that combination is what makes
# ~14 of 16 composite kernels rank-deficient.
WHITE_BANK = [("white", dict(noise_level=n)) for n in (0.1, 1.0)]

KERNEL_BANK = list(BASE_BANK)


def use_white_kernel(on=True):
    """Add or remove the WhiteKernel from the bank. Returns the bank size."""
    global KERNEL_BANK
    KERNEL_BANK = list(BASE_BANK) + (WHITE_BANK if on else [])
    return len(KERNEL_BANK)


# Conditioning diagnostics, filled in by kernelsynth()
STATS = dict(matrices=0, chol_fail=0, escalations=0, fallback=0)

J_MAX = 5           # "sample j ~ U{1,5} kernels from K"
MAX_LEN = 4096      # "The maximum sequence length is set to 4096."
N_SYNTHETIC = 1_000_000   # "We generate a total of 1 million synthetic ... sequences"

# Algorithm 1, mean function
TREND_PROB = 0.5            # "with probability 1/2 and constant otherwise"
TREND_SLOPE = (-0.01, 0.01)  # m ~ U[-0.01, 0.01]
TREND_INTERCEPT = (-0.1, 0.1)  # c ~ U[-0.1, 0.1]

# ==========================================================================
# UNSPECIFIED — the paper lists these as inputs to Algorithms 2 and 3 but
# gives no values. The defaults below are ours. Ask the authors.
# ==========================================================================
SPIKE = dict(
    types=("inverted_u", "spikes"),
    baseline=(-1.0, 1.0),        # [b_min, b_max]
    period=(10, 500),            # [p_min, p_max], in samples
    amplitude=(1.0, 10.0),       # [a_min, a_max]
    width=(3, 60),               # [w_min, w_max], in samples, clipped to < p
    noise=(0.0, 0.1),            # [sigma_min, sigma_max]
)

TSI = dict(
    p_trend=0.5, p_seas=0.7, p_noise=0.8, p_out=0.10, p_shift=0.10,
    trend_types=("linear", "exp", "poly", "piecewise"),
    wave_forms=("sine", "sawtooth", "square"),
    n_components=(1, 3),         # K ~ U{1,3}   (this one IS in the paper)
    amplitude=(0.1, 5.0),
    noise_scale=(0.01, 0.5),
    outlier_frac=(0.001, 0.02),
    outlier_scale=(3.0, 10.0),
    n_shifts=(1, 3),
    shift_scale=(0.5, 5.0),
)

# Mixture weights between the three generators — also unspecified.
MIX = dict(kernelsynth=0.6, spike=0.2, tsi=0.2)


# ==========================================================================
# Kernels (Table 9), evaluated on points evenly spaced in [0, 1]
# ==========================================================================
def _kernel(name, prm, t, r, r2, dev):
    """(L, L) covariance matrix. r = |x - x'|, r2 = r**2."""
    L = t.numel()
    if name == "constant":
        return prm["C"] * torch.ones(L, L, device=dev)
    if name == "linear":                       # sigma^2 + x . x'
        s = prm["sigma"]
        return s * s + t[:, None] * t[None, :]
    if name == "rbf":                          # exp(-r^2 / (2 l^2))
        l = prm["l"]
        return torch.exp(-r2 / (2.0 * l * l))
    if name == "rq":                           # (1 + r^2/(2 alpha))^-alpha
        a = prm["alpha"]
        return (1.0 + r2 / (2.0 * a)) ** (-a)
    if name == "matern":                       # closed forms for nu = 1/2, 3/2, 5/2
        nu, l = prm["nu"], prm["l"]
        if nu == 0.5:
            return torch.exp(-r / l)
        if nu == 1.5:
            k = math.sqrt(3.0) * r / l
            return (1.0 + k) * torch.exp(-k)
        if nu == 2.5:
            k = math.sqrt(5.0) * r / l
            return (1.0 + k + k * k / 3.0) * torch.exp(-k)
        raise ValueError(f"matern nu={nu}")
    if name == "periodic":                     # exp(-2 sin^2(pi r / p))
        p = prm["p"] / L                       # "normalized by time series length"
        s = torch.sin(math.pi * r / p)
        return torch.exp(-2.0 * s * s)
    if name == "white":                        # noise_level * I  (Chronos, TempoPFN)
        return prm["noise_level"] * torch.eye(L, device=dev)
    raise ValueError(name)


def _compose(rng, t, r, r2, dev):
    """Algorithm 1, lines 1-8: compose j ~ U{1, J_max} kernels."""
    j = int(rng.integers(1, J_MAX + 1))
    idx = rng.integers(0, len(KERNEL_BANK), size=j)
    K = _kernel(*KERNEL_BANK[idx[0]], t, r, r2, dev)
    for i in idx[1:]:
        Ki = _kernel(*KERNEL_BANK[i], t, r, r2, dev)
        K = K + Ki if rng.random() < 0.5 else K * Ki
    return K


def _mean(rng, t):
    """Algorithm 1, lines 9-15."""
    if rng.random() < TREND_PROB:
        m = rng.uniform(*TREND_SLOPE)
        c = rng.uniform(*TREND_INTERCEPT)
        return m * t + c
    return torch.zeros_like(t)


def kernelsynth(B, L, dev, rng, jitter_tries=6, jitter0=1e-4):
    """Algorithm 1, batched. Returns (B, L) float32.

    Several kernels in Table 9 are rank-deficient on their own (Constant is
    all-ones, Linear with sigma=0 is rank 1), and products of kernels can be
    badly conditioned, so a plain Cholesky fails often. We use cholesky_ex,
    which reports failures per matrix instead of raising, and re-try only the
    matrices that failed — escalating the jitter for those alone.
    """
    t = torch.linspace(0.0, 1.0, L, device=dev, dtype=torch.float32)
    r = (t[:, None] - t[None, :]).abs()
    r2 = r * r

    Ks = torch.stack([_compose(rng, t, r, r2, dev) for _ in range(B)])
    Ks = 0.5 * (Ks + Ks.transpose(-1, -2))
    mus = torch.stack([_mean(rng, t) for _ in range(B)])

    eye = torch.eye(L, device=dev)
    scale = Ks.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
    Lc = torch.zeros_like(Ks)
    todo = torch.arange(B, device=dev)
    jit = jitter0
    STATS["matrices"] += B
    for attempt in range(jitter_tries):
        sub = Ks[todo] + (jit * scale[todo])[:, None, None] * eye
        out, info = torch.linalg.cholesky_ex(sub)
        ok = info == 0
        if attempt == 0:
            STATS["chol_fail"] += int((~ok).sum())
        if ok.any():
            Lc[todo[ok]] = out[ok]
        todo = todo[~ok]
        if todo.numel() == 0:
            break
        STATS["escalations"] += int(todo.numel())
        jit *= 100.0
    if todo.numel():
        # hopeless composites: fall back to white noise rather than crash
        STATS["fallback"] += int(todo.numel())
        Lc[todo] = eye.expand(todo.numel(), L, L)

    z = torch.randn(B, L, 1, device=dev)
    return (mus + (Lc @ z).squeeze(-1)).float()


# ==========================================================================
# Algorithm 2 — Spike Process Generation
# ==========================================================================
def spike(B, L, dev, rng):
    out = np.zeros((B, L), dtype=np.float32)
    for n in range(B):
        typ = SPIKE["types"][int(rng.integers(0, len(SPIKE["types"])))]
        b = rng.uniform(*SPIKE["baseline"])
        p = int(rng.integers(SPIKE["period"][0], SPIKE["period"][1] + 1))
        a = rng.uniform(*SPIKE["amplitude"])
        w = int(rng.integers(SPIKE["width"][0], SPIKE["width"][1] + 1))
        w = max(4, min(w, p - 1))
        sig = rng.uniform(*SPIKE["noise"])

        # trapezoid shape e of length w
        u, f = w // 4, w // 2
        d = w - u - f
        e = np.concatenate([np.linspace(0.0, a, u),
                            np.full(f, a, dtype=float),
                            np.linspace(a, 0.0, d)])
        x = np.full(L, b, dtype=float)
        s = -1.0 if typ == "inverted_u" else 1.0
        for i in range(0, L, p):
            ln = min(w, L - i)
            x[i:i + ln] += s * e[:ln]
        out[n] = x + rng.normal(0.0, sig, size=L)
    return torch.from_numpy(out).to(dev)


# ==========================================================================
# Algorithm 3 — TSI (Trend, Seasonality, Irregularity)
# ==========================================================================
def _trend(rng, L, kind):
    t = np.linspace(0.0, 1.0, L)
    if kind == "linear":
        return rng.uniform(-2, 2) * t + rng.uniform(-1, 1)
    if kind == "exp":
        return rng.uniform(-1, 1) * np.exp(rng.uniform(0.5, 3.0) * t)
    if kind == "poly":
        deg = int(rng.integers(2, 4))
        return rng.uniform(-2, 2) * t ** deg + rng.uniform(-1, 1) * t
    if kind == "piecewise":
        k = int(rng.integers(1, 4))
        cuts = np.sort(rng.choice(np.arange(1, L), size=k, replace=False))
        y, prev, val = np.zeros(L), 0, 0.0
        for c in list(cuts) + [L]:
            sl = rng.uniform(-3, 3)
            seg = np.arange(c - prev) / max(L, 1)
            y[prev:c] = val + sl * seg
            val = y[c - 1]
            prev = c
        return y
    raise ValueError(kind)


def _wave(form, phase):
    if form == "sine":
        return np.sin(phase)
    if form == "sawtooth":
        return 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    if form == "square":
        return np.sign(np.sin(phase))
    raise ValueError(form)


def tsi(B, L, dev, rng):
    cfg, out = TSI, np.zeros((B, L), dtype=np.float32)
    t = np.arange(L, dtype=float)
    for n in range(B):
        x = np.zeros(L)
        if rng.random() < cfg["p_trend"]:
            kind = cfg["trend_types"][int(rng.integers(0, len(cfg["trend_types"])))]
            x = x + _trend(rng, L, kind)
        if rng.random() < cfg["p_seas"]:
            K = int(rng.integers(cfg["n_components"][0], cfg["n_components"][1] + 1))
            periods = rng.choice(PERIODS, size=min(K, len(PERIODS)), replace=False)
            for p in periods:
                form = cfg["wave_forms"][int(rng.integers(0, len(cfg["wave_forms"])))]
                A = rng.uniform(*cfg["amplitude"])
                phi = rng.uniform(0, 2 * np.pi)
                x = x + A * _wave(form, 2 * np.pi * t / float(p) + phi)
        if rng.random() < cfg["p_noise"]:
            x = x + rng.normal(0.0, rng.uniform(*cfg["noise_scale"]), size=L)
        if rng.random() < cfg["p_out"]:
            m = max(1, int(rng.uniform(*cfg["outlier_frac"]) * L))
            pos = rng.choice(L, size=m, replace=False)
            sc = rng.uniform(*cfg["outlier_scale"]) * (x.std() + 1e-6)
            x[pos] += sc * rng.choice([-1.0, 1.0], size=m)
        if rng.random() < cfg["p_shift"]:
            k = int(rng.integers(cfg["n_shifts"][0], cfg["n_shifts"][1] + 1))
            cuts = np.sort(rng.choice(np.arange(1, L), size=k, replace=False))
            for c in cuts:
                x[c:] += rng.uniform(-1, 1) * rng.uniform(*cfg["shift_scale"])
        out[n] = x
    return torch.from_numpy(out).to(dev)


# ==========================================================================
def mixed_batch(B, L, dev, rng):
    """One batch drawn from the three generators according to MIX."""
    n_k = int(round(B * MIX["kernelsynth"]))
    n_s = int(round(B * MIX["spike"]))
    n_t = B - n_k - n_s
    parts = []
    if n_k:
        parts.append(kernelsynth(n_k, L, dev, rng))
    if n_s:
        parts.append(spike(n_s, L, dev, rng))
    if n_t:
        parts.append(tsi(n_t, L, dev, rng))
    x = torch.cat(parts, 0)
    return x[torch.randperm(B, device=dev)]
