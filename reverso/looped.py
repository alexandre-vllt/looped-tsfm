"""
Looped Reverso: weight-tied block iteration over the torch-native model.

    h^(0) = embedding(x)
    h^(t+1) = beta * f(h^(t)) + (1 - beta) * h^(t)      f = layers[loop_slice]
    y^(t)   = decode(h^(t))                             t = 1..k, shared head

Implements task.md sections 3, 4, 6 and 12. Subclasses reverso_torch.model.Model
rather than editing it, so the unmodified upstream forward() stays available as
the reference for the bit-identity test (task.md 13.3).

Design notes
------------
beta == 1.0 takes an exact shortcut (h = f(h)) instead of evaluating
1.0 * f(h) + 0.0 * h. Both are mathematically f(h), but only the shortcut is
guaranteed bit-identical: `a + (b - a) != b` in floating point, and
`1.0 * f + 0.0 * h` differs from `f` when f contains -0.0. The shortcut also
means loop_k=1, loop_beta=1.0 executes exactly the original graph, so R0 is the
unmodified baseline rather than a numerically-close imitation.

The head is OUTSIDE the loop by necessity: f is shape-preserving
(B, d_model, seq_len) -> (B, d_model, seq_len), the head is the only
shape-changing map (d_model x 2048 -> 48). See task.md section 4.

No new parameters are introduced. R0 and R1 have identical parameter counts;
test_loop_identity.py asserts this.
"""
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from reverso_torch.model import Model


class LoopedModel(Model):
    """Reverso with the block stack applied `loop_k` times under shared weights.

    Extra config fields (all optional, defaults reproduce the original model):

    loop_k        int    iterations of the shared block. 1 = original.
    loop_beta     float  damping coefficient. 1.0 = bare loop (h <- f(h)).
                         task.md 3.1: this IS the residual scaling epsilon of [05],
                         which prescribes 1/k under weight tying.
    loop_start    int    first index of self.layers inside the loop (default 0)
    loop_end      int    one past the last index    (default len(self.layers))

    loop_start/loop_end exist so option 1.1b of task.md -- prelude layers[0:2],
    looped core layers[2:4] -- is a config change rather than a code change.
    """

    def __init__(self, configs):
        super().__init__(configs)

        self.loop_k = int(getattr(configs, "loop_k", 1))
        self.loop_beta = float(getattr(configs, "loop_beta", 1.0))

        n_layers = len(self.layers)
        self.loop_start = int(getattr(configs, "loop_start", 0))
        self.loop_end = int(getattr(configs, "loop_end", n_layers))

        if self.loop_k < 1:
            raise ValueError(f"loop_k must be >= 1, got {self.loop_k}")
        if not 0 <= self.loop_start < self.loop_end <= n_layers:
            raise ValueError(
                f"bad loop slice [{self.loop_start}:{self.loop_end}) "
                f"for {n_layers} layers"
            )

    # -- the three pieces of the original forward --------------------------

    def encode(self, x: torch.Tensor):
        """(B, L, C) -> h (B, d_model, L), plus the denormalisation statistics.

        Byte-for-byte the first half of Model.forward.
        """
        if self.use_norm:
            x_min = x.min(1, keepdim=True)[0].detach()
            x_max = x.max(1, keepdim=True)[0].detach()
            x_range = torch.clamp(x_max - x_min, min=1e-5).detach()
            x = (x - x_min) / x_range
            means, stdev = x_min, x_range
        else:
            means, stdev = None, None

        h = self.embedding(x).transpose(1, 2)
        return h, means, stdev

    def prelude(self, h: torch.Tensor) -> torch.Tensor:
        """layers[:loop_start] -- run once, before the loop. Empty by default."""
        if self.loop_start == 0:
            return h
        return self.layers[: self.loop_start](h)

    def core(self, h: torch.Tensor) -> torch.Tensor:
        """f: one application of the shared block. Shape-preserving."""
        return self.layers[self.loop_start : self.loop_end](h)

    def coda(self, h: torch.Tensor) -> torch.Tensor:
        """layers[loop_end:] -- run once, after the loop. Empty by default."""
        if self.loop_end == len(self.layers):
            return h
        return self.layers[self.loop_end :](h)

    def decode(self, h: torch.Tensor, means, stdev) -> torch.Tensor:
        """h (B, d_model, L) -> y (B, output_token_len, 1).

        Byte-for-byte the second half of Model.forward. Shared across every exit
        (task.md 4: `head`, `simple_q_proj`, `key_proj`, `value_proj`, `out_proj`
        all shared -- Ouro's design, zero extra parameters for adaptive depth).
        """
        temp_out = self.head(h).permute(0, 2, 1)
        q = self.simple_q_proj(temp_out)

        h_perm = h.permute(0, 2, 1)
        k = self.key_proj(h_perm)
        v = self.value_proj(h_perm)

        attn = F.scaled_dot_product_attention(q, k, v)
        out = self.out_proj(attn)

        if self.use_norm:
            out = out * stdev + means
        return out

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        x_mark=None,
        y_mark=None,
        loop_k: Optional[int] = None,
        loop_beta: Optional[float] = None,
        return_all_exits: bool = False,
        return_states: bool = False,
        **kwargs: Any,
    ):
        """
        loop_k / loop_beta override the config per call -- this is what makes
        task.md section 10 free: one trained model evaluated at k = 1, 2, 3, 4.

        Returns
        -------
        return_all_exits=False (default)
            Tensor, the last exit. Same type as Model.forward, so existing
            evaluation code needs no change.
        return_all_exits=True
            List[Tensor] of length k: [y^(1), ..., y^(k)].
        return_states=True
            additionally a dict with the retained h^(t) and f(h^(t)) for the
            section 12 diagnostics.
        """
        k = self.loop_k if loop_k is None else int(loop_k)
        beta = self.loop_beta if loop_beta is None else float(loop_beta)
        if k < 1:
            raise ValueError(f"loop_k must be >= 1, got {k}")

        h, means, stdev = self.encode(x)
        h = self.prelude(h)

        exits: List[torch.Tensor] = []
        states: List[torch.Tensor] = [h] if return_states else []
        raws: List[torch.Tensor] = []

        for t in range(k):
            f_h = self.core(h)

            if beta == 1.0:
                # exact: guarantees bit-identity with the unlooped model at k=1
                h = f_h
            else:
                h = beta * f_h + (1.0 - beta) * h

            if return_states:
                raws.append(f_h)
                states.append(h)

            # An exit at every iteration when asked, otherwise only the last --
            # decoding all k when only y^(k) is used would waste 4x the head.
            # The y^(t) are never fed back; the loop carries h (task.md 4).
            if return_all_exits or t == k - 1:
                exits.append(self.decode(self.coda(h), means, stdev))

        out = exits if return_all_exits else exits[-1]

        if return_states:
            return out, {"h": states, "f_h": raws, "k": k, "beta": beta}
        return out

    def forecast(self, x, x_mark=None, y_mark=None, **kwargs):
        return self.forward(x, x_mark, y_mark, **kwargs)


# ---------------------------------------------------------------------------
# Multi-exit loss (task.md 4.2 and 5)
# ---------------------------------------------------------------------------

def reverso_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reverso's own loss, unchanged: MAE on the unnormalised output, NaN masked."""
    m = ~torch.isnan(y)
    return (pred - y).abs()[m].mean()


def multi_exit_loss(
    exits: List[torch.Tensor],
    y: torch.Tensor,
    mode: str = "uniform",
    loss_fn=None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Combine per-exit losses.

    mode="uniform"  L = mean_t L^(t)          -- R1-uniform
    mode="last"     L = L^(k)                 -- R1-last

    Returns (objective, [L^(1), ..., L^(k)]). The per-exit list must be logged:
    task.md 7.1 takes R1's plateau decision on L^(k), not on the objective,
    because the averaged objective is not comparable to R0's single loss.

    loss_fn   per-exit loss, default `reverso_loss` (paper 2.1: MAE on the
              unnormalised output). train_loop.py passes a scale-free loss
              instead -- see done.md 9 for why and task.md for the deviation.
              The default keeps test_loop_identity.py checks 6c/6d unchanged.
    """
    lf = reverso_loss if loss_fn is None else loss_fn
    per_exit = [lf(p, y) for p in exits]

    if mode == "uniform":
        objective = torch.stack(per_exit).mean()
    elif mode == "last":
        objective = per_exit[-1]
    else:
        raise ValueError(f"unknown loss mode {mode!r}; use 'uniform' or 'last'")

    return objective, per_exit


# ---------------------------------------------------------------------------
# Diagnostics (task.md 12) -- four scalars per iteration, logged every 100 steps
# ---------------------------------------------------------------------------

@torch.no_grad()
def _flatten_samples(h: torch.Tensor, max_batch: int) -> torch.Tensor:
    """(B, D, L) -> (b, D*L) with b <= max_batch. One row per series."""
    b = min(h.shape[0], max_batch)
    return h[:b].reshape(b, -1).float()


@torch.no_grad()
def linear_cka(a: torch.Tensor, b: torch.Tensor, max_batch: int = 64) -> float:
    """Linear CKA between two representations, samples as the shared axis.

    High CKA across iterations means the loop is applying a near-identity: it
    does nothing, whatever the loss says (task.md 12).

    Computed through the b x b Gram matrices, never the p x p covariance.
    With p = d_model * seq_len = 65536, the covariance form would allocate
    17 GB; the identity

        ||Y^T X||_F^2 = tr( (X X^T)(Y Y^T) ) = sum( Kx * Ky )
        ||X^T X||_F   = sqrt( sum(Kx * Kx) ) = ||Kx||_F

    gives the same number from two b x b matrices. Features are centred over
    the sample axis, which is what makes this CKA rather than plain cosine.
    """
    X = _flatten_samples(a, max_batch)
    Y = _flatten_samples(b, max_batch)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    if X.shape[0] < 2:
        return float("nan")      # CKA is undefined for a single sample

    Kx = X @ X.T                 # (b, b)
    Ky = Y @ Y.T                 # (b, b)

    num = (Kx * Ky).sum()
    denom = Kx.norm() * Ky.norm()
    return float(num / denom) if denom > 0 else float("nan")


@torch.no_grad()
def singular_spectrum(h: torch.Tensor, max_rows: int = 8192) -> torch.Tensor:
    """sigma_i / sigma_0 of h in channel space.

    h is (B, d_model, L); flattened to (B*L, d_model) so the spectrum has
    d_model entries and answers "is the representation collapsing into a
    lower-dimensional subspace as it iterates" (task.md 12).
    """
    B, D, L = h.shape
    m = h.permute(0, 2, 1).reshape(B * L, D).float()
    if m.shape[0] > max_rows:
        idx = torch.randperm(m.shape[0], device=m.device)[:max_rows]
        m = m[idx]
    m = m - m.mean(0, keepdim=True)
    s = torch.linalg.svdvals(m)
    return s / s[0].clamp(min=1e-12)


@torch.no_grad()
def effective_rank(spectrum: torch.Tensor) -> float:
    """exp(entropy of the normalised spectrum) -- one number for the collapse."""
    p = spectrum / spectrum.sum().clamp(min=1e-12)
    ent = -(p * p.clamp(min=1e-12).log()).sum()
    return float(ent.exp())


@torch.no_grad()
def loop_diagnostics(states: dict, cka_max_batch: int = 64) -> dict:
    """The four quantities of task.md section 12, per iteration.

    Takes the dict returned by LoopedModel.forward(..., return_states=True).

    update_magnitude  2||f(h)-h||_F / (||f(h)||_F + ||h||_F)
        Do later iterations do any work? MeSH finds this collapses after
        iteration 1 in a bare loop -- which is exactly beta = 1.
    cka               CKA(h^(t), h^(t+1))
        Near-identity detector.
    sigma_ratio       sigma_1/sigma_0 of h^(t), plus effective rank
        Representation collapse.
    convergence       ||h^(t+1) - h^(t)|| / (||h^(t)|| + eps)
        Does a looped forecaster reach a fixed point at all, and how fast?
        task.md: the highest-payoff diagnostic -- it decides whether the whole
        fixed-point branch of the literature applies to time series.
    """
    hs, fs = states["h"], states["f_h"]
    eps = 1e-8
    out = {"k": states["k"], "beta": states["beta"]}

    for t, f_h in enumerate(fs):
        h_t, h_next = hs[t], hs[t + 1]

        nf = f_h.norm()
        nh = h_t.norm()
        out[f"update_magnitude/t{t + 1}"] = float(
            2.0 * (f_h - h_t).norm() / (nf + nh).clamp(min=eps)
        )
        out[f"convergence/t{t + 1}"] = float(
            (h_next - h_t).norm() / (nh + eps)
        )
        out[f"cka/t{t}_t{t + 1}"] = linear_cka(h_t, h_next, cka_max_batch)

    for t, h_t in enumerate(hs):
        spec = singular_spectrum(h_t)
        out[f"sigma_ratio/t{t}"] = float(spec[1]) if spec.numel() > 1 else float("nan")
        out[f"effective_rank/t{t}"] = effective_rank(spec)

    return out
