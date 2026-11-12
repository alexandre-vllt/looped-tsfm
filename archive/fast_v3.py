"""
Replace reverso_torch's pure-Python delta rule with the paper's own kernel.

THE PROBLEM
-----------
The repo ships two models. reverso/model.py is what the paper used:

    from flashfftconv import FlashFFTConv
    from fla.layers import DeltaNet

reverso_torch/model.py is a fallback -- its own docstring says "uses only pure
PyTorch operations. No FlashFFTConv or flash-linear-attention (fla) required."
We chose it on izar, where FlashFFTConv was absent and fla's chunked kernels
could not compile for sm_70. That choice was never revisited on nala.

The cost is in TorchDeltaNet._delta_rule_recurrent:

    for t in range(L):          # L = 2048
        kv = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        ...

~12 000 tiny CUDA launches per attention block per forward, each ~5-10 us of
pure overhead, and a backward pass through a 2048-deep autograd graph. Measured
on nala: a full nano step at batch 64 takes 2.809 s, while the SAME DeltaNet
layer under fla does 16.98 M points/s -- the model is 365x slower than one of
its own four blocks. At batch 512 that is ~625 hours for 100 k steps, against
~11-22 h with fla. The experiment is impossible without this.

WHAT THIS DOES
--------------
Builds the model normally, then swaps each AttentionBlock for one wrapping
fla.layers.DeltaNet -- the paper's exact kernel. reverso_torch/model.py is NOT
edited, so it stays available as the reference the bit-identity test compares
against.

The weights transfer because TorchDeltaNet was written to be compatible:
"produces identical state-dict keys as fla.layers.DeltaNet so that pre-trained
checkpoints load directly". The swap loads strictly and reports any mismatch
rather than silently dropping tensors.

WHAT IS NOT CHANGED
-------------------
The long convolution stays on torch.fft. FlashFFTConv cannot be built here
(system nvcc 11.7 against torch's CUDA 12.4), but it is an OPTIMISED KERNEL for
the same operation, not a different one -- both compute
irfft(rfft(x) * rfft(k)). CNNBlock already does that in one cuFFT call, so it
was never the bottleneck. And for training from scratch the choice is
self-consistent: the same kernel runs in training and evaluation, so any
difference cancels. It would only matter when comparing the released checkpoint
to the published number.

MODE
----
Measured on nala, forward+backward, bf16 (bench_mode.sh):

    d_model  batch   fused_recurrent   chunk    winner
         32     64        16.98 M     21.64 M   chunk 1.27x
         32    256        46.23 M     60.60 M   chunk 1.31x
         32    512        66.51 M     63.09 M   fused_recurrent 1.05x
         64    512        34.38 M     43.81 M   chunk 1.27x

At nano's config (d_model 32, batch 512) the two are within 5 % -- a tie, not
the "large speedup" task.md section 0 predicted. The reason is shape: 4 heads of
8 dims cannot fill tensor cores built for 64x64 tiles, and at batch 512
fused_recurrent already has 2048 independent recurrences to saturate the GPU.
Default "chunk"; "fused_recurrent" is equally defensible for nano.

    python -m reverso_torch.fast --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import torch
from torch import nn

from reverso_torch.model import AttentionBlock, Model


def _fla_deltanet(d_model: int, expand_v: float, mode: str):
    """fla.layers.DeltaNet with AttentionBlock's exact constructor arguments."""
    from fla.layers import DeltaNet
    return DeltaNet(
        mode=mode, d_model=d_model, expand_k=1.0, expand_v=expand_v,
        num_heads=4, use_beta=True, use_gate=False, use_short_conv=True,
        conv_size=4, allow_neg_eigval=False, qk_activation='silu',
        qk_norm='l2', layer_idx=0,
    )


class FastAttentionBlock(nn.Module):
    """AttentionBlock with fla's fused DeltaNet in place of TorchDeltaNet.

    forward() is copied line for line from reverso_torch.model.AttentionBlock,
    including the state-weaving branch, so the only difference is which kernel
    computes the delta rule.
    """

    def __init__(self, d_model, expand_v, state_weaving=False,
                 is_intermediate=False, mode: str = "chunk"):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.mode = mode
        self.attention = _fla_deltanet(d_model, expand_v, mode)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        residual = x_t
        if self.state_weaving and self.is_intermediate:
            x_t = x_t.clone()
            x_t[:, 0:1, :] = x_t[:, 0:1, :] + x_t[:, -1:, :]
        attn_out = self.attention(hidden_states=x_t, attention_mask=None)
        out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        out = self.norm(out)
        out = out + residual
        out = out.transpose(1, 2)
        return out


def swap_to_fla(model: nn.Module, mode: str = "chunk",
                verbose: bool = True) -> nn.Module:
    """Replace every AttentionBlock in model.layers with a FastAttentionBlock.

    In place, and returns the model. Loading is STRICT: if fla's parameter
    names ever diverge from TorchDeltaNet's, this raises instead of quietly
    leaving the new block randomly initialised -- which would train, converge,
    and be wrong.
    """
    n = 0
    layers = list(model.layers)
    for i, m in enumerate(layers):
        if not isinstance(m, AttentionBlock):
            continue
        d_model = m.attention.hidden_size
        expand_v = m.attention.value_dim / d_model
        new = FastAttentionBlock(d_model, expand_v, m.state_weaving,
                                 m.is_intermediate, mode)
        src = m.state_dict()
        try:
            new.load_state_dict(src, strict=True)
        except RuntimeError as e:
            missing = set(new.state_dict()) - set(src)
            extra = set(src) - set(new.state_dict())
            raise RuntimeError(
                f"cannot transfer AttentionBlock {i} to fla.\n"
                f"  only in fla:           {sorted(missing)}\n"
                f"  only in TorchDeltaNet: {sorted(extra)}\n"
                f"  torch error: {e}\n"
                "TorchDeltaNet is documented as sharing fla's state-dict keys; "
                "if that no longer holds, map the names explicitly rather than "
                "loading with strict=False."
            ) from None
        new.to(next(m.parameters()).device)
        layers[i] = new
        n += 1
    model.layers = nn.Sequential(*layers)
    if verbose:
        print(f"  swapped {n} AttentionBlock -> fla.DeltaNet(mode={mode!r})")
    return model


def build(configs, use_fla: bool = True, mode: str = "chunk",
          looped: bool = False, verbose: bool = True) -> nn.Module:
    """Model (or LoopedModel) with the fast delta rule."""
    if looped:
        from reverso_torch.looped import LoopedModel
        m = LoopedModel(configs)
    else:
        m = Model(configs)
    if use_fla:
        m = swap_to_fla(m, mode, verbose)
    return m


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def self_check(ckpt_dir: str, mode: str, device: str = "cuda") -> int:
    """The swap must preserve the function, not merely run.

    If the state-dict transfer silently dropped tensors the model would still
    produce output -- from a random projection. This loads the released
    checkpoint into both paths and compares.
    """
    import numpy as np
    from types import SimpleNamespace

    cfg = SimpleNamespace(**json.load(open(f"{ckpt_dir}/args.json")))
    raw = torch.load(f"{ckpt_dir}/checkpoint.pth", map_location="cpu",
                     weights_only=False)
    sd = raw
    if isinstance(raw, dict):
        for k in ("model_state_dict", "state_dict", "model", "ema", "ema_state_dict"):
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]; break
    sd = {k.removeprefix("module."): v for k, v in sd.items()
          if "flashfftconv." not in k}

    ok, bad = [], []
    def chk(name, cond, detail=""):
        print(f"{'  ok   ' if cond else '  FAIL '} {name}"
              + (f"   {detail}" if detail else ""))
        (ok if cond else bad).append(name)

    print(f"nano: d_model {cfg.d_model}, seq {cfg.seq_len}, mode {mode!r}\n")

    ref = Model(cfg); ref.load_state_dict(sd, strict=True); ref.eval().to(device)
    fast = Model(cfg); fast.load_state_dict(sd, strict=True)
    fast = swap_to_fla(fast, mode).eval().to(device)

    chk("1. parameter count unchanged by the swap",
        sum(p.numel() for p in ref.parameters()) ==
        sum(p.numel() for p in fast.parameters()),
        f"{sum(p.numel() for p in ref.parameters()):,}")

    torch.manual_seed(0)
    x = torch.randn(4, cfg.seq_len, 1, device=device)

    def run(m, bf16):
        with torch.no_grad():
            if bf16:
                with torch.autocast(device, dtype=torch.bfloat16):
                    return m(x).float()
            return m(x).float()

    def cmp(a, b):
        c = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        return c, ((a - b).norm() / a.norm()).item()

    # COMPARE AT MATCHED PRECISION. Comparing a float32 reference against a
    # bf16 fast path measures bf16, not the kernel. The delta rule accumulates
    # a state over 2048 steps; bf16's 8-bit mantissa gives ~0.4 % per step,
    # which random-walks to sqrt(2048) * 0.4 % ~ 18 %. So a large float32-vs-bf16
    # gap is expected and says nothing about whether the swap is correct.
    ref_f32 = run(ref, False)
    ref_b16 = run(ref, True)
    fast_b16 = run(fast, True)

    c_bf, r_bf = cmp(ref_b16, fast_b16)      # the one that matters: training precision
    c_q, r_q = cmp(ref_f32, ref_b16)         # bf16 alone, same kernel both sides

    # The pure-kernel comparison needs float32 on both sides, but fla's chunked
    # kernel refuses it outright:
    #   AssertionError: ChunkDeltaRuleFunction does not support float32.
    # fused_recurrent has no such restriction, so a second fla model in that
    # mode gives the float32-vs-float32 number. Same recurrence, so it is still
    # a valid measure of "is the weight transfer right".
    r_f3, c_f3, kern_note = None, None, ""
    try:
        alt = Model(cfg); alt.load_state_dict(sd, strict=True)
        alt = swap_to_fla(alt, "fused_recurrent", verbose=False).eval().to(device)
        c_f3, r_f3 = cmp(ref_f32, run(alt, False))
        kern_note = "fused_recurrent"
        del alt
        torch.cuda.empty_cache()
    except Exception as e:
        kern_note = f"unavailable ({type(e).__name__}: {str(e)[:60]})"

    print(f"       reference float32 vs bf16 (SAME kernel, bf16 cost alone): "
          f"corr {c_q:.6f}, rel {r_q:.2e}")
    if r_f3 is None:
        print(f"       torch vs fla in float32: {kern_note}")
    else:
        print(f"       torch vs fla, both float32 via {kern_note} "
              f"(pure kernel):   corr {c_f3:.6f}, rel {r_f3:.2e}")
    print(f"       torch vs fla, both bf16 (training precision):            "
          f"corr {c_bf:.6f}, rel {r_bf:.2e}")

    # train.py runs under bf16 autocast, so bf16-vs-bf16 is the comparison that
    # describes what training will actually see.
    chk("2. same function as the reference at training precision (bf16)",
        c_bf > 0.999, f"corr {c_bf:.6f}, relative L2 {r_bf:.2e}")
    if r_f3 is None:
        print("  skip   2b. kernel-vs-bf16 comparison: no float32 fla path")
    else:
        chk("2b. the kernel difference is far below the bf16 cost",
            r_f3 <= max(r_q, 1e-3) * 1.5,
            f"kernel {r_f3:.2e} vs bf16 {r_q:.2e} -- if the kernel term "
            f"dominated, the weight transfer would be suspect")
    chk("3. output is finite", bool(torch.isfinite(fast_b16).all()))

    # timing: the whole point
    import time
    def step_time(m, B=16, loops=4):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        xx = torch.randn(B, cfg.seq_len, 1, device=device)
        yy = torch.randn(B, cfg.output_token_len, 1, device=device)
        def one():
            with torch.autocast(device, dtype=torch.bfloat16):
                out = m(xx)
            (out.float() - yy).abs().mean().backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        one(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(loops): one()
        torch.cuda.synchronize()
        return (time.time() - t0) / loops

    ref.train(); fast.train()
    t_ref = step_time(ref)
    t_fast = step_time(fast)
    chk("4. the swap is faster", t_fast < t_ref,
        f"{t_ref:.3f}s -> {t_fast:.3f}s  ({t_ref / t_fast:.0f}x) at batch 16")

    print()
    print(f"FAILED: {bad}" if bad else f"all {len(ok)} checks passed")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--ckpt", default="checkpoints/reverso_nano")
    p.add_argument("--mode", default="chunk",
                   choices=["chunk", "fused_recurrent"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not a.self_check:
        p.error("pass --self-check")
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("no CUDA available")
    sys.exit(self_check(a.ckpt, a.mode, a.device))


if __name__ == "__main__":
    main()
