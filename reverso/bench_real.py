"""What does k=4 actually cost, on the real training path, at the real batch size?

WHY THIS EXISTS
The k=1 and k=4 smoke runs both reported 2.31 M pts/s at batch 64 -- four times
the loop iterations for no measurable wall-clock cost. bench_mode.sh had measured
4.02x, but it ran WITHOUT swap_to_fla, where the python delta rule over 2048
timesteps dominates and looping it 4x costs 4x. With fla the DeltaNet is nearly
free, and at batch 64 a 190 K-parameter model is latency-bound: 0.13 M points per
step at 2.31 M pts/s is ~55 ms/step, which is kernel launches and python, not
arithmetic. So batch 64 cannot answer the question.

THREE THINGS THIS DECIDES

1. THE REAL COST RATIO at the paper's batch 512. If k=4 is still ~1x, "4x
   compute" has to be argued in FLOPs, not seconds -- and the walltime budget
   for the whole (k, beta) sweep collapses, which is good news.

2. WHETHER R1 FITS AT ALL. bench_mode.sh's comment says "512 would OOM at k=4 on
   a shared GPU" -- never verified, and it used batch 64 for that reason. k=4
   with multi-exit and full BPTT holds 4 sets of activations over a 2048-step
   sequence. If it OOMs at 512 on a 40 GB A100, the options are gradient
   checkpointing, a smaller batch with accumulation, or exit-subset supervision
   -- and that is a decision to take BEFORE launching, not after an 8-hour
   crash.

3. chunk VS fused_recurrent. train_loop defaults to chunk. bench_mode.sh found
   chunk was 0.73x at d_model 32 / batch 256 -- fused_recurrent 37 % FASTER --
   because nano's head_k_dim is 8 while chunked kernels are built for 64x64
   tensor-core tiles. If that holds at 512, the default is simply wrong.

Uses the SAME construction as train_loop.py (LoopedModel -> swap_to_fla -> the
scale-free loss -> AdamW), so this measures the training path rather than
something adjacent to it.

    python bench_real.py                      # the full grid
    python bench_real.py --batches 512        # one batch size
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch

from reverso_torch.fast import swap_to_fla
from reverso_torch.looped import LoopedModel, multi_exit_loss

DEGEN = 1e-5


def scale_free_loss(pred, y, x):
    """Identical to train_loop.py's guard-v2 loss."""
    xmin = x.min(1, keepdim=True)[0]
    rng = x.max(1, keepdim=True)[0] - xmin
    degen = rng <= DEGEN
    dev = (y - xmin).abs().amax(dim=1, keepdim=True)
    drop = degen & (dev > DEGEN)
    m = (~torch.isnan(y)) & (~drop).expand_as(y)
    if not bool(m.any()):
        return (pred * 0.0).sum()
    return ((pred - y).abs() / rng.clamp(min=DEGEN))[m].mean()


def run(cfg, k, beta, mode, B, iters=6, warmup=3):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    c = SimpleNamespace(**{**vars(cfg), "loop_k": k, "loop_beta": beta})
    m = swap_to_fla(LoopedModel(c), mode).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.1)
    L, H = cfg.seq_len, cfg.output_token_len
    x = torch.randn(B, L, 1, device="cuda")
    y = torch.randn(B, H, 1, device="cuda")
    want = k > 1

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, return_all_exits=want)
        if want:
            obj, _ = multi_exit_loss([o.float() for o in out], y, "uniform",
                                     loss_fn=lambda p, t: scale_free_loss(p, t, x))
        else:
            obj = scale_free_loss(out.float(), y, x)
        obj.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 2**30
    del m, opt, x, y
    torch.cuda.empty_cache()
    return dt, mem


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/reverso_nano")
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256, 512])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    p.add_argument("--modes", nargs="+", default=["chunk", "fused_recurrent"])
    p.add_argument("--iters", type=int, default=6)
    a = p.parse_args()

    cfg = SimpleNamespace(**json.load(open(f"{a.ckpt}/args.json")))
    L = cfg.seq_len
    print(f"{torch.cuda.get_device_properties(0).name}   "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB   "
          f"torch {torch.__version__}")
    print(f"nano d_model {cfg.d_model}, seq {L} -> {cfg.output_token_len}, "
          f"4 heads -> head_k_dim {cfg.d_model // 4}\n")

    print(f"  {'mode':<17}{'batch':>6}{'k':>3}{'s/step':>9}{'M pts/s':>10}"
          f"{'peak GB':>9}{'k4/k1':>8}")
    results = {}
    for mode in a.modes:
        for B in a.batches:
            base = None
            for k in a.ks:
                beta = 1.0 if k == 1 else 0.25
                try:
                    dt, mem = run(cfg, k, beta, mode, B, a.iters)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {mode:<17}{B:>6}{k:>3}{'OOM':>9}{'':>10}{'':>9}")
                    results[(mode, B, k)] = None
                    continue
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"  {mode:<17}{B:>6}{k:>3}  {type(e).__name__}: "
                          f"{str(e)[:40]}")
                    results[(mode, B, k)] = None
                    continue
                if base is None:
                    base = dt
                results[(mode, B, k)] = (dt, mem)
                print(f"  {mode:<17}{B:>6}{k:>3}{dt:>9.3f}{B*L/dt/1e6:>10.2f}"
                      f"{mem:>9.2f}{dt/base:>7.2f}x")
            print()

    print("=" * 70)
    print(" WHAT TO CONCLUDE")
    print("=" * 70)
    r = results
    def got(mode, B, k):
        v = r.get((mode, B, k))
        return v[0] if v else None

    for mode in a.modes:
        for B in a.batches:
            t1, t4 = got(mode, B, 1), got(mode, B, 4)
            if t1 and t4:
                print(f"  {mode} @ batch {B}: k=4 costs {t4/t1:.2f}x k=1")
            elif t1 and not t4:
                print(f"  {mode} @ batch {B}: k=4 DID NOT RUN -- see above")

    print("""
  COST RATIO near 1.0x  -> the loop is free in wall-clock at this scale; the
      model is latency-bound, not compute-bound. "4x compute" must then be
      stated in FLOPs. The sweep's walltime budget is set by k=1.
  COST RATIO near 4.0x  -> compute-bound; budget the sweep at 4x for the k=4
      cells and plan the walltime ceiling (open question Q4) accordingly.

  OOM at k=4 / batch 512 -> decide BEFORE launching: gradient checkpointing on
      the looped core, or batch 128 with 4x accumulation (same effective batch,
      same data order -- the stream is ordinal-indexed, so accumulation does not
      change which samples are seen), or supervise a subset of exits.

  If fused_recurrent beats chunk at batch 512, train_loop's --mode default is
  wrong for nano and should be changed before the real runs.
""")


if __name__ == "__main__":
    main()
