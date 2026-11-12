"""Pre-training loop for Reverso — paper-faithful where the paper is explicit.

Replaces the earlier train.py. What changed and why:

  * loss is MAE on the UNNORMALIZED output with NaN targets masked out
    (paper 3.1, "Training objective"), not MSE in normalized space
  * AdamW settings are the paper's: lr 5e-4, betas (0.9, 0.999), eps 1e-8,
    weight decay 0.1, batch size 512 (-> ~1M time points per step at L=2048)
  * data comes from pre-generated shards (--data) when available, otherwise
    from the on-the-fly generators in synthetic.py
  * naive baselines are printed once so a falling loss can be judged
  * the dataset RNG state is checkpointed, so resume is exact

Still unknown and therefore ours to choose: the WSD warmup/stable/decay
fractions, the total number of steps, the real:synthetic ratio, whether an
EMA was used, and the gradient clipping value.

    python train.py --data /scratch/izar/$USER/synth --steps 2000
"""
import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from reverso_fast.model import Model


# --------------------------------------------------------------------------
def wsd_lr(step, total, warmup=0.05, stable=0.75):
    """Warmup-Stable-Decay. Fractions are UNSPECIFIED in the paper."""
    w, s = int(total * warmup), int(total * (warmup + stable))
    if step < w:
        return (step + 1) / max(w, 1)
    if step < s:
        return 1.0
    return max(0.0, 1.0 - (step - s) / max(total - s, 1))


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="reverso_small")
    p.add_argument("--data", default="",
                   help="directory of shard_*.npy; empty = generate on the fly")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=512)   # paper
    p.add_argument("--lr", type=float, default=5e-4)        # paper
    p.add_argument("--wd", type=float, default=0.1)         # paper
    p.add_argument("--beta1", type=float, default=0.9)      # paper
    p.add_argument("--beta2", type=float, default=0.999)    # paper
    p.add_argument("--eps", type=float, default=1e-8)       # paper
    p.add_argument("--clip", type=float, default=1.0)       # UNSPECIFIED
    p.add_argument("--warmup", type=float, default=0.05)    # UNSPECIFIED
    p.add_argument("--stable", type=float, default=0.75)    # UNSPECIFIED
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--out", default="runs/small")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loop-k", type=int, default=1)
    p.add_argument("--loop-inject", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    cfg = SimpleNamespace(**json.load(open(f"checkpoints/{a.size}/args.json")))
    cfg.loop_k = a.loop_k
    cfg.loop_inject = a.loop_inject
    model = Model(cfg).to(dev)
    L, H = cfg.seq_len, cfg.output_token_len
    print(f"{a.size}: {sum(q.numel() for q in model.parameters()):,} params | "
          f"seq {L} -> {H} | loop_k {a.loop_k} | batch {a.batch_size} "
          f"({a.batch_size * L / 1e6:.2f}M points/step)", flush=True)

    # ---- data ------------------------------------------------------------
    if a.data:
        from synth_dataset import SynthShards
        ds = SynthShards(a.data, seq_len=L, horizon=H, seed=a.seed)
        print(f"shards: {len(ds):,} series of {ds.series_len} pts "
              f"from {a.data}", flush=True)

        def next_batch():
            return ds.batch(a.batch_size, device=dev)
    else:
        import synthetic as S
        ds = None
        print("no --data: generating on the fly (slower)", flush=True)
        gen = np.random.default_rng(a.seed)

        def next_batch():
            s = S.mixed_batch(a.batch_size, L + H, dev, gen)
            return s[:, :L, None], s[:, L:, None]

    # ---- optimiser -------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            betas=(a.beta1, a.beta2), eps=a.eps,
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: wsd_lr(s, a.steps, a.warmup, a.stable))

    start = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); set_rng_state(ck["rng"])
        if ds is not None and ck.get("ds") is not None:
            ds.load_state(ck["ds"])
        start = ck["step"]
        print(f"resumed from {a.resume} at step {start}", flush=True)

    zx = torch.zeros(a.batch_size, L, 1, device=dev)
    zy = torch.zeros(a.batch_size, H, 1, device=dev)

    # paper 3.1: MAE on the unnormalized output, NaN targets masked out
    def loss_fn(pred, y):
        m = ~torch.isnan(y)
        return (pred - y).abs()[m].mean()

    # ---- fixed validation batch + naive baselines ------------------------
    vx, vy = next_batch()
    vx, vy = vx.detach(), vy.detach()
    with torch.no_grad():
        last = vx[:, -1:, :].expand(-1, H, -1)
        mean = vx.mean(1, keepdim=True).expand(-1, H, -1)
        drift = vx[:, -1:, :] + (vx[:, -1:, :] - vx[:, -2:-1, :]) * \
            torch.arange(1, H + 1, device=dev).view(1, H, 1)
        print(f"baselines   last {loss_fn(last, vy):.5f}   "
              f"mean {loss_fn(mean, vy):.5f}   "
              f"drift {loss_fn(drift, vy):.5f}", flush=True)

    log = open(os.path.join(a.out, "log.csv"), "a")
    if start == 0:
        log.write("step,loss,val,lr,tok_per_s\n")

    # ---- loop ------------------------------------------------------------
    model.train(); t0 = time.time(); run = 0.0
    for step in range(start, a.steps):
        x, y = next_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, zx, zy)
        loss = loss_fn(pred.float(), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run += loss.item()

        if (step + 1) % a.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = a.log_every * a.batch_size * L / dt
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                val = loss_fn(model(vx, zx, zy).float(), vy).item()
            model.train()
            tr, lr = run / a.log_every, sched.get_last_lr()[0]
            print(f"step {step + 1:6d}  loss {tr:.5f}  val {val:.5f}  "
                  f"lr {lr:.2e}  {tps / 1e3:.0f}k tok/s", flush=True)
            log.write(f"{step + 1},{tr:.6f},{val:.6f},{lr:.3e},{tps:.0f}\n")
            log.flush()
            run = 0.0; t0 = time.time()

        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng_state(),
                        "ds": ds.state() if ds is not None else None,
                        "step": step + 1, "cfg": vars(cfg), "args": vars(a)},
                       os.path.join(a.out, "last.pth"))


if __name__ == "__main__":
    main()
