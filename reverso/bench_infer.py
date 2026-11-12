"""Inference cost of R0 and R1 — forward only, no backward, no optimiser.

Everything measured so far has been TRAINING cost (forward + backward + AdamW):
0.094 s/step for R0 and 0.380 s/step for R1 at batch 512, a 4.03x ratio. That is
not what deployment pays. This measures the forward pass alone, which is the
number that matters for "what does looping cost at serving time".

Two things make the answer less simple than one number:

  * The ratio need not be 4.03x. The backward pass through a 4-iteration loop
    carries 4 sets of activations, while the forward pass only re-runs the
    block -- so the inference ratio should be closer to 4x on pure compute but
    can be lower if the embedding, head and cross-attention decoder (which run
    ONCE regardless of k) are a large share of the total at this model size.

  * GIFT-Eval does not call the model once per forecast. ReversoPredictor rolls
    out autoregressively, ceil(prediction_length / output_token_len) forward
    passes per window, so a 480-step horizon is 10 calls. The per-call number
    below multiplies by that.

Batch sizes: 512 (the training batch, for comparability with bench_real.py) and
256 (what eval_gift.py actually uses).

    python bench_infer.py
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch

from reverso_torch.fast import swap_to_fla
from reverso_torch.looped import LoopedModel


def run(cfg, k, beta, B, mode, iters=10, warmup=4, bf16=True):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    c = SimpleNamespace(**{**vars(cfg), "loop_k": k, "loop_beta": beta})
    m = swap_to_fla(LoopedModel(c), mode).cuda().eval()
    x = torch.randn(B, cfg.seq_len, 1, device="cuda")

    def one():
        with torch.no_grad():
            if bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(x)
            else:
                m(x)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 2**30
    del m, x
    torch.cuda.empty_cache()
    return dt, mem


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/reverso_nano")
    p.add_argument("--batches", type=int, nargs="+", default=[256, 512])
    p.add_argument("--mode", default="chunk")
    a = p.parse_args()

    cfg = SimpleNamespace(**json.load(open(f"{a.ckpt}/args.json")))
    print(f"{torch.cuda.get_device_properties(0).name}   torch {torch.__version__}")
    print(f"nano d_model {cfg.d_model}, seq {cfg.seq_len} -> {cfg.output_token_len}, "
          f"bf16 autocast, forward only\n")

    print(f"  {'batch':>6}{'k':>3}{'s/batch':>10}{'ms/serie':>11}"
          f"{'M pts/s':>10}{'peak GB':>9}{'k4/k1':>8}")
    for B in a.batches:
        base = None
        for k, beta in ((1, 1.0), (4, 0.25)):
            dt, mem = run(cfg, k, beta, B, a.mode)
            if base is None:
                base = dt
            print(f"  {B:>6}{k:>3}{dt:>10.4f}{dt/B*1000:>11.3f}"
                  f"{B*cfg.seq_len/dt/1e6:>10.2f}{mem:>9.2f}{dt/base:>7.2f}x")
        print()

    print("=" * 70)
    print(""" Pour mettre en face des coûts d'ENTRAÎNEMENT déjà mesurés
 (forward + backward + AdamW, batch 512) :   R0 0.094 s   R1 0.380 s   4.03x

 Et pour GIFT-Eval : le prédicteur déroule ceil(horizon / 48) passes avant
 par fenêtre, donc une prévision à 480 pas coûte 10x la ligne ci-dessus.
""")


if __name__ == "__main__":
    main()
