"""
Pre-training for the R0 (k=1) vs R1 (k=4) comparison.

    # 200-step dry run, both arms, before anything long
    python train_loop.py --smoke --loop-k 1
    python train_loop.py --smoke --loop-k 4 --loop-beta 0.25

    # the real thing
    python train_loop.py --out runs/R0 --loop-k 1 --loop-beta 1.0
    python train_loop.py --out runs/R1-uniform --loop-k 4 --loop-beta 0.25 --exit-loss uniform

WHAT THIS ENFORCES, AND WHY
---------------------------
1. THE TWO ARMS SEE BYTE-IDENTICAL DATA, IN THE SAME ORDER.
   Both corpora are pinned by sha256 and asserted at startup; the sampler is a
   counter-based PRNG so sample n is a pure function of (n, seed), independent
   of num_workers. --verify-stream additionally hashes the first N batches and
   prints the number both runs must report. Without this the measured R1-R0 gap
   would contain sampling noise, and nobody could tell how much.

2. EMA, AND EVALUATION ON THE EMA WEIGHTS.
   Xinghong Fu (Reverso's first author), 3 Oct 2026: "Yes we did use ema". It is
   not in the paper. Training with an EMA and evaluating raw weights -- or the
   reverse -- compares two different models, and the difference is the same
   order as the effect we are trying to measure. Decay 0.9999 is OURS; the real
   value is still an open question to him.

3. DIAGNOSTICS IN FLOAT32, NOT bf16.
   Measured on nala: bf16 costs 9.3 % relative error on nano's output, because
   the delta rule accumulates over 2048 steps in an 8-bit mantissa
   (sqrt(2048) * 2^-8 ~ 18 %). update_magnitude (~1-2) and convergence (~0.4-4)
   survive that. CKA does NOT -- we need to distinguish 0.9981 from 0.9999, and
   that is not credible on inputs carrying 9 % noise. So the diagnostic probe
   runs outside autocast. fla's CHUNKED kernel refuses float32 outright
   ("ChunkDeltaRuleFunction does not support float32"), so the probe
   temporarily switches the DeltaNet to fused_recurrent.

4. THE PLATEAU CRITERION IS FIXED IN ADVANCE (task.md 7.1).
   EMA of the training loss, 500-step half-life; plateau when
   (E(s-5000) - E(s)) / E(s) < 0.005 sustained for 5000 steps. Logged every 100
   steps so the decision is visible rather than eyeballed. On R1 it is taken on
   L^(k), never on the averaged multi-exit objective, which is not comparable
   to R0's single loss.

MEASURED CONTEXT (nala, 7 Oct 2026)
-----------------------------------
  fla swap          2.78 s/step -> 0.014 s at batch 16, 204x. Verified exact:
                    fla vs TorchDeltaNet agree to 6.2e-6 in float32.
  R1/R0 cost        4.02x at k=4 (task.md 1 assumed "~4x" -- it is almost
                    exactly 4x, because at d_model 32 the loop body dominates
                    and the embedding/head/decoder outside it are noise)
  multi-exit        +2.7 % over last-exit only, so R1-uniform and R1-last cost
                    the same and are fairly comparable
  memory            0.88 GB (k=1) / 3.24 GB (k=4) at batch 64 under the SLOW
                    path; re-measure under fla before trusting batch 512
  DeltaNet mode     at d_model 32 batch 512, chunk and fused_recurrent are
                    within 5 % -- the choice barely matters for nano
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from reverso_torch import augment as augment_mod
from reverso_torch.data import make_loader, stream_hash
from reverso_torch.ema import EMA
from reverso_torch.fast import swap_to_fla
from reverso_torch.looped import LoopedModel, loop_diagnostics, multi_exit_loss


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def wsd_lr(step: int, total: int, warmup: float, stable: float,
           decay_from: int = 0) -> float:
    """Warmup-Stable-Decay.

    Fractions are UNSPECIFIED in the paper. The author's reply ("W/S contributed
    about 20-40 %") is ambiguous between three readings -- see task.md 7.2 -- so
    0.05 / 0.75 / 0.20 stays ours.

    decay_from > 0 forks: warm up, hold, then decay over the 20 % of steps
    following decay_from. That is what makes "train until the loss flattens"
    workable without committing to a step count up front -- the stable phase can
    run open-ended and a decay is forked from any checkpoint.
    """
    if decay_from:
        w = int(decay_from * warmup / (warmup + stable)) or 1
        if step < w:
            return (step + 1) / w
        if step < decay_from:
            return 1.0
        tail = max(int(0.2 * decay_from), 1)
        return max(0.0, 1.0 - (step - decay_from) / tail)

    w, s = int(total * warmup), int(total * (warmup + stable))
    if step < w:
        return (step + 1) / max(w, 1)
    if step < s:
        return 1.0
    return max(0.0, 1.0 - (step - s) / max(total - s, 1))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def fused_recurrent_mode(model):
    """Temporarily put every fla DeltaNet into fused_recurrent.

    fla's chunked kernel asserts against float32, so the float32 diagnostic
    probe cannot use it. fused_recurrent computes the same recurrence and has no
    such restriction. Falls back silently if the attribute is not there.
    """
    saved = []
    for m in model.modules():
        if hasattr(m, "mode") and getattr(m, "mode", None) == "chunk":
            saved.append((m, m.mode))
            m.mode = "fused_recurrent"
    try:
        yield bool(saved)
    finally:
        for m, old in saved:
            m.mode = old


@torch.no_grad()
def probe_diagnostics(model, x, k, beta):
    """The four task.md section 12 quantities plus ||h^(t)||, in FLOAT32.

    Returns {} if the float32 path is unavailable, rather than silently
    reporting bf16 numbers that would make CKA meaningless.
    """
    model.eval()
    try:
        with fused_recurrent_mode(model) as switched:
            _, states = model(x, loop_k=k, loop_beta=beta, return_states=True)
            d = loop_diagnostics(states)
            # ||h^(t)||: the beta-probe risk. On the released checkpoint h grew
            # x71 at beta=1 over 4 iterations because every block is
            # x + LayerNorm(branch(x)) -- the LayerNorm bounds the BRANCH, not
            # the residual stream, so nothing bounds h itself.
            for t, h in enumerate(states["h"]):
                d[f"h_norm/t{t}"] = float(h.norm())
            d["_fp32"] = bool(switched)
        return d
    except Exception as e:
        print(f"  !! diagnostics unavailable: {type(e).__name__}: {str(e)[:80]}")
        return {}
    finally:
        model.train()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # what to train
    p.add_argument("--size", default="reverso_nano")
    p.add_argument("--loop-k", type=int, default=1)
    p.add_argument("--loop-beta", type=float, default=1.0)
    p.add_argument("--exit-loss", default="uniform", choices=["uniform", "last"])
    p.add_argument("--loss-space", default="normalised",
                   choices=["normalised", "raw"],
                   help="raw = paper 2.1 verbatim (MAE on the unnormalised "
                        "output). normalised = the same MAE divided by each "
                        "window's context range. DEVIATION, see done.md 9: "
                        "measured 7 Oct, the raw objective puts 68.2%% of the "
                        "gradient in 1%% of windows on our 50/50 mixture.")
    p.add_argument("--augment", action="store_true",
                   help="Algorithm 4 augmentation (paper appendix D.2). "
                        "Changes the sample stream, so the hash is new.")
    p.add_argument("--augment-no-mixup", action="store_true",
                   help="steps 1-5 per sample, no batch mixup")
    p.add_argument("--augment-cfg", default="",
                   help='JSON overrides of augment.CFG, e.g. \'{"p_censor": 0.4}\'')
    p.add_argument("--mode", default="chunk", choices=["chunk", "fused_recurrent"])
    p.add_argument("--no-fla", action="store_true",
                   help="use the pure-python delta rule (204x slower; debugging only)")
    # data
    # No "." fallback. An unset WORK used to resolve to ./synth and die with
    # "no .npy files in ./synth", which reads like a corpus problem and is an
    # environment problem -- it killed five sweep arms on 12 Oct (done.md 10).
    _work = os.environ.get("WORK")
    p.add_argument("--data", default=(_work + "/synth") if _work else None)
    p.add_argument("--real", default=(_work + "/real") if _work else None)
    p.add_argument("--p-real", type=float, default=0.5)
    p.add_argument("--real-alpha", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--verify", default="full", choices=["full", "quick", "none"])
    p.add_argument("--verify-stream", type=int, default=0,
                   help="hash the first N batches at startup and print it")
    # optimiser -- paper values where the paper states one
    p.add_argument("--batch-size", type=int, default=512)   # paper
    p.add_argument("--lr", type=float, default=5e-4)        # paper
    p.add_argument("--wd", type=float, default=0.1)         # paper
    p.add_argument("--beta1", type=float, default=0.9)      # paper
    p.add_argument("--beta2", type=float, default=0.999)    # paper
    p.add_argument("--eps", type=float, default=1e-8)       # paper
    p.add_argument("--clip", type=float, default=1.0)       # author: "0.1-1"
    p.add_argument("--ema-decay", type=float, default=0.9999)   # OURS
    p.add_argument("--warmup", type=float, default=0.05)    # OURS
    p.add_argument("--stable", type=float, default=0.75)    # OURS
    # v2, 23 oct. resample = DeterministicMix, tirage avec remise, le defaut
    # historique : le flux de tout run anterieur est inchange au bit.
    # enumerated = EnumeratedMix (reverso_torch/epochs.py) : de vraies epochs,
    # couverture exacte par creneau alloue, offset de fenetre fonction de
    # (chunk, epoch). Le hash de flux SERA different -- c'est le but.
    p.add_argument("--sampler", default="resample",
                   choices=["resample", "enumerated"],
                   help="resample (defaut, avec remise) ou enumerated "
                        "(epochs reelles, v2)")
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=64,
                   help="sampler=enumerated : combien d'epochs sont "
                        "adressables. Ce n'est PAS une regle d'arret, juste "
                        "la taille du dataset ; --steps decide la duree.")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--decay-from", type=int, default=0,
                   help="fork a decay phase from this step (task.md 7.1)")
    p.add_argument("--plateau-window", type=int, default=5000)
    p.add_argument("--plateau-tol", type=float, default=0.005,
                   help="relative held-out gain over the window below which "
                        "the curve counts as flat")
    # bookkeeping
    p.add_argument("--out", default="runs/dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-size", type=int, default=1024,
                   help="held-out windows; the plateau test reads THIS curve")
    p.add_argument("--diag-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=2000,
                   help="overwrites last.pth, for resume")
    p.add_argument("--snapshot-every", type=int, default=10000,
                   help="ALSO keep step<N>.pth, never overwritten. The headline "
                        "comparison is R0 vs R1 AT THE SAME STEP COUNT; without "
                        "kept snapshots that is unanswerable after the fact "
                        "unless both arms happen to stop on the same step. nano "
                        "is 190 K params, so a snapshot is ~2.3 MB -- 15 of them "
                        "is 35 MB, nothing against the 60 GB floor.")
    p.add_argument("--resume", default="")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="looped-tsfm")
    p.add_argument("--smoke", action="store_true",
                   help="200 steps, small batch, prints baselines -- the dry run")
    a = p.parse_args()

    if a.decay_from:
        # The fork's decay runs over the 20 % of steps after decay_from. If
        # --steps stops before that tail finishes, the run ends at a nonzero LR
        # and the checkpoint is not an annealed model -- which is the only thing
        # the decay phase was for.
        need = a.decay_from + max(int(0.2 * a.decay_from), 1)
        if a.steps < need:
            print(f"  --decay-from {a.decay_from}: extending --steps "
                  f"{a.steps} -> {need} so the decay tail completes")
            a.steps = need

    if a.smoke:
        a.steps = 200
        a.batch_size = min(a.batch_size, 64)
        a.log_every = 20
        a.diag_every = 50
        a.ckpt_every = 10**9
        a.verify_stream = a.verify_stream or 5
        a.val_every = 50            # the default 500 never fires in 200 steps
        a.val_size = min(a.val_size, 256)
        a.out = a.out if a.out != "runs/dev" else f"runs/smoke-k{a.loop_k}"

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- resume: read the step BEFORE the loader is built -----------------
    # The data stream has to be advanced to where the checkpoint stopped. If it
    # is not, a resumed run replays from ordinal 0 and its sample order stops
    # matching an uninterrupted run of the same seed -- which would silently
    # void the R0/R1 "identical data in identical order" guarantee for whichever
    # arm happened to be interrupted.
    ck = None
    start = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        start = int(ck["step"])

    if not a.data:
        raise SystemExit(
            "WORK is unset and --data was not given, so there is no corpus to "
            "read. Either `source ~/env_home.sh` or pass --data /abs/path/synth. "
            "(A tmux session does not inherit your shell environment.)")

    aug = None
    if a.augment:
        aug = dict(augment_mod.CFG)
        if a.augment_cfg:
            aug.update(json.loads(a.augment_cfg))
        if a.augment_no_mixup:
            aug["p_mixup"] = 0.0

    # ---- data, and the contract ------------------------------------------
    print("=" * 70)
    print(" data")
    print("=" * 70)
    dl = make_loader(data_dir=a.data, real_dir=a.real, seq_len=2048, horizon=48,
                     batch_size=a.batch_size, seed=a.seed, p_real=a.p_real,
                     num_workers=a.workers, verify=a.verify,
                     real_alpha=a.real_alpha, start_step=start,
                     augment=aug,
                     sampler=a.sampler, max_epochs=a.max_epochs)
    if start:
        print(f"  stream advanced to step {start} "
              f"(ordinal {start * a.batch_size:,})")
    syn_hash = json.load(open(f"{a.data}/manifest.json"))["corpus_sha256"]
    real_hash = (json.load(open(f"{a.real}/manifest.json"))["corpus_sha256"]
                 if a.real else None)
    stream = None
    if a.verify_stream:
        stream = stream_hash(
            make_loader(data_dir=a.data, real_dir=a.real, seq_len=2048, horizon=48,
                        batch_size=a.batch_size, seed=a.seed, p_real=a.p_real,
                        num_workers=0, verify="none", real_alpha=a.real_alpha,
                        augment=aug),
            a.verify_stream)
        print(f"\n  stream_sha256 (first {a.verify_stream} batches) {stream}")
        print("  BOTH ARMS MUST REPORT THIS NUMBER.")
        print(f"  It is a function of (seed {a.seed}, p_real {a.p_real}, "
              f"alpha {a.real_alpha}, batch {a.batch_size}, both corpus "
              f"hashes) -- so it is comparable only between runs that share "
              f"all of those. A smoke hash at batch {a.batch_size} will NOT "
              f"equal the production hash at batch 512.")

    # ---- model -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(" model")
    print("=" * 70)
    cfg = SimpleNamespace(**json.load(open(f"checkpoints/{a.size}/args.json")))
    cfg.loop_k, cfg.loop_beta = a.loop_k, a.loop_beta
    model = LoopedModel(cfg)
    if not a.no_fla:
        model = swap_to_fla(model, a.mode)
    model = model.to(dev)
    L, H = cfg.seq_len, cfg.output_token_len
    n_par = sum(q.numel() for q in model.parameters())
    print(f"  {a.size}: {n_par:,} params | seq {L} -> {H} | "
          f"k={a.loop_k} beta={a.loop_beta} | batch {a.batch_size} "
          f"({a.batch_size * L / 1e6:.2f} M points/step)")

    ema = EMA(model, decay=a.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            betas=(a.beta1, a.beta2), eps=a.eps,
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: wsd_lr(s, a.steps, a.warmup, a.stable, a.decay_from))

    if ck is not None:
        if ck.get("syn_hash") != syn_hash:
            raise SystemExit("resume: the corpus changed since this checkpoint. "
                             "The run would not be comparable to its own start.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); ema.load_state_dict(ck["ema"])
        print(f"  resumed from {a.resume} at step {start}")
        del ck

    def loss_fn(pred, y):
        """Reverso 2.1: MAE on the UNNORMALISED output, NaN targets masked.

        Kept exactly as the paper states it. This is the metric the `raw` rows
        are reported in, and the training objective when --loss-space raw.
        """
        m = ~torch.isnan(y)
        return (pred - y).abs()[m].mean()

    # ---- the training objective ------------------------------------------
    # DEVIATION FROM PAPER 2.1, taken 7 Oct on measured evidence. scale_probe.py
    # on the real 50/50 mixture at batch 512:
    #
    #            top 1% of windows   top 10%   mean/median
    #   raw            68.2%           92.1%      36.8x
    #   scale-free      5.7%           33.2%       1.4x
    #
    # decode() ends with `out * stdev + means`, stdev = x_range, both detached,
    # so |y_hat*s + m - y| == s * |y_hat - (y-m)/s| pointwise: the paper's loss
    # is the scale-free loss weighted by each window's range. On a corpus whose
    # sources span orders of magnitude that concentrates the gradient in a
    # scale-selected sliver -- and it makes the deliberate p_real=0.5 mixture
    # (task.md 8.4) something other than 50/50 where it counts, since the
    # synthetic half is mostly small-range.
    #
    # Costs nothing at inference: the head emits normalised values either way
    # and decode rescales by detached statistics, so predictions stay in raw
    # units and GIFT-Eval is unaffected. What it costs is comparability of the
    # absolute loss number with the paper's published nano figure.
    #
    # GUARD -- v2, 9 October, after degen_probe.py.
    #
    # v1 dropped every window whose context range hit the 1e-5 clamp. That was
    # wrong. degen_probe.py over 43 sources x 3000 windows:
    #
    #     benign (flat -> flat)        1489
    #     pathological (flat -> jump)     5
    #
    # so v1 discarded ~1500 free correct predictions to exclude 5 bad ones, and
    # it did so unevenly: wind_farms_with_missing 24.2 %, residential_pv_power
    # 9.2 %, cmip6_1990 6.5 %, residential_load_power 5.5 %, era5_2010 2.9 %.
    # Those flat stretches are a PHYSICAL REGIME -- becalmed turbines, solar at
    # night, ice-covered climate cells, zero household load -- not corruption,
    # and they are exactly the mode the model should learn. Removing a quarter
    # of one source's windows also perturbs the per-source mixture that the
    # alpha = 0.5 decision (done.md 5.3.1) chose.
    #
    # The division was never the problem. encode() already clamps its own
    # x_range at 1e-5, so for a flat window pred = o_hat*1e-5 + c and
    #
    #     |o_hat*1e-5 + c - y| / 1e-5  =  |o_hat + (c-y)/1e-5|  ->  |o_hat|
    #
    # when y == c: bounded, and it trains the model to emit 0 on a zero input,
    # i.e. "flat stays flat". Only a DEVIATING future makes (c-y)/1e-5 explode.
    #
    # So the drop is keyed on the future deviating, not on the context being
    # flat. Expected drop rate falls from 0.6 % to roughly 5/1494 of that,
    # ~0.002 %. The counter stays, because the v1 rate was asserted rather
    # than measured once already.
    DEGEN = 1e-5
    dropped = [0, 0]          # [windows dropped, windows seen]

    def scale_free_loss(pred, y, x):
        xmin = x.min(1, keepdim=True)[0]
        rng = x.max(1, keepdim=True)[0] - xmin                   # (B,1,C)
        degen = rng <= DEGEN
        # How far the future leaves the (collapsed) context level.
        dev = (y - xmin).abs().amax(dim=1, keepdim=True)         # (B,1,C)
        drop = degen & (dev > DEGEN)        # flat -> jump: unpredictable
        dropped[0] += int(drop.sum()); dropped[1] += int(drop.numel())
        m = (~torch.isnan(y)) & (~drop).expand_as(y)
        if not bool(m.any()):
            return (pred * 0.0).sum()                            # keeps the graph
        return ((pred - y).abs() / rng.clamp(min=DEGEN))[m].mean()

    if a.loss_space == "raw":
        def train_loss(pred, y, x):
            return loss_fn(pred, y)
    else:
        train_loss = scale_free_loss
    print(f"\n  TRAINING OBJECTIVE: {a.loss_space}"
          + ("  (paper 2.1 verbatim)" if a.loss_space == "raw"
             else "  (deviation from paper 2.1 -- done.md 9)"))

    # ---- held-out probe and validation batches ---------------------------
    # NOT next(iter(dl)): that is training batch 0, so the "validation" loss
    # would be measured on data the model is fitting, the baselines would be
    # computed on seen windows, and the plateau decision -- "train until the
    # VALIDATION curve flattens" -- would be read off a train curve.
    #
    # DeterministicMix is a pure function of (ordinal, seed) over 2^40 ordinals.
    # Training consumes ordinals [start_step*B, steps*B); at B=512 and 1e5 steps
    # that is 5.1e7. HOLDOUT_BASE sits 2^35 = 3.4e10 ordinals away, so the two
    # bands cannot overlap for any run this project will do, and the band does
    # not move when the batch size does. Same seed/p_real/alpha in both arms =>
    # R0 and R1 are validated on exactly the same windows.
    HOLDOUT_BASE = 1 << 35
    ds = dl.dataset

    # Avec --augment, data.py 414-417 met return_ordinal = True des que
    # p_mixup > 0 (CFG le met a 1), donc ds[i] rend (x, y, ordinal). Le
    # 2 novembre AUG est mort ici en 25 s : ValueError, too many values to
    # unpack. Aucun bras n'avait jamais active l'augmentation, donc ce chemin
    # n'avait jamais tourne.
    #
    # Et ds[i] APPLIQUE l'augmentation, donc un held-out tire de ce flux ne
    # serait plus les memes fenetres que celui des autres bras -- ce qui detruit
    # la seule propriete pour laquelle HOLDOUT_BASE existe. On reconstruit donc
    # un flux simple, memes (seed, p_real, alpha, corpus), augment=None.
    if aug is not None:
        ds = make_loader(data_dir=a.data, real_dir=a.real, seq_len=2048,
                         horizon=48, batch_size=a.batch_size, seed=a.seed,
                         p_real=a.p_real, num_workers=0, verify="none",
                         real_alpha=a.real_alpha, augment=None).dataset
        print("  held-out: flux NON augmente (augment=None), pour rester "
              "comparable aux autres bras")

    def holdout(n: int):
        # Indexer, pas desempaqueter : ds[i] rend 2 ou 3 elements selon
        # return_ordinal, et seuls les deux premiers nous interessent.
        rows = [ds[HOLDOUT_BASE + i] for i in range(n)]
        return (torch.stack([r[0] for r in rows]).to(dev),
                torch.stack([r[1] for r in rows]).to(dev))

    assert a.steps * a.batch_size < HOLDOUT_BASE, \
        "training would run into the held-out ordinal band"
    probe_x, probe_y = holdout(8)
    vx, vy = holdout(a.val_size)
    print(f"  held-out: {vx.shape[0]} windows from ordinal {HOLDOUT_BASE:,} "
          f"(disjoint from training)")

    def scale_free(pred, y, x):
        """MAE divided pointwise by the window's own context range.

        WHY BOTH METRICS EXIST
        Reverso section 2.1 trains on MAE of the UNNORMALISED output, and
        decode() ends with `out * stdev + means` where stdev = x_range. Since
        those statistics are detached,

            |y_hat*s + m - y|  ==  s * |y_hat - (y-m)/s|

        pointwise -- the paper's loss is the scale-free loss weighted by each
        window's range s. With 43 sources spanning orders of magnitude, that
        makes the raw number dominated by whichever windows happen to have the
        largest range, which is why the held-out curve swung 586 -> 225 -> 586
        -> 64 on a FIXED set in the first smoke run.

        So: TRAIN on the paper's loss (unchanged, no deviation to declare),
        JUDGE on this one. The plateau decision and the R0-vs-R1 comparison are
        taken on the scale-free number because the raw one is too noisy to read.
        """
        rng = torch.clamp(x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0],
                          min=1e-5)
        m = ~torch.isnan(y)
        e = ((pred - y).abs() / rng)[m]
        return e.sum(), m.sum()

    def validate() -> tuple:
        """Held-out MAE on the last exit, raw AND scale-free, in chunks."""
        model.eval()
        s_raw = s_nrm = c = 0.0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for i in range(0, vx.shape[0], 256):
                xb, yb = vx[i:i + 256], vy[i:i + 256]
                pred = model(xb).float()
                m = ~torch.isnan(yb)
                s_raw += (pred - yb).abs()[m].sum().item()
                sn, cn = scale_free(pred, yb, xb)
                s_nrm += sn.item(); c += cn.item()
        model.train()
        return s_raw / max(c, 1), s_nrm / max(c, 1)

    with torch.no_grad():
        last = vx[:, -1:, :].expand(-1, H, -1)
        mean = vx.mean(1, keepdim=True).expand(-1, H, -1)
        drift = vx[:, -1:, :] + (vx[:, -1:, :] - vx[:, -2:-1, :]) * \
            torch.arange(1, H + 1, device=dev).view(1, H, 1)
        base, base_n = {}, {}
        for nm, pr in (("last", last), ("mean", mean), ("drift", drift)):
            base[nm] = loss_fn(pr, vy).item()
            sn, cn = scale_free(pr, vy, vx)
            base_n[nm] = (sn / max(cn, 1)).item()
    print(f"  baselines  raw        last {base['last']:.5f}   "
          f"mean {base['mean']:.5f}   drift {base['drift']:.5f}")
    print(f"             scale-free last {base_n['last']:.5f}   "
          f"mean {base_n['mean']:.5f}   drift {base_n['drift']:.5f}")
    print("  the model must get below these or it is not learning.")
    print("  JUDGE ON THE SCALE-FREE ROW -- the raw row is dominated by "
          "whichever windows have the largest range (see scale_free()).")

    # ---- logging ---------------------------------------------------------
    meta = {"args": vars(a), "params": n_par, "git": git_sha(),
            "syn_corpus_sha256": syn_hash, "real_corpus_sha256": real_hash,
            "stream_sha256": stream, "baselines": base,
            "augment": aug,
            "baselines_scale_free": base_n,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2)

    run = None
    if a.wandb:
        import wandb
        # Name from --out's basename, so a decay fork written to runs/R0-decay
        # is a distinguishable run rather than a second series with the same
        # name. group is shared, so the k curves still overlay on one chart --
        # that overlay is the project's headline figure (CLAUDE.md rule 9).
        run = wandb.init(project=a.wandb_project,
                         name=f"{os.path.basename(a.out.rstrip('/'))}"
                              f"-k{a.loop_k}-b{a.loop_beta}-s{a.seed}",
                         group=f"k-sweep-seed{a.seed}", config=meta,
                         dir=os.environ.get("WORK", "."))
        print(f"  wandb: {run.url}")

    csvf = open(f"{a.out}/log.csv", "a", newline="")
    csvw = csv.writer(csvf)
    if start == 0:
        csvw.writerow(["step", "loss", "loss_last_exit", "lr", "ema_loss",
                       "val_loss_raw", "val_loss_scalefree",
                       "plateau_ratio", "points_per_s", "grad_norm"])

    # ---- plateau tracking (task.md 7.1) ----------------------------------
    # EMA with a 500-step half-life. The raw curve at batch 512 is too noisy to
    # read; this is what makes the flattening visible.
    # The raw train curve at batch 512 is too noisy to read, hence the EMA.
    # The PLATEAU DECISION, though, is taken on the held-out curve (val_hist),
    # because the step count was defined as "train until the VALIDATION loss
    # flattens" -- and a train curve keeps drifting down on memorisation long
    # after generalisation has stopped improving.
    ema_alpha = 1.0 - 0.5 ** (1.0 / 500)
    ema_loss = None
    val_hist = deque(maxlen=4096)        # (step, held-out MAE)
    plateau_since = None
    ratio = float("nan")
    val = valn = float("nan")

    print("\n" + "=" * 70)
    print(f" training  {start} -> {a.steps}")
    print("=" * 70)

    model.train()
    it = iter(dl)
    t0 = time.time()
    run_loss = 0.0
    want_exits = a.exit_loss == "uniform" and a.loop_k > 1

    for step in range(start, a.steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

        with torch.autocast(dev, dtype=torch.bfloat16):
            out = model(x, return_all_exits=want_exits)
        if want_exits:
            obj, per_exit = multi_exit_loss(
                [o.float() for o in out], y, "uniform",
                loss_fn=lambda p, t: train_loss(p, t, x))
            last_exit = per_exit[-1]
        else:
            obj = train_loss(out.float(), y, x)
            last_exit, per_exit = obj, [obj]

        obj.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        ema.update(model, step)

        # The plateau decision is taken on L^(k), NOT on the averaged objective:
        # R1's 1/k sum of exits is not comparable to R0's single loss.
        lv = float(last_exit)
        ema_loss = lv if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * lv
        run_loss += float(obj)

        # ---- held-out curve + plateau test, on its own cadence ------------
        # Deliberately NOT nested inside the log block: the criterion must not
        # change when --log-every does.
        if (step + 1) % a.val_every == 0:
            t_probe = time.time()
            val, valn = validate()
            # The criterion reads the SCALE-FREE curve. The raw one swung
            # 586 -> 225 -> 586 -> 64 on a fixed set in the first smoke run;
            # a plateau test on that would fire or not fire at random.
            val_hist.append((step + 1, valn))
            old = [v for s, v in val_hist if s <= step + 1 - a.plateau_window]
            if old:
                ratio = (old[-1] - valn) / max(abs(valn), 1e-12)
                if ratio < a.plateau_tol:
                    plateau_since = plateau_since or step + 1
                else:
                    plateau_since = None
            print(f"  step {step+1:7d}  HELD-OUT raw {val:.5f}  "
                  f"scale-free {valn:.5f}"
                  + (f"   {a.plateau_window}-step gain {ratio:+.4f}"
                     f"{'  PLATEAU' if plateau_since else ''}"
                     if ratio == ratio else ""), flush=True)
            if run:
                d = {"val_loss_raw": val, "val_loss": valn}
                if ratio == ratio:
                    d["plateau_ratio"] = ratio
                run.log(d, step=step + 1)
            # Push t0 FORWARD by however long validation took, rather than
            # resetting it. Resetting it here made the log block -- which fires
            # on the same step whenever val_every and log_every coincide --
            # compute dt from a timer set microseconds earlier, and report
            # 74291 M pts/s. Shifting keeps the elapsed training time intact.
            t0 += time.time() - t_probe

        if (step + 1) % a.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            pps = a.log_every * a.batch_size * L / dt
            tr = run_loss / a.log_every
            lr = sched.get_last_lr()[0]

            print(f"  step {step+1:7d}  loss {tr:.5f}  L^k {lv:.5f}  "
                  f"ema {ema_loss:.5f}  lr {lr:.2e}  "
                  f"{pps/1e6:.2f} M pts/s  |g| {float(gn):.2f}"
                  + (f"  val* {valn:.5f}" if valn == valn else "")
                  + (f"  DROPPED {dropped[0]}/{dropped[1]}"
                     if dropped[0] else ""),
                  flush=True)
            csvw.writerow([step + 1, f"{tr:.6f}", f"{lv:.6f}", f"{lr:.3e}",
                           f"{ema_loss:.6f}", f"{val:.6f}", f"{valn:.6f}",
                           f"{ratio:.6f}", f"{pps:.0f}", f"{float(gn):.4f}"])
            csvf.flush()
            if run:
                d = {"loss": tr, "loss_last_exit": lv, "ema_loss": ema_loss,
                     "lr": lr, "points_per_s": pps, "grad_norm": float(gn)}
                for i, pe in enumerate(per_exit):
                    d[f"loss_exit/t{i+1}"] = float(pe)
                run.log(d, step=step + 1)
            run_loss = 0.0
            t0 = time.time()

        if (step + 1) % a.diag_every == 0:
            t_probe = time.time()
            diag = probe_diagnostics(model, probe_x, a.loop_k, a.loop_beta)
            if diag:
                hn = [v for k_, v in diag.items() if k_.startswith("h_norm/")]
                print(f"      diag fp32={diag.get('_fp32')}  "
                      f"|h| {hn[0]:.1f} -> {hn[-1]:.1f} "
                      f"(x{hn[-1]/max(hn[0],1e-9):.1f})  "
                      + "  ".join(f"{k_.split('/')[0]}[{k_.split('/')[1]}] {v:.3f}"
                                  for k_, v in diag.items()
                                  if k_.startswith(("update_magnitude/t1",
                                                    "cka/t0", "convergence/t1"))),
                      flush=True)
                if run:
                    run.log({f"diag/{k_}": v for k_, v in diag.items()
                             if isinstance(v, (int, float))}, step=step + 1)
            t0 += time.time() - t_probe   # shift, don't reset (see above)

        snap = a.snapshot_every and (step + 1) % a.snapshot_every == 0
        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps or snap:
            blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ema": ema.state_dict(),
                    "step": step + 1, "cfg": vars(cfg), "args": vars(a),
                    "syn_hash": syn_hash, "real_hash": real_hash,
                    "stream_hash": stream,
                    "val_loss": val, "val_loss_scale_free": valn,
                    "plateau_since": plateau_since}
            torch.save(blob, f"{a.out}/last.pth")
            if snap:
                torch.save(blob, f"{a.out}/step{step + 1}.pth")

    # ---- final: report on the EMA weights --------------------------------
    print("\n" + "=" * 70)
    with ema.averaged(model):
        ema_val, ema_valn = validate()
    raw_val, raw_valn = validate()
    print(f"  held-out ({vx.shape[0]} windows)")
    print(f"    raw        weights  {raw_val:.5f}   EMA weights  {ema_val:.5f}")
    print(f"    scale-free weights  {raw_valn:.5f}   EMA weights  {ema_valn:.5f}"
          f"   <- JUDGE ON THIS ROW")
    print(f"  baselines  raw        last {base['last']:.5f}  "
          f"mean {base['mean']:.5f}  drift {base['drift']:.5f}")
    print(f"             scale-free last {base_n['last']:.5f}  "
          f"mean {base_n['mean']:.5f}  drift {base_n['drift']:.5f}")
    print(f"  EVALUATION USES THE EMA WEIGHTS (see the header).")
    if plateau_since:
        print(f"  plateau criterion satisfied since step {plateau_since}")
    print(f"  stream_sha256 {stream}")
    csvf.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
