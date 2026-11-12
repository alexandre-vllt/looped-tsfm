"""
task.md 13.3 -- the test written first.

With loop_k = 1 and loop_beta = 1.0, LoopedModel.forward must produce output
BITWISE identical to the unmodified Model.forward on the released reverso_nano
checkpoint. If this fails, nothing built on top of the loop is valid: any
difference measured at k > 1 could be the looping or could be a bug introduced
while refactoring, and there would be no way to tell which.

Also checks, in order of how much they would hurt if wrong:

  2. gradients are bitwise identical too (a wrong backward pass trains,
     converges, and is silently wrong -- task.md 13.8)
  3. parameter count is unchanged (R0 and R1 must have identical capacity;
     the whole claim is "same parameters, 4x compute")
  4. k > 1 actually changes the output (the loop is not a no-op)
  5. beta interpolates as specified, and beta=0 is the exact identity
  6. multi-exit returns k predictions, and the last equals the single-exit call
  7. shapes survive k = 1..4 and the diagnostics run without error
  8. evaluating one model at several k costs no retraining (task.md 10)

CPU, float32, fixed seed. No GPU needed -- this is why it can run today.

    python test_loop_identity.py
"""
import argparse
import json
import os
import sys

import torch

from reverso_torch.model import Model
try:
    from reverso_torch.looped import (
        LoopedModel, loop_diagnostics, multi_exit_loss, reverso_loss)
except ImportError:                                   # looped.py at repo root
    from looped import (
        LoopedModel, loop_diagnostics, multi_exit_loss, reverso_loss)

CKPT_DIR = os.environ.get("NANO_CKPT", "checkpoints/reverso_nano")
SEED = 0
BATCH = 4          # small: the delta rule here is a python loop over 2048 steps
PASS, FAIL = "  ok   ", "  FAIL "
failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def load_configs(path):
    with open(os.path.join(path, "args.json")) as f:
        return argparse.Namespace(**json.load(f))


def build(cls, configs, state, **overrides):
    cfg = argparse.Namespace(**{**vars(configs), **overrides})
    m = cls(cfg)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fla", action="store_true",
                    help="run the identity checks THROUGH the fla kernels, on "
                         "cuda -- this is the path training actually uses")
    ap.add_argument("--mode", default="fused_recurrent",
                    choices=["fused_recurrent", "chunk"])
    args = ap.parse_args()

    # WHY --fla EXISTS
    # The CPU run proves the loop wrapper is a no-op at k=1, beta=1 against the
    # PYTHON delta rule. Training does not use the python delta rule -- it uses
    # fla, 204x faster (fast.py check 4). swap_to_fla rewrites the attention
    # blocks, so "the wrapper is a no-op" has to be re-proved on the rewritten
    # model, or R0-with-fla is simply a different model from the reference and
    # the k=1 arm is not a baseline for anything.
    #
    # fused_recurrent, not chunk: fla's chunked kernel asserts against float32,
    # and a bit-identity test in bf16 can pass by rounding two different numbers
    # to the same bf16 value. fused_recurrent computes the same recurrence in
    # fp32, so an exact match here means exact.
    dev, dtype = "cpu", torch.float32
    if args.fla:
        if not torch.cuda.is_available():
            sys.exit("--fla needs a GPU (fla kernels are triton/cuda)")
        dev = "cuda"
        if args.mode == "chunk":
            dtype = torch.bfloat16
            print("!! chunk mode forces bf16; identity in bf16 is weaker than "
                  "identity in fp32 (two distinct values can round together)")

    if not os.path.isdir(CKPT_DIR):
        sys.exit(f"checkpoint dir not found: {CKPT_DIR}\n"
                 f"run from the repo root, or set NANO_CKPT")

    configs = load_configs(CKPT_DIR)
    print(f"config: d_model={configs.d_model} seq_len={configs.seq_len} "
          f"main_module={configs.main_module!r} e_layers={configs.e_layers}")

    # Same loading rules as reverso_torch.forecast.load_checkpoint: unwrap the
    # container key, strip the DDP prefix, and drop the FlashFFTConv twiddle
    # buffers, which exist only in the original model.
    raw = torch.load(os.path.join(CKPT_DIR, "checkpoint.pth"),
                     map_location="cpu", weights_only=False)
    state = raw
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model", "ema", "ema_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if "flashfftconv." not in k}

    ref = build(Model, configs, state)
    loop = build(LoopedModel, configs, state, loop_k=1, loop_beta=1.0)

    if args.fla:
        from reverso_torch.fast import swap_to_fla
        ref = swap_to_fla(ref, args.mode)
        loop = swap_to_fla(loop, args.mode)
        print(f"fla swap applied to BOTH models, mode={args.mode}")
    ref = ref.to(dev, dtype=dtype).eval()
    loop = loop.to(dev, dtype=dtype).eval()
    print(f"running on {dev}, {str(dtype).removeprefix('torch.')}")

    # layer structure, so the spec's assumptions are verified not assumed
    print(f"layers: {[type(m).__name__ for m in ref.layers]}")
    attn = [m for m in ref.layers if type(m).__name__ == 'AttentionBlock']
    if attn:
        print(f"state_weaving active on attn blocks: "
              f"{[bool(a.state_weaving and a.is_intermediate) for a in attn]}"
              f"   (task.md 2: expected all False on nano)")

    # Generated on the CPU and then moved: identical bits on either device, so
    # the cpu and --fla runs are comparing the two models on the same input.
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, configs.seq_len, 1).to(dev, dtype=dtype)
    y = torch.randn(BATCH, configs.output_token_len, 1).to(dev, dtype=dtype)

    # ---- 1. bit-identity of the forward pass --------------------------------
    with torch.no_grad():
        y_ref = ref(x)
        y_loop = loop(x)

    same = torch.equal(y_ref, y_loop)
    check("1. forward bitwise identical at k=1, beta=1", same,
          f"max |diff| = {(y_ref - y_loop).abs().max().item():.3e}"
          if not same else f"shape {tuple(y_ref.shape)}")

    # ---- 2. bit-identity of the gradients ----------------------------------
    def grads(model):
        model.zero_grad(set_to_none=True)
        reverso_loss(model(x), y).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()
                if p.grad is not None}

    g_ref, g_loop = grads(ref), grads(loop)
    same_keys = set(g_ref) == set(g_loop)
    all_same = same_keys and all(torch.equal(g_ref[n], g_loop[n]) for n in g_ref)
    worst = ""
    if same_keys and not all_same:
        n, d = max(((n, (g_ref[n] - g_loop[n]).abs().max().item()) for n in g_ref),
                   key=lambda kv: kv[1])
        worst = f"worst: {n} {d:.3e}"
    check("2. gradients bitwise identical at k=1, beta=1", all_same,
          worst or f"{len(g_ref)} tensors")

    # ---- 3. parameter count unchanged --------------------------------------
    n_ref = sum(p.numel() for p in ref.parameters())
    n_loop = sum(p.numel() for p in loop.parameters())
    check("3. parameter count unchanged", n_ref == n_loop,
          f"{n_ref:,} params (nano, task.md says ~200 K)")

    # ---- 4. k > 1 changes the output ---------------------------------------
    with torch.no_grad():
        y4 = loop(x, loop_k=4, loop_beta=1.0)
    rel = ((y4 - y_ref).norm() / y_ref.norm()).item()
    check("4. k=4 differs from k=1 (loop is not a no-op)",
          not torch.equal(y4, y_ref), f"relative change {rel:.3f}")

    # ---- 5. beta behaves as specified --------------------------------------
    with torch.no_grad():
        y_b0 = loop(x, loop_k=4, loop_beta=0.0)
    check("5a. beta=0 is the exact identity (h never moves)",
          torch.equal(y_b0, loop(x, loop_k=1, loop_beta=0.0)),
          "any k gives the same output")

    with torch.no_grad():
        h, mu, sd = loop.encode(x)
        f_h = loop.core(h)
        for b in (0.25, 0.5):
            manual = b * f_h + (1.0 - b) * h
            got = loop(x, loop_k=1, loop_beta=b)
            ok = torch.equal(got, loop.decode(manual, mu, sd))
            check(f"5b. beta={b} matches beta*f(h)+(1-beta)*h", ok)

    # ---- 6. multi-exit ------------------------------------------------------
    with torch.no_grad():
        exits = loop(x, loop_k=4, loop_beta=0.25, return_all_exits=True)
        last = loop(x, loop_k=4, loop_beta=0.25)
    check("6a. return_all_exits gives k predictions", len(exits) == 4,
          f"got {len(exits)}")
    check("6b. last exit == single-exit call", torch.equal(exits[-1], last))

    obj_u, per = multi_exit_loss(exits, y, "uniform")
    obj_l, _ = multi_exit_loss(exits, y, "last")
    check("6c. uniform objective is the mean of the per-exit losses",
          torch.allclose(obj_u, torch.stack(per).mean()),
          f"L = {[round(float(p), 4) for p in per]}")
    check("6d. last-exit objective is L^(k)", torch.equal(obj_l, per[-1]))

    # ---- 7. shapes and diagnostics -----------------------------------------
    shapes_ok = True
    for k in (1, 2, 3, 4):
        with torch.no_grad():
            out = loop(x, loop_k=k, loop_beta=0.25)
        shapes_ok &= tuple(out.shape) == (BATCH, configs.output_token_len, 1)
    check("7a. output shape stable for k=1..4", shapes_ok,
          f"{(BATCH, configs.output_token_len, 1)}")

    with torch.no_grad():
        _, states = loop(x, loop_k=4, loop_beta=0.25, return_states=True)
        diag = loop_diagnostics(states)
    check("7b. diagnostics run and are finite",
          len(states["h"]) == 5 and len(states["f_h"]) == 4
          and all(v == v for v in diag.values() if isinstance(v, float)),
          f"{len(diag)} scalars")

    print("\n  diagnostics at k=4, beta=0.25 (untrained loop on a trained k=1 model):")
    for key in sorted(diag):
        if key in ("k", "beta"):
            continue
        print(f"    {key:<28} {diag[key]:.4f}")

    # ---- 8. one model, several k -------------------------------------------
    print("\n  test-time compute curve from the single released checkpoint "
          "(task.md 10):")
    with torch.no_grad():
        for k in (1, 2, 3, 4):
            l = reverso_loss(loop(x, loop_k=k, loop_beta=0.25), y)
            print(f"    k={k}  loss {float(l):.4f}")
    print("    (meaningless numbers -- random targets, and this checkpoint was "
          "trained at k=1. It proves the mechanism, not the result.)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed -- the loop is a verified no-op at k=1, "
          "so anything measured at k>1 is the looping.")


if __name__ == "__main__":
    main()
