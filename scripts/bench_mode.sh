#!/bin/bash
# Pick the DeltaNet mode at NANO'S REAL TRAINING CONFIG, not by compute
# capability -- and measure the actual cost of k=4 against k=1.
#
#   source ~/env_home.sh && use <free gpu> && bash bench_mode.sh
#
# WHY THIS EXISTS
# test_chunk_a100.sh showed `chunk` compiles on sm_80, but NOT that it is
# faster. At d_model 32 / batch 256 it was 0.73x -- fused_recurrent was 37 %
# FASTER. The reason is shape: nano has d_model 32 with 4 heads, so
# head_k_dim = 8, while chunked kernels are built around tensor-core MMA on
# 64x64 tiles. With 8-dim heads the tensor cores idle and the chunked algorithm
# does more total FLOPs -- it trades work for parallelism, which only pays when
# parallelism is scarce. At batch 256 fused_recurrent already has 256 x 4 =
# 1024 independent recurrences, enough to saturate an A100.
#
# Reverso trains at BATCH 512, further into that regime. So task.md section 0's
# claim -- "chunk should compile, expect a large speedup" -- is wrong for nano
# and the mode must be chosen by measurement.
#
# Section 2 needs our looped model on this host:
#   scp from the Mac into $WORK/sem_proj/reverso/reverso_torch/
#     reverso_torch/looped.py
# It is skipped automatically if absent.
set -u
: "${WORK:?source ~/env_home.sh first}"

REPO=$WORK/sem_proj/reverso
PY=$WORK/venvs/reverso/bin/python

echo "=============================================================="
echo " 1. DeltaNet mode at nano's batch sizes, FORWARD + BACKWARD"
echo "=============================================================="
echo "test_chunk_a100.sh measured forward only. Training is forward+backward,"
echo "and the backward of a sequential scan is also sequential -- so the"
echo "forward-only ranking need not hold. This measures what training pays."
echo

cd "$REPO" 2>/dev/null || cd "$HOME"
"$PY" - <<'PY'
import time, torch
from fla.layers import DeltaNet

L = 2048
print(f"{torch.cuda.get_device_properties(0).name}  torch {torch.__version__}\n")

def build(mode, d):
    return DeltaNet(mode=mode, d_model=d, expand_k=1.0, expand_v=1.0,
                    num_heads=4, use_beta=True, use_gate=False,
                    use_short_conv=True, conv_size=4, allow_neg_eigval=False,
                    qk_activation='silu', qk_norm='l2', layer_idx=0).cuda()

def step(lay, x):
    o = lay(hidden_states=x)
    o = o[0] if isinstance(o, tuple) else o
    o.float().pow(2).mean().backward()
    lay.zero_grad(set_to_none=True)

print(f"  {'d_model':>8}{'batch':>7}{'fused_rec':>13}{'chunk':>13}"
      f"{'speedup':>10}{'winner':>16}")
for d in (32, 64):
    for B in (64, 256, 512):
        res, oom = {}, {}
        for mode in ("fused_recurrent", "chunk"):
            try:
                lay = build(mode, d)
                x = torch.randn(B, L, d, device="cuda", dtype=torch.bfloat16,
                                requires_grad=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(3):
                        step(lay, x)
                torch.cuda.synchronize()
                t0 = time.time(); n = 8
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(n):
                        step(lay, x)
                torch.cuda.synchronize()
                res[mode] = n * B * L / (time.time() - t0)
            except torch.cuda.OutOfMemoryError:
                oom[mode] = True; res[mode] = float("nan")
            finally:
                for v in ("lay", "x"):
                    if v in dir(): pass
                torch.cuda.empty_cache()
        f, c = res["fused_recurrent"], res["chunk"]
        sp = c / f if f == f and c == c and f > 0 else float("nan")
        win = "fused_recurrent" if sp < 1 else "chunk"
        print(f"  {d:>8}{B:>7}{f/1e6:>10.2f} M{c/1e6:>10.2f} M"
              f"{sp:>9.2f}x{win:>16}"
              + ("   (OOM)" if oom else ""))

print("\n  Nano is d_model 32. Reverso's batch is 512.")
print("  Set the mode from THIS table, not from compute capability.")
PY

echo
echo "=============================================================="
echo " 2. full nano step: k=1 vs k=4"
echo "=============================================================="
if [ ! -f "$REPO/reverso_torch/looped.py" ]; then
    echo "  SKIPPED: $REPO/reverso_torch/looped.py not found."
    echo "  From the Mac:"
    echo "    scp ~/Desktop/sem_proj/reverso/reverso_torch/looped.py \\"
    echo "        agvallet@nala.epfl.ch:$REPO/reverso_torch/"
else
cd "$REPO"
PYTHONPATH="$REPO" "$PY" - <<'PY'
import argparse, json, time, torch
from reverso_torch.looped import LoopedModel, multi_exit_loss

cfg = argparse.Namespace(**json.load(open("checkpoints/reverso_nano/args.json")))
L, H = cfg.seq_len, cfg.output_token_len
print(f"nano: d_model {cfg.d_model}, seq {L} -> {H}\n")

def timed(k, beta, B, mode, exits):
    m = LoopedModel(argparse.Namespace(**{**vars(cfg), "loop_k": k,
                                          "loop_beta": beta})).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.1)
    x = torch.randn(B, L, 1, device="cuda")
    y = torch.randn(B, H, 1, device="cuda")

    def one():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, return_all_exits=exits)
        if exits:
            loss, _ = multi_exit_loss([o.float() for o in out], y, "uniform")
        else:
            loss = (out.float() - y).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    for _ in range(3): one()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); n = 8
    for _ in range(n): one()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 2**30
    del m, opt, x, y; torch.cuda.empty_cache()
    return dt, mem

B = 64          # 512 would OOM at k=4 on a shared GPU; ratios are what matter
print(f"  batch {B}, forward+backward+AdamW, bf16 autocast")
print(f"  {'run':>16}{'s/step':>10}{'pts/s':>12}{'peak GB':>10}{'vs k=1':>9}")
base = None
for name, k, beta, exits in (("R0  k=1", 1, 1.0, False),
                             ("R1  k=2 b.25", 2, 0.25, True),
                             ("R1  k=4 b.25", 4, 0.25, True),
                             ("R1  k=4 last", 4, 0.25, False)):
    dt, mem = timed(k, beta, B, None, exits)
    if base is None: base = dt
    print(f"  {name:>16}{dt:>10.3f}{B*L/dt/1e6:>9.2f} M{mem:>10.2f}"
          f"{dt/base:>8.2f}x")

print("\n  task.md section 1 assumes R1 costs ~4x R0. The ratio above is the real")
print("  one -- the loop re-runs layers[0:4] but the embedding, the head and the")
print("  cross-attention decoder run once, so it should be under 4x.")
print("  'k=4 last' skips three decoder calls, which is the cost of multi-exit.")
PY
fi
