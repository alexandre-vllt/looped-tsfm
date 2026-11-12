#!/bin/bash
# Does the chunked DeltaNet kernel compile on sm_80, and what does it buy?
#
#   source ~/env_nala.sh && use <free-gpu> && bash test_chunk_a100.sh
#
# On izar (V100, sm_70) `chunk` died in ConvertTritonGPUToLLVM because Triton
# has no Volta lowering for the tensor-core MMA it emits, so reverso_fast was
# pinned to `fused_recurrent` -- 2048 sequential steps instead of 32. These are
# A100s at sm_80, so chunk should compile. This measures what that is worth.
set -u
python - <<'PY'
import time, torch

print("torch", torch.__version__, "| devices", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print(f"running on {p.name}  sm_{p.major}{p.minor}\n")

try:
    from fla.layers import DeltaNet
except Exception as e:
    raise SystemExit(f"cannot import fla.layers.DeltaNet: {e}")

def build(mode, d_model, heads=4):
    # exactly Reverso's AttentionBlock configuration
    return DeltaNet(
        mode=mode, d_model=d_model, expand_k=1.0, expand_v=1.0, num_heads=heads,
        use_beta=True, use_gate=False, use_short_conv=True, conv_size=4,
        allow_neg_eigval=False, qk_activation='silu', qk_norm='l2', layer_idx=0,
    ).cuda().bfloat16().eval()

def run(layer, x):
    with torch.no_grad():
        o = layer(hidden_states=x)
    return o[0] if isinstance(o, tuple) else o

L = 2048
torch.manual_seed(0)

# ---------- 1. does chunk compile at all? -------------------------------
print("=" * 66)
print("1. COMPILATION")
ok = {}
for mode in ("fused_recurrent", "chunk"):
    try:
        lay = build(mode, 64)
        x = torch.randn(2, L, 64, device="cuda", dtype=torch.bfloat16)
        run(lay, x)
        torch.cuda.synchronize()
        ok[mode] = True
        print(f"  {mode:16s} OK")
    except Exception as e:
        ok[mode] = False
        print(f"  {mode:16s} FAILED -- {type(e).__name__}: {str(e)[:160]}")

if not ok.get("chunk"):
    raise SystemExit("\nchunk still does not compile. Nothing below applies.")

# ---------- 2. do the two modes agree? ----------------------------------
print("\n" + "=" * 66)
print("2. NUMERICAL AGREEMENT  (same weights, same input)")
torch.manual_seed(0); a = build("fused_recurrent", 64)
torch.manual_seed(0); b = build("chunk", 64)
b.load_state_dict(a.state_dict())
x = torch.randn(4, L, 64, device="cuda", dtype=torch.bfloat16)
ya, yb = run(a, x).float(), run(b, x).float()
corr = torch.corrcoef(torch.stack([ya.flatten(), yb.flatten()]))[0, 1].item()
rel = ((ya - yb).norm() / ya.norm()).item()
print(f"  correlation      {corr:.6f}")
print(f"  relative L2 diff {rel:.2e}")
print("  (same recurrence, different association order -> bf16 rounding only)")

# ---------- 3. what does it buy? ----------------------------------------
print("\n" + "=" * 66)
print("3. THROUGHPUT   tokens/s, forward only, bf16, L=2048")
print(f"  {'d_model':>8} {'batch':>6} {'fused_rec':>14} {'chunk':>14} {'speedup':>9}")
for d_model in (32, 64):          # reverso_nano, reverso_small
    for B in (16, 64, 256):
        res = {}
        for mode in ("fused_recurrent", "chunk"):
            lay = build(mode, d_model)
            x = torch.randn(B, L, d_model, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                run(lay, x)
            torch.cuda.synchronize()
            t0 = time.time()
            n = 10
            for _ in range(n):
                run(lay, x)
            torch.cuda.synchronize()
            res[mode] = n * B * L / (time.time() - t0)
            del lay, x
            torch.cuda.empty_cache()
        sp = res["chunk"] / res["fused_recurrent"]
        print(f"  {d_model:>8} {B:>6} {res['fused_recurrent']/1e6:>11.2f} M "
              f"{res['chunk']/1e6:>11.2f} M {sp:>8.2f}x")

print("\n" + "=" * 66)
print("If chunk is faster, reverso_fast must stop hardcoding fused_recurrent:")
print("    mode = 'chunk' if torch.cuda.get_device_capability()[0] >= 8 "
      "else 'fused_recurrent'")
print("Carrying the V100 default to an A100 leaves this speedup unused.")
PY
