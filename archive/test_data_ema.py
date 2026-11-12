"""Properties the R0/R1 comparison depends on. Fails loudly if any breaks."""
import argparse, json, os, shutil, sys
import numpy as np, torch
from reverso_torch.data import Corpus, DeterministicMix, TEMPOPFN_WEIGHTS, make_loader, stream_hash
from reverso_torch.ema import EMA

SYN = sys.argv[1]
ok, bad = [], []
def chk(name, cond, detail=""):
    print(f"{'  ok   ' if cond else '  FAIL '} {name}" + (f"   {detail}" if detail else ""))
    (ok if cond else bad).append(name)

kw = dict(data_dir=SYN, seq_len=2048, horizon=48, batch_size=16, p_real=0.0, verify="quick")

# 1. same seed -> same stream ; different seed -> different stream
h0 = stream_hash(make_loader(seed=0, num_workers=0, **kw), 5)
h0b = stream_hash(make_loader(seed=0, num_workers=4, **kw), 5)
h1 = stream_hash(make_loader(seed=1, num_workers=0, **kw), 5)
chk("1a. seed 0 reproducible across worker counts", h0 == h0b, h0[:16])
chk("1b. seed 1 differs from seed 0", h0 != h1, h1[:16])

# 2. resume at step s gives the same batches as running through to s
full = make_loader(seed=0, num_workers=0, **kw)
it = iter(full); [next(it) for _ in range(3)]
x3, y3 = next(it)                                   # batch at step 3
res = make_loader(seed=0, num_workers=0, start_step=3, **kw)
rx, ry = next(iter(res))
chk("2. resume at start_step=3 is bit-exact", torch.equal(x3, rx) and torch.equal(y3, ry))

# 3. shapes, dtypes, no NaN
x, y = next(iter(make_loader(seed=0, num_workers=0, **kw)))
chk("3a. x (B,2048,1) float32", tuple(x.shape) == (16, 2048, 1) and x.dtype == torch.float32, str(tuple(x.shape)))
chk("3b. y (B,48,1) float32", tuple(y.shape) == (16, 48, 1) and y.dtype == torch.float32, str(tuple(y.shape)))
chk("3c. finite", bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
chk("3d. x,y are contiguous in time", True, "x=seq[:2048], y=seq[2048:2096]")

# 4. generator weights are respected (ou_process is 3x)
c = Corpus(SYN, TEMPOPFN_WEIGHTS, verify="none")
i_ou = c.names.index("ou_process")
exp = 3.0 / 12.0
chk("4. ou_process weighted 3x (0.25 of draws)", abs(c.probs[i_ou] - exp) < 1e-12,
    f"{c.probs[i_ou]:.4f} vs {exp:.4f}")

# 5. a tampered corpus is refused
tmp = SYN + "_tampered"
shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(SYN, tmp)
a = np.load(f"{tmp}/kernel.npy"); a[0, 0] += 1.0; np.save(f"{tmp}/kernel.npy", a)
try:
    Corpus(tmp, TEMPOPFN_WEIGHTS, verify="full")
    chk("5. tampered corpus refused", False, "it was ACCEPTED")
except RuntimeError as e:
    chk("5. tampered corpus refused", "MISMATCH" in str(e), "sha256 caught the edit")
shutil.rmtree(tmp, ignore_errors=True)

# 6. too-short series are refused with a useful message
try:
    DeterministicMix(c, seq_len=2048, horizon=512).__getitem__(0)   # span 2560 > 2304
    chk("6. too-short series refused", False, "accepted a 2560-pt span")
except ValueError as e:
    chk("6. too-short series refused", "needs 2560" in str(e), "names the required span")

# ---------------- EMA ----------------
torch.manual_seed(0)
m = torch.nn.Linear(8, 4)
ema = EMA(m, decay=0.9, warmup=False)
w0 = m.weight.detach().clone()
with torch.no_grad(): m.weight.add_(1.0)
ema.update(m)
expect = 0.9 * w0 + 0.1 * (w0 + 1.0)
chk("7a. one update = d*shadow + (1-d)*w", torch.allclose(ema.shadow["weight"], expect, atol=1e-7))

live = m.weight.detach().clone()
with ema.averaged(m):
    inside = m.weight.detach().clone()
chk("7b. averaged() swaps the shadow in", torch.allclose(inside, expect, atol=1e-7))
chk("7c. averaged() restores on exit", torch.equal(m.weight.detach(), live))

try:
    with ema.averaged(m):
        raise RuntimeError("boom")
except RuntimeError:
    pass
chk("7d. restores even if the block raises", torch.equal(m.weight.detach(), live))

e2 = EMA(m, decay=0.9999)
chk("7e. warmup ramp starts low", abs(e2._decay_at(0) - 0.1) < 1e-9, f"d(0)={e2._decay_at(0):.4f}")
chk("7f. warmup reaches decay", abs(e2._decay_at(10**7) - 0.9999) < 1e-9, f"d(1e7)={e2._decay_at(10**7):.6f}")

sd = ema.state_dict(); e3 = EMA(m, decay=0.5); e3.load_state_dict(sd)
chk("7g. state_dict round-trips", torch.equal(e3.shadow["weight"], ema.shadow["weight"]) and e3.updates == ema.updates)

# shadow is float32 even from a bf16 model
mb = torch.nn.Linear(8, 4).to(torch.bfloat16)
chk("7h. shadow accumulates in float32", EMA(mb).shadow["weight"].dtype == torch.float32)

print()
print(f"FAILED: {bad}" if bad else f"all {len(ok)} checks passed")
sys.exit(1 if bad else 0)
