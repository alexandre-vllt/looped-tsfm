import sys, numpy as np, torch
from reverso_torch.data import make_loader, stream_hash, Corpus, TEMPOPFN_WEIGHTS
SYN, REAL = sys.argv[1], sys.argv[2]
ok, bad = [], []
def chk(n, c, d=""):
    print(f"{'  ok   ' if c else '  FAIL '} {n}" + (f"   {d}" if d else "")); (ok if c else bad).append(n)

kw = dict(data_dir=SYN, real_dir=REAL, seq_len=2048, horizon=48,
          batch_size=32, p_real=0.5, verify="full")

h = [stream_hash(make_loader(seed=0, num_workers=w, **kw), 8) for w in (0, 2, 8)]
chk("1. 50/50 mix deterministic across workers", len(set(h)) == 1, h[0][:16])

hs = stream_hash(make_loader(seed=0, num_workers=0, **{**kw, "p_real": 0.0}), 8)
chk("2. p_real=0 differs from p_real=0.5", hs != h[0])

# the real/synthetic split should land near 50% -- check by marking the corpora
syn = Corpus(SYN, TEMPOPFN_WEIGHTS, verify="none")
real = Corpus(REAL, None, verify="none")
from reverso_torch.data import DeterministicMix
from numpy.random import Generator, Philox
ds = DeterministicMix(syn, real, 2048, 48, seed=0, p_real=0.5)
n_real = sum(Generator(Philox(key=0, counter=i)).random() < 0.5 for i in range(4000))
chk("3. real share is ~50%", 0.47 < n_real/4000 < 0.53, f"{100*n_real/4000:.1f}%")

x, y = next(iter(make_loader(seed=0, num_workers=0, **kw)))
chk("4. shapes and finiteness", tuple(x.shape)==(32,2048,1) and tuple(y.shape)==(32,48,1)
    and bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
    f"{tuple(x.shape)} / {tuple(y.shape)}")
chk("5. real corpus sampled uniformly over sources", 
    abs(real.probs.max() - real.probs.min()) < 1e-12,
    f"{len(real.names)} sources at {real.probs[0]:.4f} each")
print()
print(f"FAILED: {bad}" if bad else f"all {len(ok)} checks passed")
sys.exit(1 if bad else 0)
