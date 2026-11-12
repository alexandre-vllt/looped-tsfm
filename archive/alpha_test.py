import sys, numpy as np
from reverso_torch.data import Corpus, make_loader, stream_hash, TEMPOPFN_WEIGHTS
SYN, REAL = sys.argv[1], sys.argv[2]
ok, bad = [], []
def chk(n, c, d=""):
    print(f"{'  ok   ' if c else '  FAIL '} {n}" + (f"   {d}" if d else "")); (ok if c else bad).append(n)

for a in (0.0, 0.5, 1.0):
    c = Corpus(REAL, None, verify="none", alpha=a)
    n = np.array(c.counts, dtype=float)
    # per-series exposure ratio between the biggest and smallest source
    exp = c.probs / n
    print(f"  alpha={a}: max/min per-series exposure ratio "
          f"{exp.max()/exp.min():8.1f}   largest source gets {100*c.probs.max():.1f}% of draws")

c0 = Corpus(REAL, None, verify="none", alpha=0.0)
c5 = Corpus(REAL, None, verify="none", alpha=0.5)
c1 = Corpus(REAL, None, verify="none", alpha=1.0)
n = np.array(c5.counts, dtype=float)
chk("alpha=0 is uniform", np.allclose(c0.probs, 1/len(n)))
chk("alpha=1 is proportional", np.allclose(c1.probs, n/n.sum()))
chk("alpha=0.5 is between", bool((c5.probs.max() < c1.probs.max()) and (c5.probs.max() > c0.probs.max())))
e0 = (c0.probs/n); e5 = (c5.probs/n)
# monotone in alpha: holds for ANY size distribution, so it does not depend
# on how wide the fixture happens to be. On the real 34-source GEP-L counts
# (spread 2083x) this takes 2083:1 down to 46:1.
e1 = (c1.probs/n)
chk("exposure imbalance falls monotonically with alpha",
    (e0.max()/e0.min()) >= (e5.max()/e5.min()) >= (e1.max()/e1.min()) - 1e-9,
    f"{e0.max()/e0.min():.1f} -> {e5.max()/e5.min():.1f} -> {e1.max()/e1.min():.1f}")

kw = dict(data_dir=SYN, real_dir=REAL, seq_len=2048, horizon=48,
          batch_size=32, p_real=0.5, verify="quick", real_alpha=0.5)
h = [stream_hash(make_loader(seed=0, num_workers=w, **kw), 8) for w in (0, 2, 8)]
chk("still deterministic across workers at alpha=0.5", len(set(h))==1, h[0][:16])
hd = stream_hash(make_loader(seed=0, num_workers=0, **{**kw, "real_alpha": 0.0}), 8)
chk("alpha changes the stream (so it must be recorded)", hd != h[0])
print()
print(f"FAILED: {bad}" if bad else f"all {len(ok)} checks passed")
sys.exit(1 if bad else 0)
