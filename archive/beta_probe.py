"""How fast does ||h|| grow under the loop, as a function of beta?

task.md 3.1 / 14: beta=1 is the bare loop h <- f(h), which [05] predicts blows
up under weight tying because tied increments are correlated (Theta(k^2) norm
growth, not Theta(k)). This measures it on the released nano checkpoint.
"""
import argparse, json, torch
from reverso_torch.model import Model
from reverso_torch.looped import LoopedModel

cfg = argparse.Namespace(**json.load(open("checkpoints/reverso_nano/args.json")))
raw = torch.load("checkpoints/reverso_nano/checkpoint.pth", map_location="cpu", weights_only=False)
st = raw
if isinstance(raw, dict):
    for k in ("model_state_dict","state_dict","model","ema","ema_state_dict"):
        if k in raw and isinstance(raw[k], dict): st = raw[k]; break
st = {k.removeprefix("module."): v for k,v in st.items() if "flashfftconv." not in k}

m = LoopedModel(argparse.Namespace(**{**vars(cfg), "loop_k":4, "loop_beta":1.0}))
m.load_state_dict(st, strict=True); m.eval()

torch.manual_seed(0)
x = torch.randn(4, cfg.seq_len, 1)

print(f"{'beta':>6} | " + " | ".join(f"||h^{t}||" .rjust(10) for t in range(5)))
print("-"*78)
for beta in (1.0, 0.5, 0.25):
    with torch.no_grad():
        _, s = m(x, loop_k=4, loop_beta=beta, return_states=True)
    norms = [float(h.norm()) for h in s["h"]]
    print(f"{beta:>6} | " + " | ".join(f"{n:10.2f}" for n in norms)
          + f"   growth x{norms[-1]/norms[0]:.1f}")
