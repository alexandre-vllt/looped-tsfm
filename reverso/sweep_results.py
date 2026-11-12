"""Read every run's log.csv and compare the arms. No GPU, no eval, seconds to run.

    source ~/env_home.sh && $VENV/bin/python sweep_results.py
    $VENV/bin/python sweep_results.py --baseline RAW     # compare against another arm

This is the HELD-OUT TRAINING number, not GIFT-Eval. It is the fast read; the
benchmark is a separate ~2 h job per arm (eval_one.sh).

WHY THE STREAM COLUMN IS HERE
On 12 October the sweep's RAW and EMA arms -- which change nothing in the data
pipeline -- printed a different stream hash from R0's 8bece5b4. That is unresolved.
Until it is, an arm is comparable to another arm with the SAME hash and to nothing
else, so the hash is printed next to every number rather than mentioned in a note
somebody has to remember.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st

VAL = "val_loss_scalefree"


def load(d):
    p = os.path.join(d, "log.csv")
    if not os.path.exists(p):
        return None
    steps, val, pps, gn = [], [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                s = int(float(r["step"]))
            except (KeyError, ValueError):
                continue
            steps.append(s)
            v = r.get(VAL, "")
            if v not in ("", "nan"):
                val.append((s, float(v)))
            for key, acc in (("points_per_s", pps), ("grad_norm", gn)):
                x = r.get(key, "")
                if x not in ("", "nan"):
                    try:
                        acc.append(float(x))
                    except ValueError:
                        pass
    if not steps:
        return None
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except Exception:
            pass
    last = max(steps)
    tail = [v for s, v in val if s > last - 10000]
    return dict(
        name=os.path.basename(d), last=last, n_val=len(val),
        final=val[-1][1] if val else float("nan"),
        tail_mean=st.mean(tail) if tail else float("nan"),
        tail_sd=st.pstdev(tail) if len(tail) > 1 else float("nan"),
        tail_n=len(tail),
        pps=st.median(pps) if pps else float("nan"),
        gn=st.median(gn) if gn else float("nan"),
        stream=(meta.get("stream_sha256") or "—")[:8],
        augment=bool(meta.get("augment")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline", default="R0")
    ap.add_argument("--target-steps", type=int, default=150000)
    a = ap.parse_args()

    rows = [r for r in (load(d) for d in sorted(glob.glob(a.runs + "/*"))
                        if os.path.isdir(d)) if r]
    if not rows:
        raise SystemExit(f"no log.csv under {a.runs}/*/")

    base = next((r for r in rows if r["name"] == a.baseline), None)

    print("=" * 96)
    print(" HELD-OUT scale-free MAE  (lower is better).  Final = last validation point.")
    print("=" * 96)
    print(f"  {'arm':<10}{'steps':>9}{'done':>7}{'final':>10}{'last 10k':>11}"
          f"{'sd':>9}{'vs ' + a.baseline:>10}{'M pts/s':>9}{'stream':>10}")
    for r in sorted(rows, key=lambda x: x["final"]):
        done = "yes" if r["last"] >= a.target_steps else f"{r['last']*100//a.target_steps}%"
        d = ""
        if base and r["name"] != a.baseline and base["final"] == base["final"]:
            pct = (base["final"] - r["final"]) / base["final"] * 100
            d = f"{pct:+.2f}%"
            if r["stream"] != base["stream"]:
                d += "*"
        print(f"  {r['name']:<10}{r['last']:>9,}{done:>7}{r['final']:>10.5f}"
              f"{r['tail_mean']:>11.5f}{r['tail_sd']:>9.5f}{d:>10}"
              f"{r['pps']/1e6:>9.2f}{r['stream']:>10}")

    streams = {}
    for r in rows:
        streams.setdefault(r["stream"], []).append(r["name"])
    print("\n  stream groups -- arms are comparable WITHIN a group, not across:")
    for h, names in sorted(streams.items(), key=lambda kv: -len(kv[1])):
        print(f"    {h:<10} {', '.join(sorted(names))}")
    if base and any(r["stream"] != base["stream"] for r in rows if r is not base):
        print(f"\n  * = different sample stream from {a.baseline}: the delta mixes the change"
              f"\n      under test with a change of data. Do not quote these as results.")

    unfinished = [r["name"] for r in rows if r["last"] < a.target_steps]
    if unfinished:
        print(f"\n  not at {a.target_steps:,} steps: {', '.join(unfinished)}")

    print("\n  Next, for the arms worth it: GIFT-Eval on each checkpoint")
    print("    runs/<arm>/step150000.pth  ->  example/eval_gift.py  (~2 h per arm)")


if __name__ == "__main__":
    main()
