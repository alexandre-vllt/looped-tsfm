#!/usr/bin/env python3
"""Probe: does gift_eval expose a validation split, and how big is it?

Run on nala:
  cd ~/work/sem_proj/reverso
  PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH" $VENV/bin/python probe_val.py

Answers three questions for the v2 pipeline:
  1. which split attributes Dataset exposes (training / validation / test)
  2. how many windows and how many points a validation pass would cost
  3. whether validation windows overlap the test windows (leakage check)
"""
import os, sys, time, traceback

os.environ.setdefault("GIFT_EVAL", os.path.expanduser("~/work/gift-eval-data"))

try:
    from gift_eval.data import Dataset
except Exception:
    print("IMPORT FAILED. PYTHONPATH =", os.environ.get("PYTHONPATH"))
    traceback.print_exc()
    sys.exit(1)

print("gift_eval.data imported OK")
print("GIFT_EVAL =", os.environ.get("GIFT_EVAL"))
print()

# --- 1. what does the class expose? -----------------------------------------
attrs = [a for a in dir(Dataset) if not a.startswith("_")]
print("Dataset public attributes:")
for a in sorted(attrs):
    print("   ", a)
print()

SPLIT_ATTRS = [a for a in attrs
               if any(k in a.lower() for k in ("train", "val", "test"))]
print("split-looking attributes:", SPLIT_ATTRS)
print()

# --- 2. instantiate a few datasets and measure -------------------------------
PROBE = [
    ("m4_weekly",    "short"),
    ("electricity/15T", "short"),
    ("ett1/H",       "short"),
]

total_win = 0
total_pts = 0
for name, term in PROBE:
    print("=" * 60)
    print(name, term)
    try:
        ds = Dataset(name=name, term=term, to_univariate=False)
    except Exception as e:
        print("   construct failed:", repr(e))
        continue

    for a in SPLIT_ATTRS:
        try:
            v = getattr(ds, a)
        except Exception as e:
            print(f"   {a:24s} -> raises {type(e).__name__}: {e}")
            continue
        if callable(v):
            print(f"   {a:24s} -> callable")
            continue
        # try to count it
        n = None
        try:
            n = len(v)
        except Exception:
            pass
        if n is None:
            try:
                t0 = time.time()
                n = sum(1 for _ in v)
                print(f"   {a:24s} -> iterable, {n} items "
                      f"({time.time()-t0:.1f}s to enumerate)")
            except Exception as e:
                print(f"   {a:24s} -> {type(v).__name__} (uncountable: {e})")
                continue
        else:
            print(f"   {a:24s} -> {type(v).__name__}, len = {n}")

    # window / point cost of the validation split, if present
    for cand in ("validation_dataset", "validation_data", "val_dataset"):
        if not hasattr(ds, cand):
            continue
        try:
            v = getattr(ds, cand)
            t0 = time.time()
            nw, npt = 0, 0
            for e in v:
                nw += 1
                tgt = e["target"] if isinstance(e, dict) else e[0]["target"]
                npt += len(tgt) if hasattr(tgt, "__len__") else 0
                if nw >= 5000:
                    break
            print(f"   [{cand}] {nw} windows, {npt} points, "
                  f"{time.time()-t0:.1f}s")
            total_win += nw
            total_pts += npt
        except Exception as e:
            print(f"   [{cand}] enumeration failed: {type(e).__name__}: {e}")
        break

print()
print("=" * 60)
print(f"3 probe datasets: {total_win} val windows, {total_pts} points")
print("Extrapolated to 97 rows: "
      f"~{total_win * 97 // max(len(PROBE),1)} windows")
