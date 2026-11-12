"""Aggregate the GIFT-Eval runs into one comparison, and draw the k-curve.

Reads every results/<tag>/all_results_*.csv that eval_all.sh produced and emits
  - a comparison table over the datasets ALL arms share
  - the test-time compute curve: one R1 checkpoint scored at k = 1,2,3,4,6,8

THE ALIGNMENT RULE, which is the whole reason this is a script and not a
spreadsheet: arms are compared ONLY on the dataset/term rows that every arm
produced. If one arm crashed on a dataset, including it for the others would
silently change what the mean is over, and a 1-2 % difference between arms is
exactly the size of error that would create. The script prints how many rows it
dropped and why.

MASE is the headline: it is already scale-free per dataset (error over the
seasonal-naive error), so averaging it across datasets is meaningful, which is
not true of MAE or MSE. The geometric mean is reported alongside the arithmetic
one because MASE is a ratio -- the arithmetic mean of ratios is dominated by the
worst dataset.

    python aggregate_eval.py --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st

MASE = "eval_metrics/MASE[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
ND = "eval_metrics/ND[0.5]"


def load_tag(d: str) -> dict:
    """Newest all_results_*.csv in a tag directory -> {dataset_row: metrics}."""
    files = sorted(glob.glob(os.path.join(d, "all_results_*.csv")))
    if not files:
        return {}
    out = {}
    with open(files[-1]) as f:
        for r in csv.DictReader(f):
            key = r.get("dataset")
            if not key:
                continue
            try:
                out[key] = {"mase": float(r[MASE]), "crps": float(r[CRPS]),
                            "nd": float(r[ND]),
                            "domain": r.get("domain", "")}
            except (KeyError, ValueError):
                continue
    return out


def geo(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def parse_k(tag: str):
    m = re.search(r"_k(\d+)$", tag)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="eval_comparison.csv")
    p.add_argument("--figure", default="testtime_compute.png")
    p.add_argument("--baseline", default="R0_k1",
                   help="arm every other arm is compared against. R0_k1 is the "
                        "original run; for the 13 Oct sweep the honest baseline "
                        "is BASE, which shares its sample stream (done.md 10).")
    a = p.parse_args()

    tags = {}
    for d in sorted(glob.glob(os.path.join(a.results_dir, "*"))):
        if not os.path.isdir(d):
            continue
        rows = load_tag(d)
        if rows:
            tags[os.path.basename(d)] = rows

    if not tags:
        raise SystemExit(f"no all_results_*.csv under {a.results_dir}/*/")

    print(f"{len(tags)} arms found\n")
    for t, r in sorted(tags.items()):
        print(f"  {t:<12} {len(r):>3} dataset/term rows")

    common = set.intersection(*(set(r) for r in tags.values()))
    allrows = set().union(*(set(r) for r in tags.values()))
    dropped = sorted(allrows - common)
    print(f"\n  {len(common)} rows common to every arm"
          + (f"; DROPPED {len(dropped)} not present everywhere:" if dropped else ""))
    for d in dropped[:12]:
        have = [t for t in sorted(tags) if d in tags[t]]
        print(f"    {d:<40} only in {','.join(have)}")
    if len(dropped) > 12:
        print(f"    ... and {len(dropped)-12} more")
    if not common:
        raise SystemExit("no rows shared by all arms -- cannot compare")

    keys = sorted(common)
    summary = {}
    for t, rows in tags.items():
        mase = [rows[k]["mase"] for k in keys]
        crps = [rows[k]["crps"] for k in keys]
        nd = [rows[k]["nd"] for k in keys]
        summary[t] = {
            "n": len(keys),
            "mase_geo": geo(mase), "mase_mean": st.mean(mase),
            "mase_wins": None,
            "crps": st.mean(crps), "nd": st.mean(nd),
        }

    base = a.baseline
    if base not in summary:
        print(f"\n  !! baseline '{base}' has no results; "
              f"available: {', '.join(sorted(summary))}")
        print("     the 'vs' column and the wins column will be blank.")
    if base in summary:
        for t, rows in tags.items():
            if t == base:
                continue
            summary[t]["mase_wins"] = sum(
                1 for k in keys if rows[k]["mase"] < tags[base][k]["mase"])

    print(f"\n{'='*78}\n COMPARISON over {len(keys)} shared dataset/term rows\n{'='*78}")
    print(f"  {'arm':<12}{'MASE geo':>10}{'MASE mean':>11}{'CRPS':>9}{'ND':>8}"
          f"{'vs ' + base:>11}{'wins':>8}")
    ref = summary.get(base, {}).get("mase_geo")
    for t in sorted(summary, key=lambda x: (x.split("_k")[0], parse_k(x) or 0)):
        s = summary[t]
        d = (f"{(ref - s['mase_geo'])/ref*100:+.2f} %"
             if ref and t != base else ("—" if t == base else ""))
        w = f"{s['mase_wins']}/{len(keys)}" if s["mase_wins"] is not None else "—"
        print(f"  {t:<12}{s['mase_geo']:>10.4f}{s['mase_mean']:>11.4f}"
              f"{s['crps']:>9.4f}{s['nd']:>8.4f}{d:>11}{w:>8}")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "k", "n_rows", "MASE_geomean", "MASE_mean",
                    "CRPS_mean", "ND_mean", f"pct_better_than_{base}",
                    f"rows_better_than_{base}"])
        for t in sorted(summary):
            s = summary[t]
            d = ((ref - s["mase_geo"]) / ref * 100) if ref else float("nan")
            w.writerow([t, parse_k(t), s["n"], f"{s['mase_geo']:.5f}",
                        f"{s['mase_mean']:.5f}", f"{s['crps']:.5f}",
                        f"{s['nd']:.5f}",
                        "" if t == base else f"{d:.3f}",
                        "" if s["mase_wins"] is None else s["mase_wins"]])
    print(f"\n  wrote {a.out}")

    # ---- per-dataset detail, so a single dataset cannot drive the headline --
    det = a.out.replace(".csv", "_per_dataset.csv")
    arms = sorted(tags)
    with open(det, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + [f"MASE_{t}" for t in arms])
        for k in keys:
            w.writerow([k] + [f"{tags[t][k]['mase']:.5f}" for t in arms])
    print(f"  wrote {det}")

    # ---- the k curve ------------------------------------------------------
    curve = sorted(((parse_k(t), summary[t]["mase_geo"], t)
                    for t in summary if t.startswith("R1_k") and parse_k(t)),
                   key=lambda x: x[0])
    ctrl = sorted(((parse_k(t), summary[t]["mase_geo"], t)
                   for t in summary if t.startswith("R0_k") and parse_k(t)),
                  key=lambda x: x[0])
    if len(curve) < 2:
        print("\n  (fewer than 2 R1 k-points: skipping the figure)")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980"
    R1C, R0C = "#eb6834", "#2a78d6"       # validated categorical slots 2 and 1

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d1")
    ax.tick_params(colors=INK2, labelsize=9, length=0)

    ks = [c[0] for c in curve]; vs = [c[1] for c in curve]
    ax.plot(ks, vs, "-o", color=R1C, lw=2.2, ms=8, mec=SURFACE, mew=2,
            label="R1 — trained at k=4, β=0.25", zorder=4)
    if ctrl:
        ax.plot([c[0] for c in ctrl], [c[1] for c in ctrl], "-s", color=R0C,
                lw=2.0, ms=7, mec=SURFACE, mew=2, zorder=3,
                label="R0 — trained at k=1 (control)")
    if base in summary:
        ax.axhline(summary[base]["mase_geo"], color=MUTED, lw=1.2,
                   ls=(0, (4, 3)), zorder=1)
        ax.text(min(ks), summary[base]["mase_geo"], "R0 at k=1 ",
                fontsize=8.5, color=MUTED, va="bottom", ha="left")
    # mark the k it was trained at
    if 4 in ks:
        ax.annotate("trained here", (4, vs[ks.index(4)]), xytext=(0, -26),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xticks(ks)
    ax.set_xlabel("loop iterations at EVALUATION time  (k)", fontsize=10, color=INK2)
    ax.set_ylabel("GIFT-Eval MASE  (geometric mean)", fontsize=10, color=INK2)
    ax.set_title("One checkpoint, scored at several depths\n"
                 f"weight-tied block, {len(keys)} GIFT-Eval dataset/term rows",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.20)
    fig.text(0.012, 0.03,
             "Lower is better. MASE < 1 beats the seasonal-naive baseline. "
             "No retraining: the loop is weight-tied,\nso k at eval is free to "
             "differ from k at training (task.md §10). seed 0, one run per arm.",
             fontsize=7.5, color=MUTED)
    fig.savefig(a.figure, dpi=220, facecolor=SURFACE)
    print(f"  wrote {a.figure}")


if __name__ == "__main__":
    main()
